import warnings
from collections import defaultdict

import torch
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSliceSampler, SliceSampler


class Buffer:
    def __init__(self, config, sampler=None):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.mirror = bool(getattr(config, "mirror", False))
        self.mirror_opp_warm_start = bool(getattr(config, "mirror_opp_warm_start", False))
        self._original_batch_size = None
        self.num_eps = 0
        if sampler is None:
            sampler = SliceSampler(
                num_slices=self.batch_size, end_key=None, traj_key="episode", truncated_key=None, strict_length=True
            )
        self._buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=config.max_size, device=self.storage_device, ndim=2),
            sampler=sampler,
            prefetch=0,
            batch_size=self.batch_size * (self.batch_length + 1),  # +1 for context
        )

    def add_transition(self, data):
        # This is batched data and lifted for storage.
        # (B, ...) -> (B, 1, ...)
        self._buffer.extend(data.unsqueeze(1))

    def sample(self):
        try:
            sample_td, info = self._buffer.sample(return_info=True)
        except RuntimeError as e:
            if "sufficient length" in str(e):
                warnings.warn(f"Replay buffer sample skipped: {e}")
                return None
            raise
        # The sampler returns a flattened batch of length B*(T+1).
        # (B*(T+1), ...) -> (B, T+1, ...)
        sample_td = sample_td.view(-1, self.batch_length + 1)
        # Stash the sampled episode IDs for downstream logging.
        self.last_sampled_episodes = sample_td["episode"][:, 0].tolist()
        src_dev = sample_td.device
        if src_dev.type == "cpu" and self.device.type == "cuda":
            sample_td = sample_td.pin_memory().to(self.device, non_blocking=True)
        elif src_dev != self.device:
            sample_td = sample_td.to(self.device, non_blocking=True)
        # The initial ones are used only to extract the latent vector
        initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
        opp_initial = None
        if "opp_stoch" in sample_td.keys():
            opp_initial = (sample_td["opp_stoch"][:, 0], sample_td["opp_deter"][:, 0])
        data = sample_td[:, 1:]
        data.set_("action", sample_td["action"][:, :-1])  # action is 1 step back
        if "opponent_action" in sample_td.keys():
            data.set_("opponent_action", sample_td["opponent_action"][:, :-1])
        index = [ind.view(-1, self.batch_length + 1)[:, 1:] for ind in info["index"]]

        self._original_batch_size = data.shape[0]
        if self.mirror and "opponent_reward" in data.keys():
            data, initial, opp_initial = self._mirror_trajectories(data, initial, opp_initial)

        return data, index, initial, opp_initial

    def _mirror_trajectories(self, data, initial, opp_initial=None):
        """Swap player/opponent perspectives to create mirrored copies.

        Doubles the batch: ``[original_0..B-1, mirrored_0..B-1]``.

        When ``mirror_opp_warm_start`` is False (default), mirrored
        trajectories get ``is_first[:, 0] = True`` and zero RSSM initial
        states so the RSSM re-initialises from its learned prior.

        When ``mirror_opp_warm_start`` is True **and** opponent RSSM states
        are available, mirrored trajectories use the opponent's t=0 state as
        the player initial (warm-start), the player's stored per-timestep
        states as the opponent imagination states, and ``is_first[:, 0]`` is
        NOT forced True so the warm initial survives the first ``obs_step``.
        """
        mirror = data.clone()

        # Swap state observations
        mirror.set_("policy", data["opponent"].clone())
        mirror.set_("opponent", data["policy"].clone())

        # Swap images
        if "image" in data.keys() and "opponent_image" in data.keys():
            mirror.set_("image", data["opponent_image"].clone())
            mirror.set_("opponent_image", data["image"].clone())

        # Swap actions
        if "opponent_action" in data.keys():
            mirror.set_("action", data["opponent_action"].clone())
            mirror.set_("opponent_action", data["action"].clone())

        # Swap rewards
        mirror.set_("reward", data["opponent_reward"].clone())
        mirror.set_("opponent_reward", data["reward"].clone())

        # --- Opponent RSSM states for imagination ---
        has_opp = "opp_stoch" in data.keys() and opp_initial is not None
        if has_opp and self.mirror_opp_warm_start:
            # Mirrored "opponent" was the original player → use player's
            # stored per-timestep states as the imagination opponent states.
            mirror.set_("opp_stoch", data["stoch"].clone())
            mirror.set_("opp_deter", data["deter"].clone())
        elif has_opp:
            # Zero opponent states for mirrored half (backward-compatible).
            mirror.set_("opp_stoch", torch.zeros_like(data["opp_stoch"]))
            mirror.set_("opp_deter", torch.zeros_like(data["opp_deter"]))

        # --- is_first handling ---
        if has_opp and self.mirror_opp_warm_start:
            # Keep original is_first — do NOT force True at t=0.
            # The warm initial state (from opponent) must survive the first
            # obs_step; obs_step zeros state when is_first=True.
            pass
        else:
            # Current behaviour: force RSSM re-init for mirrored trajectories.
            mirror["is_first"] = mirror["is_first"].clone()
            mirror["is_first"][:, 0] = True

        # --- Concatenate original + mirrored ---
        combined = torch.cat([data, mirror], dim=0)

        # --- Player initial ---
        stoch_orig, deter_orig = initial
        if has_opp and self.mirror_opp_warm_start:
            # Mirrored "player" was the original opponent → warm-start
            # from the opponent's t=0 RSSM state.
            opp_init_stoch, opp_init_deter = opp_initial
            combined_initial = (
                torch.cat([stoch_orig, opp_init_stoch], dim=0),
                torch.cat([deter_orig, opp_init_deter], dim=0),
            )
        else:
            combined_initial = (
                torch.cat([stoch_orig, torch.zeros_like(stoch_orig)], dim=0),
                torch.cat([deter_orig, torch.zeros_like(deter_orig)], dim=0),
            )

        # --- Opponent initial (for imagination warm-start) ---
        if opp_initial is not None:
            opp_s, opp_d = opp_initial
            if self.mirror_opp_warm_start:
                # Mirrored opponent initial = original player initial.
                combined_opp_initial = (
                    torch.cat([opp_s, stoch_orig], dim=0),
                    torch.cat([opp_d, deter_orig], dim=0),
                )
            else:
                combined_opp_initial = (
                    torch.cat([opp_s, torch.zeros_like(opp_s)], dim=0),
                    torch.cat([opp_d, torch.zeros_like(opp_d)], dim=0),
                )
        else:
            combined_opp_initial = None

        return combined, combined_initial, combined_opp_initial

    def update(self, index, stoch, deter):
        # Flatten the data
        index = [ind.reshape(-1) for ind in index]
        # (B, T, S, K) -> (B*T, S, K)
        stoch = stoch.reshape(-1, *stoch.shape[2:])
        # (B, T, D) -> (B*T, D)
        deter = deter.reshape(-1, *deter.shape[2:])
        # In storage, the length is the first dimension, and the batch (number of environments) is the second dimension.
        self._buffer[index[1], index[0]].set_("stoch", stoch)
        self._buffer[index[1], index[0]].set_("deter", deter)

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()


class PrioritizedBuffer(Buffer):
    """Prioritized replay buffer with episode-level tagging.

    New transitions start at ``baseline_priority``. At episode end, call
    :meth:`tag_episode` for each matched tag, then :meth:`flush_episode` once.

    Minimal setup outline::

            buffer:
                max_size: 5e5
                batch_size: 64
                batch_length: 128
                prioritized:
                    alpha: 0.7
                    beta: 0.0
                    baseline_priority: 1.0
                    tags:
                        - name: "goal_scored"
                            priority: 20.0
                            termination_term: "goal_scored"

            replay_buffer = PrioritizedBuffer(config.buffer)
            replay_buffer.tag_episode(episode_id, priority=20.0, tag="goal_scored")
            replay_buffer.flush_episode(episode_id)
    """

    def __init__(self, config):
        prioritized = config.prioritized
        self._baseline_priority = float(getattr(prioritized, "baseline_priority", 1.0))
        self._debug_metrics = bool(getattr(prioritized, "debug_metrics", False))
        self._episode_indices: dict[int, list] = defaultdict(list)
        self._episode_max_priority: dict[int, float] = {}
        self._episode_tags: dict[int, set[str]] = defaultdict(set)
        self._transitions_added: int = 0
        sampler = PrioritizedSliceSampler(
            max_capacity=int(config.max_size),
            alpha=float(prioritized.alpha),
            beta=float(getattr(prioritized, "beta", 0.0)),
            num_slices=int(config.batch_size),
            traj_key="episode",
            strict_length=True,
            end_key=None,
            truncated_key=None,
        )
        super().__init__(config, sampler=sampler)

    def add_transition(self, data):
        # (B, ...) -> (B, 1, ...) as in Buffer; captures storage indices.
        indices = self._buffer.extend(data.unsqueeze(1))  # shape (B, 2): [timestep, env]
        # Reset new transitions to baseline priority so they don't inherit the
        # inflated max_priority that results from tagging other episodes.
        self._buffer.update_priority(indices, torch.full((len(indices),), self._baseline_priority))
        self._transitions_added += len(data)
        # Accumulate indices per episode for later priority update.
        episode_ids = data["episode"]
        for i in range(len(episode_ids)):
            self._episode_indices[episode_ids[i].item()].append(indices[i : i + 1])

    def tag_episode(self, episode_id: int, priority: float, tag: str | None = None) -> None:
        """Boost sampling priority for every transition in *episode_id*.

        The optional *tag* name is tracked in :attr:`_episode_tags` for
        debug metrics (current tag counts, sampled fractions).
        """
        idx_list = self._episode_indices.get(episode_id, [])
        if not idx_list:
            return
        cur_max = self._episode_max_priority.get(episode_id, 0.0)
        if priority > cur_max:
            all_idx = torch.cat(idx_list)  # (T, 2)
            self._buffer.update_priority(all_idx, torch.full((len(all_idx),), priority))
            self._episode_max_priority[episode_id] = priority
        if tag is not None:
            self._episode_tags[episode_id].add(tag)

    def flush_episode(self, episode_id: int) -> None:
        """Drop the index accumulator for an untagged (or fully tagged) episode."""
        self._episode_indices.pop(episode_id, None)
        self._episode_max_priority.pop(episode_id, None)

    def flush_all_episodes(self) -> None:
        """Clear all accumulators (e.g. before eval resets discard in-progress episodes)."""
        self._episode_indices.clear()

    def compute_current_tag_counts(self) -> tuple[dict[str, int], int]:
        """Scan buffer storage for per-tag episode counts.

        Returns ``(tag_counts, unique_episode_count)``.
        Also prunes ``_episode_tags`` of evicted episode IDs.
        O(buffer_size) -- call infrequently (e.g. at eval time).
        """
        storage = self._buffer.storage  # LazyTensorStorage
        if storage.shape is None:
            return {}, 0
        raw_td = storage._storage  # underlying TensorDict
        episode_tensor = raw_td["episode"]
        current_ids: set[int] = set(episode_tensor.reshape(-1).unique().tolist())
        current_ids.discard(0)  # uninitialized slots before buffer fills

        tag_counts: dict[str, int] = {}
        for eid in current_ids:
            for tag in self._episode_tags.get(eid, ()):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Prune evicted episodes to prevent unbounded growth
        stale = set(self._episode_tags.keys()) - current_ids
        for eid in stale:
            del self._episode_tags[eid]

        return tag_counts, len(current_ids)

    def compute_sampled_tag_fractions(self) -> dict[str, float]:
        """Fraction of last-sampled episodes that carry each tag."""
        ep_ids = getattr(self, "last_sampled_episodes", None)
        if not ep_ids:
            return {}
        unique_eps = set(ep_ids)
        tag_hits: dict[str, int] = {}
        for eid in unique_eps:
            for tag in self._episode_tags.get(eid, ()):
                tag_hits[tag] = tag_hits.get(tag, 0) + 1
        n = max(len(unique_eps), 1)
        return {tag: count / n for tag, count in tag_hits.items()}

    def get_episode_tags(self, episode_id: int) -> set[str]:
        """Return tag names associated with an episode."""
        return self._episode_tags.get(episode_id, set())

