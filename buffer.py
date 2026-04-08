import warnings
from collections import defaultdict, deque

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
        data = sample_td[:, 1:]
        data.set_("action", sample_td["action"][:, :-1])  # action is 1 step back
        if "opponent_action" in sample_td.keys():
            data.set_("opponent_action", sample_td["opponent_action"][:, :-1])
        index = [ind.view(-1, self.batch_length + 1)[:, 1:] for ind in info["index"]]

        self._original_batch_size = data.shape[0]
        if self.mirror and "opponent_reward" in data.keys():
            data, initial = self._mirror_trajectories(data, initial)

        return data, index, initial

    def _mirror_trajectories(self, data, initial):
        """Swap player/opponent perspectives to create mirrored copies.

        Doubles the batch: ``[original_0..B-1, mirrored_0..B-1]``.
        Mirrored trajectories get ``is_first[:, 0] = True`` and zero RSSM
        initial states so the RSSM re-initialises from its learned prior.
        """
        from tensordict import TensorDict

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

        # Force RSSM re-initialisation for mirrored trajectories.
        # NOTE: This forces the RSSM to start from its learned prior for
        # mirrored sequences. The first few timesteps may have lower-quality
        # latent representations compared to original trajectories (which
        # have warm-started posteriors from buffer.update()). If this hurts
        # training quality, revisit: options include storing opponent RSSM
        # states in the buffer, or re-encoding from opponent observations.
        mirror["is_first"] = mirror["is_first"].clone()
        mirror["is_first"][:, 0] = True

        # Concatenate original + mirrored
        combined = torch.cat([data, mirror], dim=0)

        # Zero initial RSSM states for mirrored half (the RSSM observe()
        # will re-initialise from its learned prior due to is_first=True)
        stoch_orig, deter_orig = initial
        combined_initial = (
            torch.cat([stoch_orig, torch.zeros_like(stoch_orig)], dim=0),
            torch.cat([deter_orig, torch.zeros_like(deter_orig)], dim=0),
        )

        return combined, combined_initial

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
        self._buffer_capacity: int = int(config.max_size)
        self._episode_indices: dict[int, list] = defaultdict(list)
        self._tag_counts: dict[str, int] = defaultdict(int)
        self._current_tag_counts: dict[str, int] = defaultdict(int)
        self._tag_episode_queue: deque[tuple[int, str]] = deque()
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
        # Track total transitions for sliding-window eviction of current_tag_counts.
        self._transitions_added += len(data)
        # Evict tagged episodes from the front of the queue once all their
        # transitions are guaranteed overwritten (buffer has cycled past them).
        while self._tag_episode_queue:
            front_end, front_tag = self._tag_episode_queue[0]
            if self._transitions_added - front_end >= self._buffer_capacity:
                self._tag_episode_queue.popleft()
                self._current_tag_counts[front_tag] = max(0, self._current_tag_counts[front_tag] - 1)
            else:
                break
        # Accumulate indices per episode for later priority update.
        episode_ids = data["episode"]
        for i in range(len(episode_ids)):
            self._episode_indices[episode_ids[i].item()].append(indices[i : i + 1])

    def tag_episode(self, episode_id: int, priority: float, tag: str | None = None) -> None:
        """Boost sampling priority for every transition in *episode_id*.

        The optional *tag* name is only used for internal statistics returned
        by :meth:`get_tag_counts`.
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
            self._tag_counts[tag] += 1
            self._current_tag_counts[tag] += 1
            self._tag_episode_queue.append((self._transitions_added, tag))

    def flush_episode(self, episode_id: int) -> None:
        """Drop the index accumulator for an untagged (or fully tagged) episode."""
        self._episode_indices.pop(episode_id, None)
        self._episode_max_priority.pop(episode_id, None)

    def flush_all_episodes(self) -> None:
        """Clear all accumulators (e.g. before eval resets discard in-progress episodes)."""
        self._episode_indices.clear()

    def get_tag_counts(self) -> dict[str, int]:
        """Cumulative number of episodes tagged per tag name since training start."""
        return dict(self._tag_counts)

    def get_current_tag_counts(self) -> dict[str, int]:
        """Approximate number of tagged episodes currently in the buffer.

        Uses a sliding deque: an episode is considered evicted once
        ``_transitions_added - episode_end`` >= ``max_size``, i.e. the buffer
        has cycled past all of its transitions.
        """
        return dict(self._current_tag_counts)
