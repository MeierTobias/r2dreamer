import copy
import math
from collections import OrderedDict

import networks
import rssm
import tools
import torch
import torch.nn.functional as F
from networks import Projector
from optim import LaProp, clip_grad_agc_
from tensordict import TensorDict
from tools import to_f32
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR


class Dreamer(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.train_device = torch.device(config.train_device or config.device)
        self.sim_device = torch.device(config.sim_device or config.device)
        self.device = self.sim_device  # env-side tensors use sim_device
        self.multi_gpu = (self.sim_device != self.train_device)
        self.micro_batch_size = int(config.micro_batch_size)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = networks.ReturnEMA(device=self.train_device)
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)
        self.rep_loss = str(config.rep_loss)

        # Opponent separation: feed opponent actions to the world model
        self.opponent_separation = bool(getattr(config, "opponent_separation", False))
        self._imag_opponent = str(getattr(config, "imag_opponent", "random"))
        # Opponent networks for "selfplay" imagination mode — set externally
        # via set_imag_opponent_networks() after the self-play wrapper creates them.
        self._imag_opp_rssm = None
        self._imag_opp_actor = None
        wm_act_dim = self.act_dim * 2 if self.opponent_separation else self.act_dim

        # World model components
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(config.encoder, shapes)
        self.embed_size = self.encoder.out_dim
        self.rssm = rssm.RSSM(
            config.rssm,
            self.embed_size,
            wm_act_dim,
        )
        self.reward = networks.MLPHead(config.reward, self.rssm.feat_size)
        self.cont = networks.MLPHead(config.cont, self.rssm.feat_size)

        config.actor.shape = (act_space.n,) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))
        self.act_discrete = False
        if hasattr(act_space, "multi_discrete"):
            config.actor.dist = config.actor.dist.multi_disc
            self.act_discrete = True
        elif hasattr(act_space, "discrete"):
            config.actor.dist = config.actor.dist.disc
            self.act_discrete = True
        else:
            config.actor.dist = config.actor.dist.cont

        # Actor-critic components
        self.actor = networks.MLPHead(config.actor, self.rssm.feat_size)
        self.value = networks.MLPHead(config.critic, self.rssm.feat_size)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0

        self._loss_scales = dict(config.loss_scales)
        self._log_grads = bool(config.log_grads)

        modules = {
            "rssm": self.rssm,
            "actor": self.actor,
            "value": self.value,
            "reward": self.reward,
            "cont": self.cont,
            "encoder": self.encoder,
        }

        if self.rep_loss == "dreamer":
            self.decoder = networks.MultiDecoder(
                config.decoder,
                self.rssm._deter,
                self.rssm.flat_stoch,
                shapes,
            )
            recon = self._loss_scales.pop("recon")
            self._loss_scales.update({k: recon for k in self.decoder.all_keys})
            modules.update({"decoder": self.decoder})
        elif self.rep_loss == "r2dreamer" or self.rep_loss == "infonce":
            # add projector for latent to embedding
            self.prj = Projector(self.rssm.feat_size, self.embed_size)
            modules.update({"projector": self.prj})
            self.barlow_lambd = float(config.r2dreamer.lambd)
        elif self.rep_loss == "dreamerpro":
            dpc = config.dreamer_pro
            self.warm_up = int(dpc.warm_up)
            self.num_prototypes = int(dpc.num_prototypes)
            self.proto_dim = int(dpc.proto_dim)
            self.temperature = float(dpc.temperature)
            self.sinkhorn_eps = float(dpc.sinkhorn_eps)
            self.sinkhorn_iters = int(dpc.sinkhorn_iters)
            self.ema_update_every = int(dpc.ema_update_every)
            self.ema_update_fraction = float(dpc.ema_update_fraction)
            self.freeze_prototypes_iters = int(dpc.freeze_prototypes_iters)
            self.aug_max_delta = float(dpc.aug.max_delta)
            self.aug_same_across_time = bool(dpc.aug.same_across_time)
            self.aug_bilinear = bool(dpc.aug.bilinear)

            self._prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.proto_dim))
            self.obs_proj = nn.Linear(self.embed_size, self.proto_dim)
            self.feat_proj = nn.Linear(self.rssm.feat_size, self.proto_dim)
            self._ema_encoder = copy.deepcopy(self.encoder)
            self._ema_obs_proj = copy.deepcopy(self.obs_proj)
            for param in self._ema_encoder.parameters():
                param.requires_grad = False
            for param in self._ema_obs_proj.parameters():
                param.requires_grad = False
            self._ema_updates = 0
            modules.update({
                "prototypes": self._prototypes,
                "obs_proj": self.obs_proj,
                "feat_proj": self.feat_proj,
                "ema_encoder": self._ema_encoder,
                "ema_obs_proj": self._ema_obs_proj,
            })
        # count number of parameters in each module
        for key, module in modules.items():
            if isinstance(module, nn.Parameter):
                print(f"{module.numel():>14,}: {key}")
            else:
                print(f"{sum(p.numel() for p in module.parameters()):>14,}: {key}")
        self._named_params = OrderedDict()
        for name, module in modules.items():
            if isinstance(module, nn.Parameter):
                self._named_params[name] = module
            else:
                for param_name, param in module.named_parameters():
                    self._named_params[f"{name}.{param_name}"] = param
        print(f"Optimizer has: {sum(p.numel() for p in self._named_params.values())} parameters.")

        def _agc(params):
            clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)

        self._agc = _agc
        self._optimizer = LaProp(
            self._named_params.values(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
        self._scaler = GradScaler()

        def lr_lambda(step):
            if config.warmup:
                return min(1.0, (step + 1) / config.warmup)
            return 1.0

        self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        self.train()
        self.clone_and_freeze()
        self._compile = config.compile
        self._compiled = False
        # Inference copies are created by to() which is called from the
        # training script after construction.  Set aliases here so the
        # attributes exist before to() is invoked.
        self._inference_encoder = self._frozen_encoder
        self._inference_rssm = self._frozen_rssm
        self._inference_actor = self._frozen_actor

    def _update_slow_target(self):
        """Update slow-moving value target network."""
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    def train(self, mode=True):
        super().train(mode)
        # slow_value should be always eval mode
        self._slow_value.train(False)
        return self

    def clone_and_freeze(self):
        # NOTE: "requires_grad" affects whether a parameter is updated
        # not whether gradients flow through its operations
        self._frozen_encoder = copy.deepcopy(self.encoder)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.encoder.named_parameters(), self._frozen_encoder.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_rssm = copy.deepcopy(self.rssm)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.rssm.named_parameters(), self._frozen_rssm.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_reward = copy.deepcopy(self.reward)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.reward.named_parameters(), self._frozen_reward.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_cont = copy.deepcopy(self.cont)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.cont.named_parameters(), self._frozen_cont.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_actor = copy.deepcopy(self.actor)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.actor.named_parameters(), self._frozen_actor.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_value = copy.deepcopy(self.value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.value.named_parameters(), self._frozen_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_slow_value = copy.deepcopy(self._slow_value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self._slow_value.named_parameters(), self._frozen_slow_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

    def _create_inference_copies(self):
        """Create inference copies on sim_device for act() and video_pred().

        Single GPU: reuse frozen copies (shared .data, zero overhead).
        Multi-GPU: independent copies on sim_device.
        """
        if not self.multi_gpu:
            self._inference_encoder = self._frozen_encoder
            self._inference_rssm = self._frozen_rssm
            self._inference_actor = self._frozen_actor
            if hasattr(self, "decoder"):
                self._inference_decoder = self.decoder
            return
        # Multi-GPU: independent copies on sim_device
        enc = copy.deepcopy(self.encoder).to(self.sim_device).eval()
        rssm = copy.deepcopy(self.rssm).to(self.sim_device).eval()
        actor = copy.deepcopy(self.actor).to(self.sim_device).eval()
        # Fix RSSM._device so initial() creates tensors on sim_device
        rssm._device = self.sim_device
        modules = [enc, rssm, actor]
        # Decoder copy for video_pred on sim_device (~58 MB for 14.5M params)
        if hasattr(self, "decoder"):
            dec = copy.deepcopy(self.decoder).to(self.sim_device).eval()
            self._inference_decoder_orig = dec
            self._inference_decoder = dec
            modules.append(dec)
        for m in modules:
            for p in m.parameters():
                p.requires_grad_(False)
        # Keep uncompiled originals for load_state_dict (compiled modules
        # prefix keys with _orig_mod. which mismatches the source state_dict).
        self._inference_encoder_orig = enc
        self._inference_rssm_orig = rssm
        self._inference_actor_orig = actor
        if self._compile:
            self._inference_encoder = torch.compile(enc, mode="reduce-overhead")
            self._inference_rssm = torch.compile(rssm, mode="reduce-overhead")
            self._inference_actor = torch.compile(actor, mode="reduce-overhead")
        else:
            self._inference_encoder = enc
            self._inference_rssm = rssm
            self._inference_actor = actor

    def _sync_inference_copies(self):
        """Sync inference copies from trainable modules after optimizer step.

        Single GPU: frozen copies share .data, always up to date — no-op.
        Multi-GPU: load_state_dict on the uncompiled originals; compiled
        graphs share the same parameter tensors and pick up updates automatically.
        """
        if not self.multi_gpu:
            return
        self._inference_encoder_orig.load_state_dict(self.encoder.state_dict())
        self._inference_rssm_orig.load_state_dict(self.rssm.state_dict())
        self._inference_actor_orig.load_state_dict(self.actor.state_dict())
        if hasattr(self, "_inference_decoder_orig"):
            self._inference_decoder_orig.load_state_dict(self.decoder.state_dict())

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Re-establish shared memory after moving the model to a new device
        self.clone_and_freeze()
        self._create_inference_copies()
        return self

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        """Policy inference step."""
        # obs: dict of (B, *), state: (stoch: (B, S, K), deter: (B, D), prev_action: (B, A))
        torch.compiler.cudagraph_mark_step_begin()
        p_obs = self.preprocess(obs)
        # (B, E)
        embed = self._inference_encoder(p_obs)
        prev_stoch, prev_deter, prev_action = (
            state["stoch"],
            state["deter"],
            state["prev_action"],
        )
        # (B, S, K), (B, D)
        stoch, deter, _ = self._inference_rssm.obs_step(prev_stoch, prev_deter, prev_action, embed, obs["is_first"])
        # (B, F)
        feat = self._inference_rssm.get_feat(stoch, deter)
        action_dist = self._inference_actor(feat)
        # (B, A)
        action = action_dist.mode if eval else action_dist.rsample()
        # Build the RSSM prev_action: action dim * 2 when opponent_separation is on.
        if self.opponent_separation:
            opp_act = obs.get("opponent_action", torch.zeros_like(action))
            wm_action = torch.cat([action, opp_act], dim=-1)
        else:
            wm_action = action
        return action, TensorDict(
            {"stoch": stoch, "deter": deter, "prev_action": wm_action},
            batch_size=state.batch_size,
        )

    @torch.no_grad()
    def get_initial_state(self, B):
        stoch, deter = self._inference_rssm.initial(B)
        wm_act_dim = self.act_dim * 2 if self.opponent_separation else self.act_dim
        prev_action = torch.zeros(B, wm_act_dim, dtype=torch.float32, device=self.sim_device)
        action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.sim_device)
        return TensorDict(
            {"stoch": stoch, "deter": deter, "prev_action": prev_action, "action": action}, batch_size=(B,)
        )

    def set_imag_opponent_networks(self, opp_rssm, opp_actor):
        """Set frozen opponent RSSM and actor for ``"selfplay"`` imagination.

        Single GPU: receives the same module instances owned by
        :class:`DreamerSelfPlayWrapper`, so weight updates propagate
        automatically via shared references.

        Multi-GPU: the wrapper's copies live on ``sim_device`` but imagination
        runs on ``train_device``, so we deep-copy to ``train_device``.  Call
        :meth:`sync_imag_opponent_networks` after opponent weight updates.
        """
        if self.multi_gpu:
            self._imag_opp_rssm = copy.deepcopy(opp_rssm).to(self.train_device).eval()
            self._imag_opp_rssm._device = self.train_device
            self._imag_opp_actor = copy.deepcopy(opp_actor).to(self.train_device).eval()
            for p in self._imag_opp_rssm.parameters():
                p.requires_grad_(False)
            for p in self._imag_opp_actor.parameters():
                p.requires_grad_(False)
            # Keep references to the sim-device originals for syncing.
            self._imag_opp_rssm_src = opp_rssm
            self._imag_opp_actor_src = opp_actor
        else:
            self._imag_opp_rssm = opp_rssm
            self._imag_opp_actor = opp_actor

    def sync_imag_opponent_networks(self):
        """Sync imagination opponent copies from sim_device originals.

        Only needed in multi-GPU mode; single-GPU shares references.
        Call after :meth:`DreamerSelfPlayWrapper.maybe_update_opponent`.
        """
        if not self.multi_gpu:
            return
        if self._imag_opp_rssm is None:
            return
        self._imag_opp_rssm.load_state_dict(self._imag_opp_rssm_src.state_dict())
        self._imag_opp_actor.load_state_dict(self._imag_opp_actor_src.state_dict())

    @torch.no_grad()
    def video_pred(self, data, initial):
        torch.compiler.cudagraph_mark_step_begin()
        if self.multi_gpu:
            # Run on sim_device using inference copies — avoids competing
            # with CUDA graph private pools on train_device.
            data = data.to(self.sim_device)
            initial = tuple(v.to(self.sim_device) for v in initial)
            p_data = self.preprocess(data)
            return self._video_pred(
                p_data, initial,
                encoder=self._inference_encoder_orig,
                rssm=self._inference_rssm_orig,
                decoder=self._inference_decoder,
            )
        p_data = self.preprocess(data)
        return self._video_pred(p_data, initial)

    def _video_pred(self, data, initial, encoder=None, rssm=None, decoder=None):
        """Video prediction utility."""
        if self.rep_loss != "dreamer":
            raise NotImplementedError("video_pred requires decoder and is only supported when rep_loss == 'dreamer'.")

        encoder = encoder or self.encoder
        rssm = rssm or self.rssm
        decoder = decoder or self.decoder

        B = min(data["action"].shape[0], 6)
        # (B, T, E)
        embed = encoder(data)

        # Build world-model action for video prediction.
        if self.opponent_separation and "opponent_action" in data.keys():
            wm_action = torch.cat([data["action"], data["opponent_action"]], dim=-1)
        else:
            wm_action = data["action"]

        T = wm_action.shape[1]
        context_len = min(5, T)

        post_stoch, post_deter, _ = rssm.observe(
            embed[:B, :context_len],
            wm_action[:B, :context_len],
            tuple(val[:B] for val in initial),
            data["is_first"][:B, :context_len],
        )
        recon = decoder(post_stoch, post_deter)["image"].mode()[:B]

        if T > context_len:
            init_stoch, init_deter = post_stoch[:, -1], post_deter[:, -1]
            open_action = wm_action[:B, context_len:]
            prior_stoch, prior_deter = rssm.imagine_with_action(
                init_stoch,
                init_deter,
                open_action,
            )
            openl = decoder(prior_stoch, prior_deter)["image"].mode()
            model = torch.cat([recon[:, :context_len], openl], 1)
        else:
            model = recon[:, :context_len]

        truth = data["image"][:B, : model.shape[1]]
        error = (model - truth + 1.0) / 2.0
        return torch.cat([truth, model, error], 2)

    def update(self, replay_buffer):
        """Sample a batch from replay and perform one optimization step.

        When ``micro_batch_size < batch_size``, the sampled batch is split into
        micro-batches and gradients are accumulated before the optimizer step.
        This produces mathematically identical gradients while reducing peak GPU
        memory proportionally.
        """
        if self._compile and not self._compiled:
            print("Compiling update function with torch.compile...")
            self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")
            self._compiled = True
        sample = replay_buffer.sample()
        if sample is None:
            return {}  # skip this update – trajectories too short
        data, index, initial, opp_initial = sample
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        self._update_slow_target()
        if self.rep_loss == "dreamerpro":
            self.ema_update()

        B = p_data.shape[0]
        mbs = self.micro_batch_size
        num_acc = B // mbs
        loss_scale = 1.0 / num_acc

        # Collect posteriors from each micro-batch for buffer update
        all_stoch = []
        all_deter = []
        mets = {}

        for i in range(num_acc):
            s = i * mbs
            e = s + mbs
            micro_data = p_data[s:e]
            micro_initial = (initial[0][s:e], initial[1][s:e])
            micro_opp_initial = None
            if opp_initial is not None:
                micro_opp_initial = (opp_initial[0][s:e], opp_initial[1][s:e])
            with autocast(device_type=self.train_device.type, dtype=torch.float16):
                (stoch, deter), mets = self._cal_grad(
                    micro_data, micro_initial, loss_scale, opp_initial=micro_opp_initial
                )
            all_stoch.append(stoch)
            all_deter.append(deter)

        # Optimizer step (once, after all micro-batches)
        self._scaler.unscale_(self._optimizer)  # unscale grads in params
        if self.rep_loss == "dreamerpro" and self._ema_updates < self.freeze_prototypes_iters:
            self._prototypes.grad.zero_()
        if self._log_grads:
            old_params = [p.data.clone().detach() for p in self._named_params.values()]
            grads = [p.grad for p in self._named_params.values() if p.grad is not None]  # log grads before clipping
            grad_norm = tools.compute_global_norm(grads)
            grad_rms = tools.compute_rms(grads)
            mets["opt/grad_norm"] = grad_norm
            mets["opt/grad_rms"] = grad_rms
        self._agc(self._named_params.values())  # clipping
        self._scaler.step(self._optimizer)  # update params
        self._scaler.update()  # adjust scale
        self._scheduler.step()  # increment scheduler
        self._optimizer.zero_grad(set_to_none=True)  # reset grads
        mets["opt/lr"] = self._scheduler.get_lr()[0]
        mets["opt/grad_scale"] = self._scaler.get_scale()
        if self._log_grads:
            updates = [(new - old) for (new, old) in zip(self._named_params.values(), old_params)]
            update_rms = tools.compute_rms(updates)
            params_rms = tools.compute_rms(self._named_params.values())
            mets["opt/param_rms"] = params_rms
            mets["opt/update_rms"] = update_rms

        # Update latent vectors in replay buffer with concatenated posteriors.
        # When trajectory mirroring is active the batch is doubled (original +
        # mirrored), but only the original half has valid storage indices.
        all_stoch = torch.cat(all_stoch, dim=0)
        all_deter = torch.cat(all_deter, dim=0)
        orig_B = getattr(replay_buffer, "_original_batch_size", all_stoch.shape[0])
        replay_buffer.update(index, all_stoch[:orig_B].detach(), all_deter[:orig_B].detach())
        self._sync_inference_copies()
        return mets

    def _cal_grad(self, data, initial, loss_scale=1.0, opp_initial=None):
        """Compute gradients for one (micro-)batch.

        Notes
        -----
        This function computes:
        1) World model loss (dynamics + representation)
        2) Optional representation loss variants (Dreamer, R2-Dreamer, InfoNCE, DreamerPro)
        3) Imagination rollouts for actor-critic updates
        4) Replay-based value learning
        """
        # data: dict of (B, T, *), initial: (stoch: (B, S, K), deter: (B, D))
        losses = {}
        metrics = {}
        B, T = data.shape

        # === World model: posterior rollout and KL losses ===
        # (B, T, E)
        embed = self.encoder(data)
        # Build world-model action: 4D (player + opponent) when opponent_separation is on.
        if self.opponent_separation:
            wm_action = torch.cat([data["action"], data["opponent_action"]], dim=-1)  # (B, T, 4)
        else:
            wm_action = data["action"]  # (B, T, 2)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        post_stoch, post_deter, post_logit = self.rssm.observe(embed, wm_action, initial, data["is_first"])
        # (B, T, S, K)
        _, prior_logit = self.rssm.prior(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = torch.mean(dyn_loss)
        losses["rep"] = torch.mean(rep_loss)
        # === Representation / auxiliary losses ===
        # (B, T, F)
        feat = self.rssm.get_feat(post_stoch, post_deter)
        if self.rep_loss == "dreamer":
            recon_losses = {
                key: torch.mean(-dist.log_prob(data[key])) for key, dist in self.decoder(post_stoch, post_deter).items()
            }
            losses.update(recon_losses)
        elif self.rep_loss == "r2dreamer":
            # R2-Dreamer: Barlow Twins style redundancy reduction between latent features and encoder embeddings.
            # Flatten batch/time dims for a single cross-correlation matrix.
            # (B, T, F) -> (B*T, F)
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important

            x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)

            c = torch.mm(x1_norm.T, x2_norm) / (B * T)
            invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
            off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
            redundancy_loss = c[off_diag_mask].pow(2).sum()
            losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss
        elif self.rep_loss == "infonce":
            # Contrastive (InfoNCE) objective between projected latent features and encoder embeddings.
            # (B, T, F) -> (B*T, F)
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important
            logits = torch.matmul(x1, x2.T)
            norm_logits = logits - torch.max(logits, 1)[0][:, None]
            labels = torch.arange(norm_logits.shape[0]).long().to(self.train_device)
            losses["infonce"] = torch.nn.functional.cross_entropy(norm_logits, labels)
        elif self.rep_loss == "dreamerpro":
            # DreamerPro uses augmentation + EMA targets + Sinkhorn assignment.
            with torch.no_grad():
                data_aug = self.augment_data(data)
                initial_aug = (
                    # (B, ...) -> (2B, ...)
                    torch.cat([initial[0], initial[0]], dim=0),
                    torch.cat([initial[1], initial[1]], dim=0),
                )
                ema_proj = self.ema_proj(data_aug)

            embed_aug = self.encoder(data_aug)
            post_stoch_aug, post_deter_aug, _ = self.rssm.observe(
                embed_aug, data_aug["action"], initial_aug, data_aug["is_first"]
            )
            proto_losses = self.proto_loss(post_stoch_aug, post_deter_aug, embed_aug, ema_proj)
            losses.update(proto_losses)
        else:
            raise NotImplementedError

        # reward and continue
        losses["rew"] = torch.mean(-self.reward(feat).log_prob(to_f32(data["reward"])))
        cont = 1.0 - to_f32(data["is_terminal"])
        losses["con"] = torch.mean(-self.cont(feat).log_prob(cont))
        # log
        metrics["dyn_entropy"] = torch.mean(self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(self.rssm.get_dist(post_logit).entropy())

        # === Imagination rollout for actor-critic ===
        # (B*T, S, K), (B*T, D)
        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
        )
        # Opponent RSSM start for "selfplay" imagination mode.
        opp_start = None
        if self.opponent_separation and self._imag_opponent == "selfplay":
            if "opp_stoch" in data:
                # Per-timestep opponent states from buffer: (B, T, ...) → (B*T, ...)
                opp_start = (
                    data["opp_stoch"].reshape(-1, *data["opp_stoch"].shape[2:]).detach(),
                    data["opp_deter"].reshape(-1, *data["opp_deter"].shape[2:]).detach(),
                )
        # (B, T, ...) -> (B*T, ...)
        imag_feat, imag_action = self._imagine(start, self.imag_horizon + 1, opp_start=opp_start)
        imag_feat, imag_action = imag_feat.detach(), imag_action.detach()

        # (B*T, T_imag, 1)
        imag_reward = self._frozen_reward(imag_feat).mode()
        # (B*T, T_imag, 1)  probability of continuation
        imag_cont = self._frozen_cont(imag_feat).mean
        # (B*T, T_imag, 1)
        imag_value = self._frozen_value(imag_feat).mode()
        imag_slow_value = self._frozen_slow_value(imag_feat).mode()
        disc = 1 - 1 / self.horizon
        # (B*T, T_imag, 1)
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(
            last, term, imag_reward, imag_value, imag_value, disc, self.lamb
        )  # (B*T, T_imag-1, 1)
        ret_offset, ret_scale = self.return_ema(ret)
        # (B*T, T_imag-1, 1)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = self.actor(imag_feat)
        # (B*T, T_imag-1, 1)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))

        imag_value_dist = self.value(imag_feat)
        # (B*T, T_imag, 1)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        losses["value"] = torch.mean(
            weight[:, :-1].detach()
            * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[
                :, :-1
            ].unsqueeze(-1)
        )
        # log
        ret_normed = (ret - ret_offset) / ret_scale
        metrics["ret"] = torch.mean(ret_normed)
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
        metrics["adv"] = torch.mean(adv)
        metrics["adv_std"] = torch.std(adv)
        metrics["con"] = torch.mean(imag_cont)
        metrics["rew"] = torch.mean(imag_reward)
        metrics["val"] = torch.mean(imag_value)
        metrics["tar"] = torch.mean(ret)
        metrics["slowval"] = torch.mean(imag_slow_value)
        metrics["weight"] = torch.mean(weight)
        metrics["action_entropy"] = torch.mean(entropy)
        metrics.update(tools.tensorstats(imag_action, "action"))

        # === Replay-based value learning (keep gradients through world model) ===
        last, term, reward = (
            to_f32(data["is_last"]),
            to_f32(data["is_terminal"]),
            to_f32(data["reward"]),
        )
        feat = self.rssm.get_feat(post_stoch, post_deter)
        boot = ret[:, 0].reshape(B, T, 1)
        value = self._frozen_value(feat).mode()
        slow_value = self._frozen_slow_value(feat).mode()
        disc = 1 - 1 / self.horizon
        weight = 1.0 - last
        ret = self._lambda_return(last, term, reward, value, boot, disc, self.lamb)
        ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)

        # Keep this attached to the world model so gradients can flow through
        value_dist = self.value(feat)
        losses["repval"] = torch.mean(
            weight[:, :-1]
            * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value.detach()))[:, :-1].unsqueeze(
                -1
            )
        )
        # log
        metrics.update(tools.tensorstats(ret, "ret_replay"))
        metrics.update(tools.tensorstats(value, "value_replay"))
        metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))

        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        self._scaler.scale(total_loss * loss_scale).backward()

        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        return (post_stoch, post_deter), metrics

    @torch.no_grad()
    def _imagine(self, start, imag_horizon, opp_start=None):
        """Roll out the policy in latent space.

        Parameters
        ----------
        start : tuple[Tensor, Tensor]
            Player RSSM state ``(stoch, deter)`` with batch dim ``B``.
        imag_horizon : int
            Number of imagination steps.
        opp_start : tuple[Tensor, Tensor] | None
            Opponent RSSM state for ``"selfplay"`` imagination.  When
            ``None`` the opponent RSSM is initialised from zeros.
        """
        # (B, S, K), (B, D)
        feats = []
        actions = []
        stoch, deter = start
        B = stoch.shape[0]

        # Initialise opponent RSSM for selfplay imagination.
        selfplay = self.opponent_separation and self._imag_opponent == "selfplay"
        if selfplay:
            if opp_start is not None:
                opp_stoch, opp_deter = opp_start
            else:
                opp_stoch, opp_deter = self._imag_opp_rssm.initial(B)
            opp_prev_action = torch.zeros(B, self._imag_opp_rssm._act_dim, device=stoch.device)

        for _ in range(imag_horizon):
            # (B, F)
            feat = self._frozen_rssm.get_feat(stoch, deter)
            # (B, A)
            player_action = self._frozen_actor(feat).rsample()
            feats.append(feat)
            actions.append(player_action)
            # Build world-model action: concatenate opponent action when separation is on.
            if self.opponent_separation:
                if selfplay:
                    # Opponent acts from its own RSSM state.
                    opp_feat = self._imag_opp_rssm.get_feat(opp_stoch, opp_deter)
                    opp_action = self._imag_opp_actor(opp_feat).rsample()
                    # Player world-model action: [player, opponent]
                    wm_action = torch.cat([player_action, opp_action], dim=-1)  # (B, 4)
                    # Opponent RSSM prev_action: [opponent, player] (reversed
                    # perspective, matching DreamerSelfPlayWrapper.step()).
                    opp_prev_action = torch.cat([opp_action, player_action], dim=-1)
                    # Opponent prior transition (no observations in imagination).
                    opp_stoch, opp_deter = self._imag_opp_rssm.img_step(
                        opp_stoch, opp_deter, opp_prev_action
                    )
                else:
                    opp_action = self._get_imag_opponent_action(feat, player_action)
                    wm_action = torch.cat([player_action, opp_action], dim=-1)  # (B, 4)
            else:
                wm_action = player_action  # (B, 2)
            stoch, deter = self._frozen_rssm.img_step(stoch, deter, wm_action)

        # Stack along sequence dim T_imag.
        # (B, T_imag, F), (B, T_imag, A) — actions are 2D (player only)
        return torch.stack(feats, dim=1), torch.stack(actions, dim=1)

    @torch.no_grad()
    def _get_imag_opponent_action(self, feat, player_action):
        """Generate opponent actions during imagination rollouts.

        Used for the ``"random"`` and ``"zero"`` modes.  The ``"selfplay"``
        mode is handled directly in :meth:`_imagine` via a parallel opponent
        RSSM and is not routed through this method.
        """
        B = feat.shape[0]
        if self._imag_opponent == "zero":
            return torch.zeros(B, self.act_dim, device=feat.device)
        else:  # "random"
            return 2 * torch.rand(B, self.act_dim, device=feat.device) - 1

    @torch.no_grad()
    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        """
        lamb=1 means discounted Monte Carlo return.
        lamb=0 means fixed 1-step return.
        """
        assert last.shape == term.shape == reward.shape == value.shape == boot.shape
        live = (1 - to_f32(term))[:, 1:] * disc
        cont = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)

    @torch.no_grad()
    def preprocess(self, data):
        if "image" in data:
            data["image"] = to_f32(data["image"]) / 255.0
        return data

    @torch.no_grad()
    def augment_data(self, data):
        data_aug = {k: torch.cat([v, v], axis=0) for k, v in data.items()}
        # (B, T, H, W, C) -> (B, T, C, H, W)
        image = data_aug["image"].permute(0, 1, 4, 2, 3)
        data_aug["image"] = self.random_translate(
            image,
            self.aug_max_delta,
            same_across_time=self.aug_same_across_time,
            bilinear=self.aug_bilinear,
        )
        # (B, T, C, H, W) -> (B, T, H, W, C)
        data_aug["image"] = data_aug["image"].permute(0, 1, 3, 4, 2)
        return data_aug

    @torch.no_grad()
    def ema_proj(self, data):
        with torch.no_grad():
            embed = self._ema_encoder(data)
            proj = self._ema_obs_proj(embed)
        return F.normalize(proj, p=2, dim=-1)

    @torch.no_grad()
    def ema_update(self):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)
        self._prototypes.data.copy_(prototypes)
        if self._ema_updates % self.ema_update_every == 0:
            mix = self.ema_update_fraction if self._ema_updates > 0 else 1.0
            for s, d in zip(self.encoder.parameters(), self._ema_encoder.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
            for s, d in zip(self.obs_proj.parameters(), self._ema_obs_proj.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
        self._ema_updates += 1

    def sinkhorn(self, scores):
        """Sinkhorn-Knopp normalization.

        Notes
        -----
        Given a score matrix, we iteratively normalize rows and columns in log
        space so that the resulting assignment matrix is approximately doubly
        stochastic.
        """
        shape = scores.shape
        K = shape[0]
        scores = scores.reshape(-1)
        log_Q = F.log_softmax(scores / self.sinkhorn_eps, dim=0)
        log_Q = log_Q.reshape(K, -1)
        N = log_Q.shape[1]
        for _ in range(self.sinkhorn_iters):
            log_row_sums = torch.logsumexp(log_Q, dim=1, keepdim=True)
            log_Q = log_Q - log_row_sums - math.log(K)
            log_col_sums = torch.logsumexp(log_Q, dim=0, keepdim=True)
            log_Q = log_Q - log_col_sums - math.log(N)
        log_Q = log_Q + math.log(N)
        Q = torch.exp(log_Q)
        return Q.reshape(shape)

    def proto_loss(self, post_stoch, post_deter, embed, ema_proj):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)

        obs_proj = self.obs_proj(embed)
        obs_norm = torch.norm(obs_proj, dim=-1)
        obs_proj = F.normalize(obs_proj, p=2, dim=-1)

        B, T = obs_proj.shape[:2]
        # (B, T, P) -> (B*T, P)
        obs_proj = obs_proj.reshape(B * T, -1)
        obs_scores = torch.matmul(obs_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        obs_scores = obs_scores.reshape(B, T, -1).permute(2, 0, 1)
        obs_scores = obs_scores[:, :, self.warm_up :]
        obs_logits = F.log_softmax(obs_scores / self.temperature, dim=0)
        obs_logits_1, obs_logits_2 = torch.chunk(obs_logits, 2, dim=1)

        # (B, T, P) -> (B*T, P)
        ema_proj = ema_proj.reshape(B * T, -1)
        ema_scores = torch.matmul(ema_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        ema_scores = ema_scores.reshape(B, T, -1).permute(2, 0, 1)
        ema_scores = ema_scores[:, :, self.warm_up :]
        ema_scores_1, ema_scores_2 = torch.chunk(ema_scores, 2, dim=1)

        with torch.no_grad():
            ema_targets_1 = self.sinkhorn(ema_scores_1)
            ema_targets_2 = self.sinkhorn(ema_scores_2)
        ema_targets = torch.cat([ema_targets_1, ema_targets_2], dim=1)

        feat = self.rssm.get_feat(post_stoch, post_deter)
        feat_proj = self.feat_proj(feat)
        feat_norm = torch.norm(feat_proj, dim=-1)
        feat_proj = F.normalize(feat_proj, p=2, dim=-1)

        # (B, T, P) -> (B*T, P)
        feat_proj = feat_proj.reshape(B * T, -1)
        feat_scores = torch.matmul(feat_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        feat_scores = feat_scores.reshape(B, T, -1).permute(2, 0, 1)
        feat_scores = feat_scores[:, :, self.warm_up :]
        feat_logits = F.log_softmax(feat_scores / self.temperature, dim=0)

        swav_loss = -0.5 * torch.mean(torch.sum(ema_targets_2 * obs_logits_1, dim=0)) - 0.5 * torch.mean(
            torch.sum(ema_targets_1 * obs_logits_2, dim=0)
        )
        temp_loss = -torch.mean(torch.sum(ema_targets * feat_logits, dim=0))
        norm_loss = torch.mean(torch.square(obs_norm - 1)) + torch.mean(torch.square(feat_norm - 1))

        return {
            "swav": swav_loss,
            "temp": temp_loss,
            "norm": norm_loss,
        }

    @torch.no_grad()
    def random_translate(self, x, max_delta, same_across_time=False, bilinear=False):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        pad = int(max_delta)

        # Pad
        x_padded = F.pad(x_flat, (pad, pad, pad, pad), "replicate")
        h_padded, w_padded = H + 2 * pad, W + 2 * pad

        # Create base grid
        eps_h = 1.0 / h_padded
        eps_w = 1.0 / w_padded
        arange_h = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, h_padded, device=x.device, dtype=x.dtype)[:H]
        arange_w = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, w_padded, device=x.device, dtype=x.dtype)[:W]
        arange_h = arange_h.unsqueeze(1).repeat(1, W).unsqueeze(2)
        arange_w = arange_w.unsqueeze(0).repeat(H, 1).unsqueeze(2)
        base_grid = torch.cat([arange_w, arange_h], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(B * T, 1, 1, 1)

        # Create shift
        if same_across_time:
            shift = torch.randint(0, 2 * pad + 1, size=(B, 1, 1, 1, 2), device=x.device, dtype=x.dtype)
            shift = shift.repeat(1, T, 1, 1, 1).reshape(B * T, 1, 1, 2)
        else:
            shift = torch.randint(0, 2 * pad + 1, size=(B * T, 1, 1, 2), device=x.device, dtype=x.dtype)

        shift = shift * 2.0 / torch.tensor([w_padded, h_padded], device=x.device, dtype=x.dtype)

        # Apply shift and sample
        grid = base_grid + shift
        mode = "bilinear" if bilinear else "nearest"
        x_translated = F.grid_sample(x_padded, grid, mode=mode, padding_mode="zeros", align_corners=False)

        return x_translated.reshape(B, T, C, H, W)
