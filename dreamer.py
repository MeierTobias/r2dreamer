import copy
import math
import time
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


class PerfTimer:
    """Lightweight wall-clock timer for profiling multi-GPU phases.

    Uses ``time.perf_counter()`` instead of CUDA events to avoid
    cross-device issues when timing phases that span multiple GPUs.
    Relies on ``torch.cuda.synchronize()`` being called at phase
    boundaries (which the two-sync dispatch already does).
    """

    def __init__(self):
        self._starts = {}
        self._results = {}

    def start(self, name):
        self._starts[name] = time.perf_counter()

    def stop(self, name):
        if name in self._starts:
            self._results[name] = (time.perf_counter() - self._starts[name]) * 1000

    def elapsed_ms(self):
        """Return {name: ms} for all completed timings."""
        return dict(self._results)


class _FakeOptimizer:
    """Minimal shim so GradScaler.unscale_ can iterate parameter groups."""

    def __init__(self, params):
        self.param_groups = [{"params": list(params)}]


class Dreamer(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        # Device setup.  train_devices is the canonical config; train_device
        # and sim_device are derived by train_dreamer.py and propagated here.
        self.train_device = torch.device(config.train_device or config.device)
        self.sim_device = torch.device(config.sim_device or config.device)
        self.device = self.sim_device  # env-side tensors use sim_device
        self.multi_gpu = (self.sim_device != self.train_device)

        # Data-parallel: train_devices is a list; first element == train_device.
        train_devices_cfg = getattr(config, "train_devices", None)
        if train_devices_cfg and len(train_devices_cfg) > 1:
            self.train_devices = [torch.device(d) for d in train_devices_cfg]
        else:
            self.train_devices = [self.train_device]
        self.data_parallel = len(self.train_devices) > 1

        self.micro_batch_size = int(config.micro_batch_size)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = networks.ReturnEMA(device=self.train_device)
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)
        self.rep_loss = str(config.rep_loss)
        if self.data_parallel:
            assert self.rep_loss != "dreamerpro", (
                "DreamerPro is not supported with data-parallel training. "
                "Sinkhorn assignment requires the full batch and cannot be split across GPUs."
            )

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
        self._compile_mode = str(getattr(config, "compile_mode", "reduce-overhead"))
        # Inference copies can use a different compile mode than training.
        # "reduce-overhead" uses CUDA graphs (fast but ~3 GB private pool),
        # "default" avoids CUDA graphs (lower VRAM), None disables compilation.
        _inf_mode = getattr(config, "inference_compile_mode", None)
        if _inf_mode is None:
            self._inference_compile_mode = self._compile_mode
        elif str(_inf_mode).lower() in ("none", "false", "off"):
            self._inference_compile_mode = None
        else:
            self._inference_compile_mode = str(_inf_mode)
        self._compiled = False
        # Inference copies are created by to() which is called from the
        # training script after construction.  Set aliases here so the
        # attributes exist before to() is invoked.
        self._inference_encoder = self._frozen_encoder
        self._inference_rssm = self._frozen_rssm
        self._inference_actor = self._frozen_actor
        # Data-parallel replicas are created by to() after construction.
        self._replicas = []
        self._replica_scalers = []
        # Deferred inference sync: training thread increments _weights_version
        # after each optimizer step; main thread calls sync_inference_if_needed()
        # to pull new weights before act().
        self._weights_version = 0
        self._inference_version = 0

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
        if self._compile and self._inference_compile_mode is not None:
            _inf_mode = self._inference_compile_mode
            print(f"Compiling inference copies with torch.compile(mode={_inf_mode!r})")
            self._inference_encoder = torch.compile(enc, mode=_inf_mode)
            self._inference_rssm = torch.compile(rssm, mode=_inf_mode)
            self._inference_actor = torch.compile(actor, mode=_inf_mode)
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

    def sync_inference_if_needed(self):
        """Sync inference copies only when training has produced new weights.

        Called by the env collection thread before each act() call.  Avoids
        redundant syncs when no optimizer step has occurred since the last sync.
        Safe to call from a single thread (the env thread) without locking.
        """
        if self._inference_version < self._weights_version:
            self._sync_inference_copies()
            self._inference_version = self._weights_version

    def _create_training_replicas(self):
        """Create model replicas on secondary training GPUs for data-parallel training.

        Each replica is a fully self-contained dict with trainable modules,
        frozen copies (sharing .data with trainable), opponent imagination
        modules, and uncompiled ``orig`` references for ``load_state_dict``.
        """
        if not self.data_parallel:
            self._replicas = []
            self._replica_scalers = []
            return

        self._replicas = []
        self._replica_scalers = []
        secondary_devices = self.train_devices[1:]

        trainable_names = ["encoder", "rssm", "actor", "value", "reward", "cont"]
        if hasattr(self, "decoder"):
            trainable_names.append("decoder")
        if hasattr(self, "prj"):
            trainable_names.append("prj")

        for dev in secondary_devices:
            replica = {}

            # --- Trainable modules ---
            for name in trainable_names:
                replica[name] = copy.deepcopy(getattr(self, name)).to(dev)
            replica["rssm"]._device = dev

            # --- Frozen copies (share .data with replica's trainable) ---
            frozen_names = ["encoder", "rssm", "reward", "cont", "actor", "value"]
            replica["frozen"] = {}
            for name in frozen_names:
                src = replica[name]
                frozen = copy.deepcopy(src)
                for p_orig, p_frozen in zip(src.parameters(), frozen.parameters()):
                    p_frozen.data = p_orig.data
                    p_frozen.requires_grad_(False)
                frozen.eval()
                replica["frozen"][name] = frozen
            replica["frozen"]["rssm"]._device = dev

            # slow_value: independent frozen copy from primary
            slow = copy.deepcopy(self._slow_value).to(dev)
            for p in slow.parameters():
                p.requires_grad_(False)
            slow.eval()
            replica["frozen"]["slow_value"] = slow

            # --- Opponent imagination modules (if selfplay) ---
            if self._imag_opp_rssm is not None:
                opp_rssm = copy.deepcopy(self._imag_opp_rssm).to(dev).eval()
                opp_rssm._device = dev
                opp_actor = copy.deepcopy(self._imag_opp_actor).to(dev).eval()
                for p in opp_rssm.parameters():
                    p.requires_grad_(False)
                for p in opp_actor.parameters():
                    p.requires_grad_(False)
                replica["imag_opp_rssm"] = opp_rssm
                replica["imag_opp_actor"] = opp_actor

            replica["device"] = dev

            # --- Uncompiled originals for load_state_dict (Decision 4) ---
            replica["orig"] = {}
            for name in trainable_names:
                replica["orig"][name] = replica[name]
            for name in replica["frozen"]:
                replica["orig"]["frozen_" + name] = replica["frozen"][name]
            if "imag_opp_rssm" in replica:
                replica["orig"]["imag_opp_rssm"] = replica["imag_opp_rssm"]
                replica["orig"]["imag_opp_actor"] = replica["imag_opp_actor"]

            # NOTE: individual replica sub-modules are NOT compiled.
            # The whole-function compilation of _cal_grad_with_modules traces
            # through them automatically.  Compiling sub-modules individually
            # would create redundant CUDA graph private pools on each replica
            # GPU (~3 GiB overhead), causing OOM on 24 GiB cards.

            self._replicas.append(replica)
            self._replica_scalers.append(GradScaler())

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Re-establish shared memory after moving the model to a new device
        self.clone_and_freeze()
        self._create_inference_copies()
        self._create_training_replicas()
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

        # Create opponent copies on existing data-parallel replicas.
        # Replicas may have been created by to() before this method was called,
        # so they won't have opponent modules yet.
        if self.data_parallel and self._imag_opp_rssm is not None:
            for replica in self._replicas:
                if "imag_opp_rssm" in replica:
                    continue  # already present
                dev = replica["device"]
                opp_rssm_copy = copy.deepcopy(self._imag_opp_rssm).to(dev).eval()
                opp_rssm_copy._device = dev
                opp_actor_copy = copy.deepcopy(self._imag_opp_actor).to(dev).eval()
                for p in opp_rssm_copy.parameters():
                    p.requires_grad_(False)
                for p in opp_actor_copy.parameters():
                    p.requires_grad_(False)
                # Store originals for load_state_dict and as the active modules.
                # Not compiled individually — _cal_grad_with_modules traces through them.
                replica["orig"]["imag_opp_rssm"] = opp_rssm_copy
                replica["orig"]["imag_opp_actor"] = opp_actor_copy
                replica["imag_opp_rssm"] = opp_rssm_copy
                replica["imag_opp_actor"] = opp_actor_copy

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
        self._broadcast_opponent_to_replicas()

    def _broadcast_params(self):
        """Copy primary model parameters to all replicas after optimizer step.

        Uses ``replica["orig"]`` references so ``load_state_dict`` works on
        compiled modules (keys are not prefixed with ``_orig_mod.``).
        Frozen copies share ``.data`` with trainable modules and update
        automatically; only ``slow_value`` needs an explicit sync.
        """
        if not self.data_parallel:
            return

        trainable_names = ["encoder", "rssm", "actor", "value", "reward", "cont"]
        if hasattr(self, "decoder"):
            trainable_names.append("decoder")
        if hasattr(self, "prj"):
            trainable_names.append("prj")

        for replica in self._replicas:
            for name in trainable_names:
                replica["orig"][name].load_state_dict(
                    getattr(self, name).state_dict()
                )
            replica["orig"]["frozen_slow_value"].load_state_dict(
                self._slow_value.state_dict()
            )

    def _broadcast_opponent_to_replicas(self):
        """Copy imagination opponent weights to replicas."""
        if not self.data_parallel:
            return
        if self._imag_opp_rssm is None:
            return
        for replica in self._replicas:
            if "imag_opp_rssm" not in replica:
                continue
            replica["orig"]["imag_opp_rssm"].load_state_dict(
                self._imag_opp_rssm.state_dict()
            )
            replica["orig"]["imag_opp_actor"].load_state_dict(
                self._imag_opp_actor.state_dict()
            )

    def _reduce_gradients(self):
        """Unscale replica gradients and sum into primary model parameters.

        Each replica has its own ``GradScaler``.  We unscale first (via
        ``_FakeOptimizer``) so all gradients are in the same fp32 space,
        then accumulate into the primary's ``.grad`` tensors.

        Returns
        -------
        bool
            ``True`` if any replica scaler detected inf/nan during unscaling.
            The caller should skip the optimizer step in this case because the
            primary scaler's ``found_inf`` flag won't reflect the replica's inf.
        """
        if not self.data_parallel:
            return False

        trainable_names = ["encoder", "rssm", "actor", "value", "reward", "cont"]
        if hasattr(self, "decoder"):
            trainable_names.append("decoder")
        if hasattr(self, "prj"):
            trainable_names.append("prj")

        # Unscale each replica's gradients with its own scaler and track inf.
        replica_found_inf = False
        for replica, scaler in zip(self._replicas, self._replica_scalers):
            params = []
            for name in trainable_names:
                params.extend(replica["orig"][name].parameters())
            fake_opt = _FakeOptimizer(params)
            scaler.unscale_(fake_opt)
            # Check if this scaler detected inf/nan.
            opt_state = scaler._per_optimizer_states[id(fake_opt)]
            if any(v.item() for v in opt_state["found_inf_per_device"].values()):
                replica_found_inf = True

        # Sum replica gradients into primary.
        for replica in self._replicas:
            for name in trainable_names:
                primary_module = getattr(self, name)
                replica_module = replica["orig"][name]
                for p_primary, p_replica in zip(
                    primary_module.parameters(), replica_module.parameters()
                ):
                    if p_primary.grad is not None and p_replica.grad is not None:
                        p_primary.grad.add_(p_replica.grad.to(self.train_device))
                    elif p_replica.grad is not None:
                        p_primary.grad = p_replica.grad.to(self.train_device)
            # Zero replica gradients for next step.
            for name in trainable_names:
                replica["orig"][name].zero_grad(set_to_none=True)

        return replica_found_inf

    @torch.no_grad()
    def video_pred(self, data, initial):
        torch.compiler.cudagraph_mark_step_begin()
        if self.multi_gpu:
            # Run on sim_device using inference copies — avoids competing
            # with CUDA graph private pools on train_device.
            # Slice to 6 samples before the device transfer to avoid moving
            # the full batch_size worth of data to sim_device.
            B = min(data["action"].shape[0], 6)
            data = {k: v[:B] for k, v in data.items()}
            initial = tuple(v[:B] for v in initial)
            data = {k: v.to(self.sim_device) for k, v in data.items()}
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
        # Slice to B samples *before* the encoder to avoid allocating
        # full-batch CNN activations (saves GBs of VRAM on sim_device).
        data = {k: v[:B] for k, v in data.items()}
        initial = tuple(v[:B] for v in initial)
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
            embed[:, :context_len],
            wm_action[:, :context_len],
            initial,
            data["is_first"][:, :context_len],
        )
        recon = decoder(post_stoch, post_deter)["image"].mode()

        if T > context_len:
            init_stoch, init_deter = post_stoch[:, -1], post_deter[:, -1]
            open_action = wm_action[:, context_len:]
            prior_stoch, prior_deter = rssm.imagine_with_action(
                init_stoch,
                init_deter,
                open_action,
            )
            openl = decoder(prior_stoch, prior_deter)["image"].mode()
            model = torch.cat([recon[:, :context_len], openl], 1)
        else:
            model = recon[:, :context_len]

        truth = data["image"][:, : model.shape[1]]
        error = (model - truth + 1.0) / 2.0
        return torch.cat([truth, model, error], 2)

    def update(self, replay_buffer):
        """Sample a batch from replay and perform one optimization step.

        When ``micro_batch_size < batch_size``, the sampled batch is split into
        micro-batches and gradients are accumulated before the optimizer step.
        This produces mathematically identical gradients while reducing peak GPU
        memory proportionally.

        With ``data_parallel=True``, micro-batches are distributed across
        training GPUs for parallel forward+backward, then gradients are reduced
        to the primary GPU for a single optimizer step.
        """
        if self._compile and not self._compiled:
            _mode = self._compile_mode
            if self.data_parallel:
                # Two-sync path: compile Part A and Part B separately.
                print(f"Compiling _forward_world_model with torch.compile(mode={_mode!r})...")
                self._forward_world_model = torch.compile(
                    self._forward_world_model, mode=_mode
                )
                print(f"Compiling _forward_actor_critic_and_backward with torch.compile(mode={_mode!r})...")
                self._forward_actor_critic_and_backward = torch.compile(
                    self._forward_actor_critic_and_backward, mode=_mode
                )
                # Per-replica compiled instances.
                import types
                self._forward_wm_per_replica = []
                self._forward_ac_per_replica = []
                _uncompiled_wm = type(self)._forward_world_model
                _uncompiled_ac = type(self)._forward_actor_critic_and_backward
                for ri, replica in enumerate(self._replicas):
                    dev = replica["device"]
                    print(f"Compiling Part A+B for replica {ri} ({dev}) "
                          f"with mode={_mode!r}...")
                    self._forward_wm_per_replica.append(
                        torch.compile(types.MethodType(_uncompiled_wm, self), mode=_mode)
                    )
                    self._forward_ac_per_replica.append(
                        torch.compile(types.MethodType(_uncompiled_ac, self), mode=_mode)
                    )
            else:
                # Single-GPU path: compile the monolithic function.
                print(f"Compiling _cal_grad_with_modules with torch.compile(mode={_mode!r})...")
                self._cal_grad_with_modules = torch.compile(
                    self._cal_grad_with_modules, mode=_mode
                )
            self._compiled = True

        _timer = PerfTimer()
        _t0 = time.perf_counter()

        torch.cuda.nvtx.range_push("sample")
        _timer.start("sample")
        sample = replay_buffer.sample()
        _timer.stop("sample")
        torch.cuda.nvtx.range_pop()
        if sample is None:
            return {}  # skip this update – trajectories too short
        data, index, initial, opp_initial = sample
        torch.compiler.cudagraph_mark_step_begin()
        torch.cuda.nvtx.range_push("preprocess")
        _timer.start("preprocess")
        p_data = self.preprocess(data)
        _timer.stop("preprocess")
        torch.cuda.nvtx.range_pop()
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

        if not self.data_parallel:
            # --- Single training GPU: existing sequential micro-batch loop ---
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
        else:
            # --- Multi-GPU: two-sync parallel dispatch ---
            # All GPUs run Part A (world model + imagination) in parallel,
            # sync to compute return_ema over the full batch, then all GPUs
            # run Part B (actor-critic + backward) in parallel.
            num_gpus = len(self.train_devices)
            gpu_assignments = [[] for _ in range(num_gpus)]
            for i in range(num_acc):
                gpu_assignments[i % num_gpus].append(i)

            # Helper dicts for primary modules.
            _primary_modules = {
                "encoder": self.encoder, "rssm": self.rssm,
                "actor": self.actor, "value": self.value,
                "reward": self.reward, "cont": self.cont,
                "decoder": getattr(self, "decoder", None),
                "prj": getattr(self, "prj", None),
            }
            _primary_frozen = {
                "rssm": self._frozen_rssm,
                "reward": self._frozen_reward,
                "cont": self._frozen_cont,
                "actor": self._frozen_actor,
                "value": self._frozen_value,
                "slow_value": self._frozen_slow_value,
            }
            _primary_opp = {
                "rssm": self._imag_opp_rssm,
                "actor": self._imag_opp_actor,
            }

            # == Pre-stage: async data transfers to all replica GPUs ==
            torch.cuda.nvtx.range_push("prestage_transfers")
            _timer.start("prestage")
            replica_staged = []
            for gpu_idx, replica in enumerate(self._replicas, start=1):
                dev = replica["device"]
                xfer_stream = torch.cuda.Stream(dev)
                with torch.cuda.stream(xfer_stream):
                    batches = []
                    for i in gpu_assignments[gpu_idx]:
                        s, e = i * mbs, (i + 1) * mbs
                        micro_opp = None
                        if opp_initial is not None:
                            micro_opp = (
                                opp_initial[0][s:e].to(dev, non_blocking=True),
                                opp_initial[1][s:e].to(dev, non_blocking=True),
                            )
                        batches.append((
                            i,
                            p_data[s:e].to(dev, non_blocking=True),
                            (initial[0][s:e].to(dev, non_blocking=True),
                             initial[1][s:e].to(dev, non_blocking=True)),
                            micro_opp,
                        ))
                replica_staged.append((gpu_idx, replica, xfer_stream, batches))
            _timer.stop("prestage")
            torch.cuda.nvtx.range_pop()

            # == Part A: all GPUs run world model + imagination in parallel ==
            torch.cuda.nvtx.range_push("part_a")
            _timer.start("part_a")
            # Collect (micro_batch_idx, ret, intermediates, losses, metrics)
            # per GPU for Part B.
            part_a_results_primary = []
            part_a_results_replica = []  # (gpu_idx, micro_idx, ret_on_primary, intermediates, losses, metrics)

            # Primary GPU Part A
            torch.cuda.nvtx.range_push("part_a_primary")
            for i in gpu_assignments[0]:
                s, e = i * mbs, (i + 1) * mbs
                micro_data = p_data[s:e]
                micro_initial = (initial[0][s:e], initial[1][s:e])
                micro_opp = None
                if opp_initial is not None:
                    micro_opp = (opp_initial[0][s:e], opp_initial[1][s:e])
                with autocast(device_type=self.train_device.type, dtype=torch.float16):
                    ret, intermediates, losses, part_mets = self._forward_world_model(
                        modules=_primary_modules,
                        frozen=_primary_frozen,
                        opp_modules=_primary_opp,
                        data=micro_data, initial=micro_initial,
                        opp_initial=micro_opp,
                    )
                part_a_results_primary.append((i, ret, intermediates, losses, part_mets))
            torch.cuda.nvtx.range_pop()

            # Replica GPUs Part A (data already pre-staged)
            for gpu_idx, replica, xfer_stream, batches in replica_staged:
                dev = replica["device"]
                ri = gpu_idx - 1
                torch.cuda.current_stream(dev).wait_stream(xfer_stream)
                torch.cuda.nvtx.range_push(f"part_a_replica_{ri}")
                _wm_fn = self._forward_wm_per_replica[ri] if hasattr(self, "_forward_wm_per_replica") else self._forward_world_model
                for i, micro_data, micro_initial, micro_opp in batches:
                    with autocast(device_type=dev.type, dtype=torch.float16):
                        ret, intermediates, losses, part_mets = _wm_fn(
                            modules={
                                "encoder": replica["encoder"],
                                "rssm": replica["rssm"],
                                "actor": replica["actor"],
                                "value": replica["value"],
                                "reward": replica["reward"],
                                "cont": replica["cont"],
                                "decoder": replica.get("decoder"),
                                "prj": replica.get("prj"),
                            },
                            frozen=replica["frozen"],
                            opp_modules={
                                "rssm": replica.get("imag_opp_rssm"),
                                "actor": replica.get("imag_opp_actor"),
                            },
                            data=micro_data, initial=micro_initial,
                            opp_initial=micro_opp,
                        )
                    part_a_results_replica.append((gpu_idx, i, ret, intermediates, losses, part_mets))
                torch.cuda.nvtx.range_pop()
            _timer.stop("part_a")
            torch.cuda.nvtx.range_pop()  # part_a

            # == Sync 1: compute return_ema over full batch ==
            torch.cuda.nvtx.range_push("sync_ret_ema")
            _timer.start("sync_ema")
            for dev in self.train_devices:
                torch.cuda.synchronize(dev)
            # Gather ret tensors from all GPUs to primary for quantile computation.
            all_ret_for_ema = []
            for _, ret, _, _, _ in part_a_results_primary:
                all_ret_for_ema.append(ret)
            for _, _, ret, _, _, _ in part_a_results_replica:
                all_ret_for_ema.append(ret.to(self.train_device))
            concatenated_ret = torch.cat(all_ret_for_ema)
            ret_offset, ret_scale = self.return_ema(concatenated_ret)
            ret_norm_state = (ret_offset.item(), ret_scale.item())
            _timer.stop("sync_ema")
            torch.cuda.nvtx.range_pop()

            # == Part B: all GPUs run actor-critic + backward in parallel ==
            torch.cuda.nvtx.range_push("part_b")
            _timer.start("part_b")
            all_results = []

            # Primary GPU Part B
            torch.cuda.nvtx.range_push("part_b_primary")
            for i, ret, intermediates, losses, part_mets in part_a_results_primary:
                with autocast(device_type=self.train_device.type, dtype=torch.float16):
                    (stoch, deter), mets = self._forward_actor_critic_and_backward(
                        modules=_primary_modules,
                        frozen=_primary_frozen,
                        scaler=self._scaler,
                        intermediates=intermediates,
                        ret=ret,
                        ret_norm_state=ret_norm_state,
                        loss_scale=loss_scale,
                        partial_losses=losses,
                    )
                mets.update(part_mets)
                all_results.append((i, stoch, deter))
            torch.cuda.nvtx.range_pop()

            # Replica GPUs Part B
            for gpu_idx, mi, ret, intermediates, losses, part_mets in part_a_results_replica:
                ri = gpu_idx - 1
                replica = self._replicas[ri]
                rep_scaler = self._replica_scalers[ri]
                dev = replica["device"]
                _ac_fn = self._forward_ac_per_replica[ri] if hasattr(self, "_forward_ac_per_replica") else self._forward_actor_critic_and_backward
                torch.cuda.nvtx.range_push(f"part_b_replica_{ri}")
                with autocast(device_type=dev.type, dtype=torch.float16):
                    (stoch, deter), mets = _ac_fn(
                        modules={
                            "encoder": replica["encoder"],
                            "rssm": replica["rssm"],
                            "actor": replica["actor"],
                            "value": replica["value"],
                            "reward": replica["reward"],
                            "cont": replica["cont"],
                            "decoder": replica.get("decoder"),
                            "prj": replica.get("prj"),
                        },
                        frozen=replica["frozen"],
                        scaler=rep_scaler,
                        intermediates=intermediates,
                        ret=ret,
                        ret_norm_state=ret_norm_state,
                        loss_scale=loss_scale,
                        partial_losses=losses,
                    )
                mets.update(part_mets)
                all_results.append((mi, stoch.to(self.train_device), deter.to(self.train_device)))
                torch.cuda.nvtx.range_pop()
            _timer.stop("part_b")
            torch.cuda.nvtx.range_pop()  # part_b

            # == Sync 2: wait for all GPUs ==
            _timer.start("sync_grad")
            for dev in self.train_devices:
                torch.cuda.synchronize(dev)
            _timer.stop("sync_grad")

            # Collect results in original micro-batch order.
            all_results.sort(key=lambda x: x[0])
            for _, stoch, deter in all_results:
                all_stoch.append(stoch)
                all_deter.append(deter)

        # --- Optimizer step (runs on primary training GPU) ---
        torch.cuda.nvtx.range_push("reduce_gradients")
        _timer.start("reduce_grad")
        self._scaler.unscale_(self._optimizer)  # unscale primary grads
        # Reduce replica gradients AFTER primary unscale so both sides are
        # in the same unscaled fp32 space.  No-op when data_parallel=False.
        # Returns True if any replica had inf — primary scaler won't know.
        _replica_inf = self._reduce_gradients()
        _timer.stop("reduce_grad")
        torch.cuda.nvtx.range_pop()
        if _replica_inf:
            # A replica produced inf grads.  Zero primary grads and skip step
            # (mirrors what GradScaler.step does when it detects inf itself).
            self._optimizer.zero_grad(set_to_none=True)
        else:
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
            torch.cuda.nvtx.range_push("optimizer_step")
            _timer.start("optimizer")
            self._scaler.step(self._optimizer)  # update params
        # update/scheduler/zero always run — matches single-GPU behaviour
        # where scaler.step() may skip internally but the rest proceeds.
        self._scaler.update()  # adjust scale
        self._scheduler.step()  # increment scheduler
        if not _replica_inf:
            _timer.stop("optimizer")
            torch.cuda.nvtx.range_pop()
            self._optimizer.zero_grad(set_to_none=True)  # reset grads
        mets["opt/lr"] = self._scheduler.get_lr()[0]
        mets["opt/grad_scale"] = self._scaler.get_scale()
        if self._log_grads:
            updates = [(new - old) for (new, old) in zip(self._named_params.values(), old_params)]
            update_rms = tools.compute_rms(updates)
            params_rms = tools.compute_rms(self._named_params.values())
            mets["opt/param_rms"] = params_rms
            mets["opt/update_rms"] = update_rms

        # Update replica scalers.
        for rep_scaler in self._replica_scalers:
            rep_scaler.update()

        # --- Sync weights ---
        torch.cuda.nvtx.range_push("broadcast_params")
        _timer.start("broadcast")
        if self.data_parallel:
            self._broadcast_params()       # primary -> replicas
        _timer.stop("broadcast")
        torch.cuda.nvtx.range_pop()
        # Signal that new weights are available for inference copies.
        # The main thread calls sync_inference_if_needed() to pull them
        # before the next act() call, avoiding blocking here.
        self._weights_version += 1

        # Total update wall-clock time.
        _total_ms = (time.perf_counter() - _t0) * 1000
        # Collect GPU timings (safe after the sync barriers above).
        _gpu_times = _timer.elapsed_ms()
        for k, v in _gpu_times.items():
            mets[f"timing/{k}_ms"] = v
        mets["timing/total_update_ms"] = _total_ms

        # Update latent vectors in replay buffer with concatenated posteriors.
        # When trajectory mirroring is active the batch is doubled (original +
        # mirrored), but only the original half has valid storage indices.
        all_stoch = torch.cat(all_stoch, dim=0)
        all_deter = torch.cat(all_deter, dim=0)
        orig_B = getattr(replay_buffer, "_original_batch_size", all_stoch.shape[0])
        replay_buffer.update(index, all_stoch[:orig_B].detach(), all_deter[:orig_B].detach())
        return mets

    def _cal_grad(self, data, initial, loss_scale=1.0, opp_initial=None):
        """Backward pass on primary GPU using self's modules."""
        return self._cal_grad_with_modules(
            modules={
                "encoder": self.encoder,
                "rssm": self.rssm,
                "actor": self.actor,
                "value": self.value,
                "reward": self.reward,
                "cont": self.cont,
                "decoder": getattr(self, "decoder", None),
                "prj": getattr(self, "prj", None),
            },
            frozen={
                "encoder": self._frozen_encoder,
                "rssm": self._frozen_rssm,
                "reward": self._frozen_reward,
                "cont": self._frozen_cont,
                "actor": self._frozen_actor,
                "value": self._frozen_value,
                "slow_value": self._frozen_slow_value,
            },
            opp_modules={
                "rssm": self._imag_opp_rssm,
                "actor": self._imag_opp_actor,
            },
            scaler=self._scaler,
            data=data,
            initial=initial,
            loss_scale=loss_scale,
            opp_initial=opp_initial,
            ret_norm_state=None,
        )

    def _cal_grad_on_replica(self, replica, replica_scaler, data, initial,
                             loss_scale, ret_norm_state, opp_initial=None):
        """Run _cal_grad_with_modules on a replica's device."""
        dev = replica["device"]
        data_dev = data.to(dev)
        initial_dev = (initial[0].to(dev), initial[1].to(dev))
        opp_initial_dev = None
        if opp_initial is not None:
            opp_initial_dev = (opp_initial[0].to(dev), opp_initial[1].to(dev))

        # Use the per-replica compiled function (if available) to avoid
        # cross-device inductor state overhead from the primary's compilation.
        # Determine replica index from device.
        per_replica = getattr(self, "_cal_grad_per_replica", None)
        if per_replica is not None:
            ri = next(i for i, r in enumerate(self._replicas) if r["device"] == dev)
            fn = per_replica[ri]
        else:
            fn = self._cal_grad_with_modules
        return fn(
            modules={
                "encoder": replica["encoder"],
                "rssm": replica["rssm"],
                "actor": replica["actor"],
                "value": replica["value"],
                "reward": replica["reward"],
                "cont": replica["cont"],
                "decoder": replica.get("decoder"),
                "prj": replica.get("prj"),
            },
            frozen=replica["frozen"],
            opp_modules={
                "rssm": replica.get("imag_opp_rssm"),
                "actor": replica.get("imag_opp_actor"),
            },
            scaler=replica_scaler,
            data=data_dev,
            initial=initial_dev,
            loss_scale=loss_scale,
            opp_initial=opp_initial_dev,
            ret_norm_state=ret_norm_state,
        )

    def _cal_grad_with_modules(
        self, modules, frozen, opp_modules, scaler,
        data, initial, loss_scale, opp_initial=None,
        ret_norm_state=None,
    ):
        """Core training computation, parameterised by module set.

        Parameters
        ----------
        modules : dict
            Trainable modules: encoder, rssm, actor, value, reward, cont,
            and optionally decoder, prj.
        frozen : dict
            Frozen copies: encoder, rssm, reward, cont, actor, value, slow_value.
        opp_modules : dict
            Opponent imagination modules: rssm, actor (may be None).
        scaler : GradScaler
            Per-device gradient scaler.
        ret_norm_state : tuple[float, float] | None
            Pre-computed ``(ret_offset, ret_scale)`` from primary's
            ``return_ema``.  If ``None``, calls ``self.return_ema`` directly
            (primary GPU path).
        """
        encoder = modules["encoder"]
        _rssm = modules["rssm"]
        actor = modules["actor"]
        value = modules["value"]
        reward = modules["reward"]
        cont = modules["cont"]
        decoder = modules.get("decoder")
        prj = modules.get("prj")
        frozen_rssm = frozen["rssm"]
        frozen_actor = frozen["actor"]
        frozen_reward = frozen["reward"]
        frozen_cont = frozen["cont"]
        frozen_value = frozen["value"]
        frozen_slow_value = frozen["slow_value"]
        imag_opp_rssm = opp_modules.get("rssm")
        imag_opp_actor = opp_modules.get("actor")
        dev = data.device if hasattr(data, "device") else next(encoder.parameters()).device

        # data: dict of (B, T, *), initial: (stoch: (B, S, K), deter: (B, D))
        losses = {}
        metrics = {}
        B, T = data.shape

        # === World model: posterior rollout and KL losses ===
        # (B, T, E)
        embed = encoder(data)
        # Build world-model action: 4D (player + opponent) when opponent_separation is on.
        if self.opponent_separation:
            wm_action = torch.cat([data["action"], data["opponent_action"]], dim=-1)  # (B, T, 4)
        else:
            wm_action = data["action"]  # (B, T, 2)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        post_stoch, post_deter, post_logit = _rssm.observe(embed, wm_action, initial, data["is_first"])
        # (B, T, S, K)
        _, prior_logit = _rssm.prior(post_deter)
        dyn_loss, rep_loss = _rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = torch.mean(dyn_loss)
        losses["rep"] = torch.mean(rep_loss)
        # === Representation / auxiliary losses ===
        # (B, T, F)
        feat = _rssm.get_feat(post_stoch, post_deter)
        if self.rep_loss == "dreamer":
            recon_losses = {
                key: torch.mean(-dist.log_prob(data[key])) for key, dist in decoder(post_stoch, post_deter).items()
            }
            losses.update(recon_losses)
        elif self.rep_loss == "r2dreamer":
            # R2-Dreamer: Barlow Twins style redundancy reduction between latent features and encoder embeddings.
            # Flatten batch/time dims for a single cross-correlation matrix.
            # (B, T, F) -> (B*T, F)
            x1 = prj(feat[:, :].reshape(B * T, -1))
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
            x1 = prj(feat[:, :].reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important
            logits = torch.matmul(x1, x2.T)
            norm_logits = logits - torch.max(logits, 1)[0][:, None]
            labels = torch.arange(norm_logits.shape[0]).long().to(dev)
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

            embed_aug = encoder(data_aug)
            post_stoch_aug, post_deter_aug, _ = _rssm.observe(
                embed_aug, data_aug["action"], initial_aug, data_aug["is_first"]
            )
            proto_losses = self.proto_loss(post_stoch_aug, post_deter_aug, embed_aug, ema_proj)
            losses.update(proto_losses)
        else:
            raise NotImplementedError

        # reward and continue
        losses["rew"] = torch.mean(-reward(feat).log_prob(to_f32(data["reward"])))
        cont_target = 1.0 - to_f32(data["is_terminal"])
        losses["con"] = torch.mean(-cont(feat).log_prob(cont_target))
        # log
        metrics["dyn_entropy"] = torch.mean(_rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(_rssm.get_dist(post_logit).entropy())

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
                # Per-timestep opponent states from buffer: (B, T, ...) -> (B*T, ...)
                opp_start = (
                    data["opp_stoch"].reshape(-1, *data["opp_stoch"].shape[2:]).detach(),
                    data["opp_deter"].reshape(-1, *data["opp_deter"].shape[2:]).detach(),
                )
        # (B, T, ...) -> (B*T, ...)
        imag_feat, imag_action = self._imagine_with_modules(
            start, self.imag_horizon + 1,
            frozen_rssm=frozen_rssm,
            frozen_actor=frozen_actor,
            imag_opp_rssm=imag_opp_rssm,
            imag_opp_actor=imag_opp_actor,
            opp_start=opp_start,
        )
        imag_feat, imag_action = imag_feat.detach(), imag_action.detach()

        # (B*T, T_imag, 1)
        imag_reward = frozen_reward(imag_feat).mode()
        # (B*T, T_imag, 1)  probability of continuation
        imag_cont = frozen_cont(imag_feat).mean
        # (B*T, T_imag, 1)
        imag_value = frozen_value(imag_feat).mode()
        imag_slow_value = frozen_slow_value(imag_feat).mode()
        disc = 1 - 1 / self.horizon
        # (B*T, T_imag, 1)
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(
            last, term, imag_reward, imag_value, imag_value, disc, self.lamb
        )  # (B*T, T_imag-1, 1)
        # Decision 5: use pre-computed return_ema state or compute on primary.
        if ret_norm_state is not None:
            ret_offset, ret_scale = ret_norm_state
        else:
            ret_offset, ret_scale = self.return_ema(ret)
        # (B*T, T_imag-1, 1)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = actor(imag_feat)
        # (B*T, T_imag-1, 1)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))
        # SkyDreamer smoothness regularization on policy mean only (keeps entropy bonus on σ untouched).
        mu = policy.mean
        mu_diff = mu[:, 1:] - mu[:, :-1]
        smoothness = (mu_diff ** 2).sum(dim=-1, keepdim=True)
        losses["smoothness"] = torch.mean(weight[:, :-1].detach() * smoothness)

        imag_value_dist = value(imag_feat)
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
        if ret_norm_state is not None:
            metrics["ret_005"] = ret_norm_state[0]
            metrics["ret_095"] = ret_norm_state[1]
        else:
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
        metrics["action_mean_delta_l2"] = torch.mean(smoothness)
        metrics.update(tools.tensorstats(imag_action, "action"))

        # === Replay-based value learning (keep gradients through world model) ===
        last, term, reward_data = (
            to_f32(data["is_last"]),
            to_f32(data["is_terminal"]),
            to_f32(data["reward"]),
        )
        feat = _rssm.get_feat(post_stoch, post_deter)
        boot = ret[:, 0].reshape(B, T, 1)
        value_replay = frozen_value(feat).mode()
        slow_value_replay = frozen_slow_value(feat).mode()
        disc = 1 - 1 / self.horizon
        weight = 1.0 - last
        ret = self._lambda_return(last, term, reward_data, value_replay, boot, disc, self.lamb)
        ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)

        # Keep this attached to the world model so gradients can flow through
        value_dist = value(feat)
        losses["repval"] = torch.mean(
            weight[:, :-1]
            * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value_replay.detach()))[:, :-1].unsqueeze(
                -1
            )
        )
        # log
        metrics.update(tools.tensorstats(ret, "ret_replay"))
        metrics.update(tools.tensorstats(value_replay, "value_replay"))
        metrics.update(tools.tensorstats(slow_value_replay, "slow_value_replay"))

        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        scaler.scale(total_loss * loss_scale).backward()

        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        return (post_stoch, post_deter), metrics

    # ------------------------------------------------------------------
    # Split versions of _cal_grad_with_modules for two-sync parallelism.
    # Part A: world model + imagination → produces `ret` tensor.
    # Part B: actor-critic losses + backward (needs ret_norm_state from sync).
    # Used only in the data-parallel multi-GPU path.
    # ------------------------------------------------------------------

    def _forward_world_model(
        self, modules, frozen, opp_modules,
        data, initial, opp_initial=None,
    ):
        """Part A: world model forward + imagination rollout.

        Runs encoder → RSSM → decoder/aux losses → imagination → lambda returns.
        Everything up to (but NOT including) return_ema / advantage computation.

        Returns
        -------
        ret : Tensor
            Lambda returns from imagination, shape ``(B*T, imag_horizon-1, 1)``.
        intermediates : dict
            All tensors needed by Part B, staying on the same device.
        partial_losses : dict
            World model losses (dyn, rep, decoder, rew, con).
        partial_metrics : dict
            World model metrics.
        """
        encoder = modules["encoder"]
        _rssm = modules["rssm"]
        reward = modules["reward"]
        cont = modules["cont"]
        decoder = modules.get("decoder")
        prj = modules.get("prj")
        frozen_rssm = frozen["rssm"]
        frozen_actor = frozen["actor"]
        frozen_reward = frozen["reward"]
        frozen_cont = frozen["cont"]
        frozen_value = frozen["value"]
        frozen_slow_value = frozen["slow_value"]
        imag_opp_rssm = opp_modules.get("rssm")
        imag_opp_actor = opp_modules.get("actor")
        dev = data.device if hasattr(data, "device") else next(encoder.parameters()).device

        losses = {}
        metrics = {}
        B, T = data.shape

        # === World model: posterior rollout and KL losses ===
        embed = encoder(data)
        if self.opponent_separation:
            wm_action = torch.cat([data["action"], data["opponent_action"]], dim=-1)
        else:
            wm_action = data["action"]
        post_stoch, post_deter, post_logit = _rssm.observe(embed, wm_action, initial, data["is_first"])
        _, prior_logit = _rssm.prior(post_deter)
        dyn_loss, rep_loss = _rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = torch.mean(dyn_loss)
        losses["rep"] = torch.mean(rep_loss)

        # === Representation / auxiliary losses ===
        feat = _rssm.get_feat(post_stoch, post_deter)
        if self.rep_loss == "dreamer":
            recon_losses = {
                key: torch.mean(-dist.log_prob(data[key])) for key, dist in decoder(post_stoch, post_deter).items()
            }
            losses.update(recon_losses)
        elif self.rep_loss == "r2dreamer":
            x1 = prj(feat[:, :].reshape(B * T, -1))
            x2 = embed.reshape(B * T, -1).detach()
            x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)
            c = torch.mm(x1_norm.T, x2_norm) / (B * T)
            invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
            off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
            redundancy_loss = c[off_diag_mask].pow(2).sum()
            losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss
        elif self.rep_loss == "infonce":
            x1 = prj(feat[:, :].reshape(B * T, -1))
            x2 = embed.reshape(B * T, -1).detach()
            logits = torch.matmul(x1, x2.T)
            norm_logits = logits - torch.max(logits, 1)[0][:, None]
            labels = torch.arange(norm_logits.shape[0]).long().to(dev)
            losses["infonce"] = torch.nn.functional.cross_entropy(norm_logits, labels)
        else:
            raise NotImplementedError(f"rep_loss={self.rep_loss!r} not supported in split path")

        losses["rew"] = torch.mean(-reward(feat).log_prob(to_f32(data["reward"])))
        cont_target = 1.0 - to_f32(data["is_terminal"])
        losses["con"] = torch.mean(-cont(feat).log_prob(cont_target))
        metrics["dyn_entropy"] = torch.mean(_rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(_rssm.get_dist(post_logit).entropy())

        # === Imagination rollout for actor-critic ===
        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
        )
        opp_start = None
        if self.opponent_separation and self._imag_opponent == "selfplay":
            if "opp_stoch" in data:
                opp_start = (
                    data["opp_stoch"].reshape(-1, *data["opp_stoch"].shape[2:]).detach(),
                    data["opp_deter"].reshape(-1, *data["opp_deter"].shape[2:]).detach(),
                )
        imag_feat, imag_action = self._imagine_with_modules(
            start, self.imag_horizon + 1,
            frozen_rssm=frozen_rssm,
            frozen_actor=frozen_actor,
            imag_opp_rssm=imag_opp_rssm,
            imag_opp_actor=imag_opp_actor,
            opp_start=opp_start,
        )
        imag_feat, imag_action = imag_feat.detach(), imag_action.detach()

        imag_reward = frozen_reward(imag_feat).mode()
        imag_cont = frozen_cont(imag_feat).mean
        imag_value = frozen_value(imag_feat).mode()
        imag_slow_value = frozen_slow_value(imag_feat).mode()
        disc = 1 - 1 / self.horizon
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(
            last, term, imag_reward, imag_value, imag_value, disc, self.lamb
        )

        # Pack everything Part B needs.
        intermediates = {
            "post_stoch": post_stoch, "post_deter": post_deter,
            "imag_feat": imag_feat, "imag_action": imag_action,
            "imag_reward": imag_reward, "imag_value": imag_value,
            "imag_slow_value": imag_slow_value,
            "imag_cont": imag_cont, "weight": weight,
            "data": data, "B": B, "T": T,
        }
        return ret, intermediates, losses, metrics

    def _forward_actor_critic_and_backward(
        self, modules, frozen, scaler,
        intermediates, ret, ret_norm_state, loss_scale, partial_losses,
    ):
        """Part B: actor-critic losses + backward pass.

        Uses ``ret_norm_state`` (from the cross-GPU return_ema sync) to
        normalize advantages, then computes policy/value/replay-value losses,
        sums everything, and calls ``backward()``.

        Returns
        -------
        posteriors : tuple[Tensor, Tensor]
            ``(post_stoch, post_deter)`` for buffer update.
        metrics : dict
            All training metrics (world model + actor-critic).
        """
        _rssm = modules["rssm"]
        actor = modules["actor"]
        value = modules["value"]
        frozen_value = frozen["value"]
        frozen_slow_value = frozen["slow_value"]

        post_stoch = intermediates["post_stoch"]
        post_deter = intermediates["post_deter"]
        imag_feat = intermediates["imag_feat"]
        imag_action = intermediates["imag_action"]
        imag_value = intermediates["imag_value"]
        imag_slow_value = intermediates["imag_slow_value"]
        weight = intermediates["weight"]
        data = intermediates["data"]
        B = intermediates["B"]
        T = intermediates["T"]

        losses = dict(partial_losses)  # copy so we don't mutate caller's dict
        metrics = {}

        # === Actor-critic with synced return normalization ===
        ret_offset, ret_scale = ret_norm_state
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = actor(imag_feat)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))
        # SkyDreamer smoothness regularization on policy mean only (keeps entropy bonus on σ untouched).
        mu = policy.mean
        mu_diff = mu[:, 1:] - mu[:, :-1]
        smoothness = (mu_diff ** 2).sum(dim=-1, keepdim=True)
        losses["smoothness"] = torch.mean(weight[:, :-1].detach() * smoothness)

        imag_value_dist = value(imag_feat)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        losses["value"] = torch.mean(
            weight[:, :-1].detach()
            * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[
                :, :-1
            ].unsqueeze(-1)
        )

        # Metrics
        ret_normed = (ret - ret_offset) / ret_scale
        metrics["ret"] = torch.mean(ret_normed)
        metrics["ret_005"] = ret_norm_state[0]
        metrics["ret_095"] = ret_norm_state[1]
        metrics["adv"] = torch.mean(adv)
        metrics["adv_std"] = torch.std(adv)
        metrics["con"] = torch.mean(intermediates["imag_cont"])
        metrics["rew"] = torch.mean(intermediates["imag_reward"])
        metrics["val"] = torch.mean(imag_value)
        metrics["tar"] = torch.mean(ret)
        metrics["slowval"] = torch.mean(imag_slow_value)
        metrics["weight"] = torch.mean(weight)
        metrics["action_entropy"] = torch.mean(entropy)
        metrics["action_mean_delta_l2"] = torch.mean(smoothness)
        metrics.update(tools.tensorstats(imag_action, "action"))

        # === Replay-based value learning (keep gradients through world model) ===
        last, term, reward_data = (
            to_f32(data["is_last"]),
            to_f32(data["is_terminal"]),
            to_f32(data["reward"]),
        )
        feat = _rssm.get_feat(post_stoch, post_deter)
        boot = ret[:, 0].reshape(B, T, 1)
        value_replay = frozen_value(feat).mode()
        slow_value_replay = frozen_slow_value(feat).mode()
        disc = 1 - 1 / self.horizon
        weight_replay = 1.0 - last
        ret_replay = self._lambda_return(last, term, reward_data, value_replay, boot, disc, self.lamb)
        ret_padded = torch.cat([ret_replay, 0 * ret_replay[:, -1:]], 1)

        value_dist = value(feat)
        losses["repval"] = torch.mean(
            weight_replay[:, :-1]
            * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value_replay.detach()))[:, :-1].unsqueeze(
                -1
            )
        )
        metrics.update(tools.tensorstats(ret_replay, "ret_replay"))
        metrics.update(tools.tensorstats(value_replay, "value_replay"))
        metrics.update(tools.tensorstats(slow_value_replay, "slow_value_replay"))

        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        scaler.scale(total_loss * loss_scale).backward()

        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        return (post_stoch, post_deter), metrics

    @torch.no_grad()
    def _imagine(self, start, imag_horizon, opp_start=None):
        """Roll out the policy in latent space using primary frozen modules."""
        return self._imagine_with_modules(
            start, imag_horizon,
            frozen_rssm=self._frozen_rssm,
            frozen_actor=self._frozen_actor,
            imag_opp_rssm=self._imag_opp_rssm,
            imag_opp_actor=self._imag_opp_actor,
            opp_start=opp_start,
        )

    @torch.no_grad()
    def _imagine_with_modules(
        self, start, imag_horizon,
        frozen_rssm, frozen_actor,
        imag_opp_rssm=None, imag_opp_actor=None,
        opp_start=None,
    ):
        """Roll out the policy in latent space using provided modules.

        Same logic as ``_imagine`` but parameterised so replicas can pass
        their own frozen copies and opponent modules.
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
                opp_stoch, opp_deter = imag_opp_rssm.initial(B)
            opp_prev_action = torch.zeros(B, imag_opp_rssm._act_dim, device=stoch.device)

        for _ in range(imag_horizon):
            # (B, F)
            feat = frozen_rssm.get_feat(stoch, deter)
            # (B, A)
            player_action = frozen_actor(feat).rsample()
            feats.append(feat)
            actions.append(player_action)
            # Build world-model action: concatenate opponent action when separation is on.
            if self.opponent_separation:
                if selfplay:
                    # Opponent acts from its own RSSM state.
                    opp_feat = imag_opp_rssm.get_feat(opp_stoch, opp_deter)
                    opp_action = imag_opp_actor(opp_feat).rsample()
                    # Player world-model action: [player, opponent]
                    wm_action = torch.cat([player_action, opp_action], dim=-1)  # (B, 4)
                    # Opponent RSSM prev_action: [opponent, player] (reversed
                    # perspective, matching DreamerSelfPlayWrapper.step()).
                    opp_prev_action = torch.cat([opp_action, player_action], dim=-1)
                    # Opponent prior transition (no observations in imagination).
                    opp_stoch, opp_deter = imag_opp_rssm.img_step(
                        opp_stoch, opp_deter, opp_prev_action
                    )
                else:
                    opp_action = self._get_imag_opponent_action(feat, player_action)
                    wm_action = torch.cat([player_action, opp_action], dim=-1)  # (B, 4)
            else:
                wm_action = player_action  # (B, 2)
            stoch, deter = frozen_rssm.img_step(stoch, deter, wm_action)

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
