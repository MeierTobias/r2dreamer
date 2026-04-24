import cProfile
import json
import os
import queue
import threading
import time

import tools
import torch
from torch.profiler import ProfilerActivity


class OnlineTrainer:
    def __init__(
        self, config, replay_buffer, logger, logdir, train_stepper, eval_stepper, initial_step=0, save_fn=None
    ):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.logdir = logdir
        self.train_stepper = train_stepper
        self.eval_stepper = eval_stepper
        self.steps = int(config.steps)
        self.pretrain = int(config.pretrain)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.video_pred_log = bool(config.video_pred_log)
        self.params_hist_log = bool(config.params_hist_log)
        self.batch_length = int(config.batch_length)
        batch_steps = int(config.batch_size * config.batch_length)
        # train_ratio is based on data steps rather than environment steps.
        self._updates_needed = tools.Every(batch_steps / config.train_ratio * config.action_repeat)
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(config.update_log_every)
        self._should_eval = tools.Every(self.eval_every)
        self._action_repeat = config.action_repeat
        # Periodic checkpointing
        self._save_fn = save_fn
        self._should_save = (
            tools.Every(int(config.save_checkpoint_every)) if int(config.save_checkpoint_every) > 0 else None
        )
        # Resume state
        if initial_step > 0:
            self._step = initial_step
            self._should_pretrain._once = False
            self._should_eval._last = self._step
            self._should_log._last = self._step + self._action_repeat
            self._updates_needed._last = self._step
            if self._should_save is not None:
                self._should_save._last = self._step
        else:
            self._step = 0
        self._fps = tools.FPSTracker()
        # Profiling config (optional section in trainer config).
        _prof_raw = getattr(config, "profiling", None)
        if _prof_raw is not None:
            try:
                from omegaconf import OmegaConf
                self._profiler_cfg = OmegaConf.to_container(_prof_raw, resolve=True) if OmegaConf.is_config(_prof_raw) else dict(_prof_raw)
            except ImportError:
                self._profiler_cfg = dict(_prof_raw) if _prof_raw else {}
        else:
            self._profiler_cfg = {}

    def on_episode_end(self, episode_id: int, env_index: int) -> None:
        """Called when an episode ends.  Override to tag episodes in the buffer."""
        pass

    def on_log(self) -> None:
        """Called at each logging step.  Override to add custom metrics."""
        pass

    def eval(self, agent, train_step):
        """Run evaluation episodes.

        Device handling is delegated to ``self.eval_stepper``.
        """
        print("Evaluating the policy...")
        stepper = self.eval_stepper
        # Reset all environments so eval always starts from fresh episodes.
        stepper.reset()
        agent.eval()
        # (B,)
        done = torch.ones(stepper.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(stepper.env_num, dtype=torch.bool, device=agent.device)
        steps = torch.zeros(stepper.env_num, dtype=torch.int32, device=agent.device)
        returns = torch.zeros(stepper.env_num, dtype=torch.float32, device=agent.device)
        log_metrics = {}
        # cache is only used for video logging / open-loop prediction.
        cache = []
        agent_state = agent.get_initial_state(stepper.env_num)
        # (B, A)
        act = agent_state["action"].clone()
        _eval_iters = 0
        while not once_done.all():
            _eval_iters += 1
            steps += ~done * ~once_done
            # Step environments via the stepper (handles device transfers).
            trans, done = stepper.step(act.detach(), done.detach())

            # Store transition.
            # We keep the observation and the action that produced it together.
            trans["action"] = act
            if len(cache) < self.batch_length:
                cache.append(trans.clone())
            # (B, A)
            act, agent_state = agent.act(trans, agent_state, eval=True)
            returns += trans["reward"][:, 0] * ~once_done
            for key, value in trans.items():
                if key.startswith("log_"):
                    if key not in log_metrics:
                        log_metrics[key] = torch.zeros_like(returns)
                    log_metrics[key] += value[:, 0] * ~once_done
            once_done |= done
        # dict of (B, T, *)
        cache = torch.stack(cache, dim=1) if len(cache) else None
        self.logger.scalar("episode/eval_score", returns.mean())
        self.logger.scalar("episode/eval_length", steps.to(torch.float32).mean())
        for key, value in log_metrics.items():
            if key == "log_success":
                value = torch.clip(value, max=1.0)  # make sure 1.0 for success episode
            self.logger.scalar(f"episode/eval_{key[4:]}", value.mean())
        if cache is not None and "image" in cache:
            self.logger.video("eval_video", tools.to_np(cache["image"][:1]))
        if self.video_pred_log and cache is not None:
            initial = agent.get_initial_state(1)
            vp = agent.video_pred(
                cache[:1],  # give only first batch
                (initial["stoch"], initial["deter"]),
            )
            if vp is not None:
                self.logger.video("eval_open_loop", tools.to_np(vp))
        # Use total env interactions (iters * envs) for FPS so that
        # wall-clock time spent on envs that finished early is accounted for.
        total_eval_steps = _eval_iters * stepper.env_num
        self.logger.write(train_step)
        agent.train()
        return total_eval_steps

    # ------------------------------------------------------------------
    # Async training thread
    # ------------------------------------------------------------------

    def _training_loop(self, agent, stop_event):
        """Background thread: runs agent.update() on cuda:1..N.

        Only started when ``agent.multi_gpu`` is True (sim and training on
        disjoint devices).  All compiled functions were already traced on the
        main thread (via the warmup update call).  This thread only executes
        the cached compiled code — no Dynamo retracing occurs.
        """
        # --- Profiling setup ---
        _profiler_cfg = getattr(self, "_profiler_cfg", {})
        _jsonl_path = os.path.join(str(self.logdir), "profiler", "update_timing.jsonl")
        os.makedirs(os.path.dirname(_jsonl_path), exist_ok=True)
        _jsonl_file = open(_jsonl_path, "a")
        _update_idx = 0

        # torch.profiler (Level 2)
        _torch_profiler = None
        if _profiler_cfg.get("torch_profiler", True):
            _prof_dir = os.path.join(str(self.logdir), "profiler")
            os.makedirs(_prof_dir, exist_ok=True)

            def _trace_handler(p):
                p.export_chrome_trace(os.path.join(_prof_dir, f"trace_update_{p.step_num}.json"))
                summary = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
                with open(os.path.join(_prof_dir, f"summary_update_{p.step_num}.txt"), "w") as f:
                    f.write(summary)
            _torch_profiler = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    skip_first=100,
                    wait=int(_profiler_cfg.get("torch_profiler_wait", 2000)),
                    warmup=2,
                    active=int(_profiler_cfg.get("torch_profiler_active", 5)),
                    repeat=0,
                ),
                on_trace_ready=_trace_handler,
                with_stack=True,
                record_shapes=True,
                profile_memory=True,
            )
            _torch_profiler.__enter__()

        # cProfile (Level 3)
        _cprofile = None
        _cprofile_every = int(_profiler_cfg.get("cprofile_every", 50000))
        if _profiler_cfg.get("cprofile", True):
            _cprofile = cProfile.Profile()
            _cprofile.enable()

        try:
            while not stop_event.is_set():
                # Pause/resume protocol.
                if not self._train_resume.is_set():
                    self._train_paused.set()
                    self._train_resume.wait()
                    self._train_paused.clear()
                    if stop_event.is_set():
                        break

                # Wait for an update request from the main thread.
                _queue_wait_start = time.perf_counter()
                try:
                    update_num = self._train_request_queue.get(timeout=0.01)
                except queue.Empty:
                    continue
                _queue_wait_ms = (time.perf_counter() - _queue_wait_start) * 1000

                metrics = {}
                completed = 0
                _batch_queue_wait_ms = _queue_wait_ms  # preserve for result
                for _ in range(update_num):
                    # Check for pause between individual updates so the
                    # main thread doesn't have to wait for the whole batch.
                    if not self._train_resume.is_set():
                        break
                    metrics = agent.update(self.replay_buffer)
                    _update_idx += 1
                    completed += 1
                    # Write per-update timing to JSONL (Level 5).
                    _timing = {k: v for k, v in metrics.items() if k.startswith("timing/")}
                    if _timing:
                        _timing["step"] = self._step
                        _timing["update_idx"] = _update_idx
                        _timing["queue_wait_ms"] = _queue_wait_ms
                        _jsonl_file.write(json.dumps(_timing, default=float) + "\n")
                        _queue_wait_ms = 0  # only first update in batch waited
                    if _torch_profiler is not None:
                        _torch_profiler.step()
                if completed > 0:
                    metrics["timing/train_queue_wait_ms"] = _batch_queue_wait_ms
                    self._train_result_queue.put((completed, metrics))

                # Periodic JSONL flush.
                if _update_idx % 100 == 0:
                    _jsonl_file.flush()

                # Periodic cProfile dump (Level 3).
                if _cprofile is not None and _update_idx % _cprofile_every == 0 and _update_idx > 0:
                    _prof_path = os.path.join(str(self.logdir), "profiler", f"cprofile_train_{_update_idx}.prof")
                    _cprofile.dump_stats(_prof_path)
                    print(f"cProfile dump: {_prof_path}")

        except Exception as e:
            self._train_exception = e
        finally:
            # Clean up profilers.
            if _torch_profiler is not None:
                _torch_profiler.__exit__(None, None, None)
            if _cprofile is not None:
                _cprofile.disable()
                _prof_path = os.path.join(str(self.logdir), "profiler", "cprofile_train_final.prof")
                os.makedirs(os.path.dirname(_prof_path), exist_ok=True)
                _cprofile.dump_stats(_prof_path)
                print(f"cProfile final dump: {_prof_path}")
            _jsonl_file.flush()
            _jsonl_file.close()
            self._train_paused.set()

    def _pause_training(self):
        """Pause the training thread and wait until it has stopped."""
        if not self._async_training:
            return
        self._train_resume.clear()
        self._train_paused.wait()

    def _resume_training(self):
        """Resume the training thread."""
        if not self._async_training:
            return
        self._train_paused.clear()
        self._train_resume.set()

    def _check_train_exception(self):
        """Re-raise any exception from the training thread."""
        if not self._async_training:
            return
        exc = self._train_exception
        if exc is not None:
            self._train_exception = None
            raise RuntimeError("Training thread failed") from exc

    def begin(self, agent):
        """Main training loop with async training.

        The main thread runs env stepping + inference on cuda:0 (IsaacSim
        requires main-thread).  A background thread runs agent.update on
        cuda:1..N.  All torch.compile compilation is done on the main thread
        before the training thread starts.

        When ``agent.multi_gpu`` is False (sim and training share a device),
        async training is disabled and ``agent.update()`` runs inline on the
        main thread.  Concurrent CUDA-graph replay (from compile mode
        ``reduce-overhead``) and RNG use from ``act()`` on the same device
        otherwise trigger "Offset increment outside graph capture" errors.
        """
        self._async_training = bool(getattr(agent, "multi_gpu", False))
        stepper = self.train_stepper
        video_cache = []
        if self._step == 0:
            self._step = self.replay_buffer.count() * self._action_repeat
        update_count = 0
        # (B,)
        done = torch.ones(stepper.env_num, dtype=torch.bool, device=agent.device)
        returns = torch.zeros(stepper.env_num, dtype=torch.float32, device=agent.device)
        lengths = torch.zeros(stepper.env_num, dtype=torch.int32, device=agent.device)
        episode_ids = torch.arange(stepper.env_num, dtype=torch.int32, device=agent.device)
        # Global counter for unique episode IDs — incremented whenever an env
        # resets so SliceSampler never samples across episode boundaries.
        _next_episode_id = stepper.env_num
        train_metrics = {}
        episode_scores: list[float] = []
        episode_lengths: list[float] = []
        stepper.reset()
        agent_state = agent.get_initial_state(stepper.env_num)
        # (B, A)
        act = agent_state["action"].clone()

        # =============================================================
        # Phase 1: Fill buffer (sequential, compiles inference modules)
        # =============================================================
        _training_started = False
        _training_in_flight = False

        # Threading primitives (initialised now, used after Phase 2).
        self._train_resume = threading.Event()
        self._train_resume.set()
        self._train_paused = threading.Event()
        self._train_paused.clear()
        self._train_request_queue = queue.Queue()
        self._train_result_queue = queue.Queue()
        self._train_exception = None
        stop_event = threading.Event()
        train_thread = None

        # Level 1 main-loop timing accumulators (reset each log window).
        _main_timing = {
            "env_step_ms": 0.0, "act_inference_ms": 0.0,
            "buffer_add_ms": 0.0, "sync_inference_ms": 0.0,
            "main_loop_total_ms": 0.0,
        }
        _main_timing_count = 0

        try:
            while self._step < self.steps:
                # --- Collect training results (non-blocking) ---
                if self._async_training and _training_in_flight:
                    try:
                        _num, _metrics = self._train_result_queue.get_nowait()
                        train_metrics = _metrics
                        update_count += _num
                        _training_in_flight = False
                    except queue.Empty:
                        pass
                if _training_started:
                    self._check_train_exception()

                # --- Evaluation ---
                if self._should_eval(self._step) and self.eval_episode_num > 0:
                    # Wait for all in-flight training to finish so metrics
                    # and weights are fully up to date before eval.
                    if self._async_training and _training_in_flight:
                        _num, _metrics = self._train_result_queue.get()
                        train_metrics = _metrics
                        update_count += _num
                        _training_in_flight = False
                    if _training_started:
                        self._pause_training()
                    if hasattr(self.replay_buffer, "flush_all_episodes"):
                        self.replay_buffer.flush_all_episodes()
                    # Separate FPS tracker for eval so it doesn't pollute
                    # the training FPS measurement.
                    _eval_fps_tracker = tools.FPSTracker(warmup=False)
                    _eval_fps_tracker.reset()
                    agent.sync_inference_if_needed()
                    _eval_steps = self.eval(agent, self._step)
                    if _eval_steps is not None:
                        _eval_fps = _eval_fps_tracker.compute(_eval_steps * self._action_repeat)
                        if _eval_fps is not None:
                            self.logger.scalar("fps/eval", _eval_fps)
                    # Reset training FPS baseline so eval wall-clock time
                    # is excluded from the next fps/train measurement.
                    self._fps.reset(self._step, skip_next=True)
                    stepper.reset()
                    done = torch.ones(stepper.env_num, dtype=torch.bool, device=agent.device)
                    returns.zero_()
                    lengths.zero_()
                    agent_state = agent.get_initial_state(stepper.env_num)
                    act = agent_state["action"].clone()
                    episode_ids = torch.arange(
                        _next_episode_id, _next_episode_id + stepper.env_num,
                        dtype=torch.int32, device=agent.device,
                    )
                    _next_episode_id += stepper.env_num
                    video_cache = []
                    if _training_started:
                        self._resume_training()

                # --- Periodic checkpoint saving ---
                if self._save_fn is not None and self._should_save is not None and self._should_save(self._step):
                    if self._async_training and _training_in_flight:
                        _num, _metrics = self._train_result_queue.get()
                        train_metrics = _metrics
                        update_count += _num
                        _training_in_flight = False
                    if _training_started:
                        self._pause_training()
                    self._save_fn(self._step)
                    # Reset FPS baseline so save wall-clock time is excluded.
                    self._fps.reset(self._step, skip_next=True)
                    if _training_started:
                        self._resume_training()

                # --- Collect episode metrics ---
                if done.any():
                    for i, d in enumerate(done):
                        if d and lengths[i] > 0:
                            if i == 0 and len(video_cache) > 0:
                                video = torch.stack(video_cache, axis=0)
                                self.logger.video("train_video", tools.to_np(video[None]))
                                video_cache = []
                            episode_scores.append(returns[i].item())
                            episode_lengths.append(lengths[i].item())
                            returns[i] = lengths[i] = 0
                self._step += stepper.count_active_steps(done) * self._action_repeat
                lengths += ~done

                # --- Env step + inference (main thread, cuda:0) ---
                _loop_t0 = time.perf_counter()

                _t = time.perf_counter()
                agent.sync_inference_if_needed()
                _main_timing["sync_inference_ms"] += (time.perf_counter() - _t) * 1000

                _t = time.perf_counter()
                trans, done = stepper.step(act.detach(), done.detach())
                _main_timing["env_step_ms"] += (time.perf_counter() - _t) * 1000

                _t = time.perf_counter()
                act, agent_state = agent.act(trans.clone(), agent_state, eval=False)
                _main_timing["act_inference_ms"] += (time.perf_counter() - _t) * 1000

                # Store transition.
                trans["action"] = act * ~done.unsqueeze(-1)
                if "opponent_action" in trans:
                    trans["opponent_action"] = trans["opponent_action"] * ~done.unsqueeze(-1)
                trans["stoch"] = agent_state["stoch"]
                trans["deter"] = agent_state["deter"]
                trans["episode"] = episode_ids
                if "image" in trans:
                    video_cache.append(trans["image"][0])
                _t = time.perf_counter()
                self.replay_buffer.add_transition(trans.detach())
                _main_timing["buffer_add_ms"] += (time.perf_counter() - _t) * 1000
                returns += trans["reward"][:, 0]

                # Bump episode IDs AFTER storing the transition.
                for _i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                    finished_ep_id = episode_ids[_i].item()
                    episode_ids[_i] = _next_episode_id
                    _next_episode_id += 1
                    self.on_episode_end(finished_ep_id, _i)

                _main_timing["main_loop_total_ms"] += (time.perf_counter() - _loop_t0) * 1000
                _main_timing_count += 1

                # --- Dispatch training updates ---
                # Matches the original sequential logic exactly:
                #   if buffer has enough data:
                #       update_num = _updates_needed(step)
                #       for _ in range(update_num):
                #           agent.update(replay_buffer)
                # The only difference: agent.update() runs on a background
                # thread.  We block-wait for each batch to finish before
                # dispatching the next, so _step never races ahead and
                # _updates_needed always returns the same count as the
                # original code.  The async benefit: the single env step
                # that follows the dispatch overlaps with the training batch.
                if self.replay_buffer.count() // stepper.env_num > self.batch_length + 1:
                    if not _training_started:
                        # =====================================================
                        # Phase 2: Warmup compile (sequential, main thread)
                        # =====================================================
                        _warmup_count = 10
                        print(f"Warmup: {_warmup_count} updates on main thread (triggers torch.compile)...")
                        for _ in range(_warmup_count):
                            train_metrics = agent.update(self.replay_buffer)
                        update_count += _warmup_count
                        if self._async_training:
                            # =================================================
                            # Phase 3: Start training thread (multi-GPU)
                            # =================================================
                            train_thread = threading.Thread(
                                target=self._training_loop,
                                args=(agent, stop_event),
                                daemon=True,
                                name="training",
                            )
                            train_thread.start()
                            _training_started = True
                            print("Training thread started.")
                            if self._should_pretrain() and self.pretrain > 0:
                                print(f"Pretrain: dispatching {self.pretrain} updates to training thread...")
                                self._train_request_queue.put(self.pretrain)
                                _training_in_flight = True
                        else:
                            # Single-GPU: run updates inline on the main thread.
                            _training_started = True
                            print("Training runs inline on main thread (sim and train share a device).")
                            if self._should_pretrain() and self.pretrain > 0:
                                print(f"Pretrain: running {self.pretrain} updates inline...")
                                for _ in range(self.pretrain):
                                    train_metrics = agent.update(self.replay_buffer)
                                update_count += self.pretrain
                    else:
                        if self._async_training:
                            # Wait for previous batch to finish (same as the
                            # original synchronous code where updates block).
                            if _training_in_flight:
                                _num, _metrics = self._train_result_queue.get()
                                train_metrics = _metrics
                                update_count += _num
                                _training_in_flight = False
                            update_num = self._updates_needed(self._step)
                            if update_num > 0:
                                self._train_request_queue.put(update_num)
                                _training_in_flight = True
                        else:
                            update_num = self._updates_needed(self._step)
                            for _ in range(update_num):
                                train_metrics = agent.update(self.replay_buffer)
                            update_count += update_num

                # --- Log training metrics ---
                if self._should_log(self._step) and train_metrics:
                    if episode_scores:
                        self.logger.scalar("episode/score", sum(episode_scores) / len(episode_scores))
                        self.logger.scalar("episode/length", sum(episode_lengths) / len(episode_lengths))
                        episode_scores.clear()
                        episode_lengths.clear()
                    for name, value in train_metrics.items():
                        value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                        self.logger.scalar(f"train/{name}", value)
                    self.logger.scalar("train/opt/updates", update_count)
                    # Compute FPS before video_pred so its wall-clock time
                    # is excluded from the next interval's measurement (via
                    # the reset below) rather than skipping this one.
                    _train_fps = self._fps.compute(self._step)
                    if _train_fps is not None:
                        self.logger.scalar("fps/train", _train_fps)
                    if self.video_pred_log:
                        # video_pred reads inference copies on sim device.
                        # Pause training to ensure consistent weights during sync.
                        if _training_started:
                            self._pause_training()
                        agent.sync_inference_if_needed()
                        _sample = self.replay_buffer.sample()
                        if _sample is not None:
                            data, _, initial, _opp_initial = _sample
                            vp = agent.video_pred(data, initial)
                            if vp is not None:
                                self.logger.video("open_loop", tools.to_np(vp))
                        if _training_started:
                            self._resume_training()
                            self._fps.reset(self._step, skip_next=True)
                    if self.params_hist_log:
                        for name, param in agent._named_params.items():
                            self.logger.histogram(name, tools.to_np(param))
                    # Log main-loop timing averages (Level 1).
                    if _main_timing_count > 0:
                        for k, v in _main_timing.items():
                            self.logger.scalar(f"timing/{k}", v / _main_timing_count)
                        # Reset accumulators for next window.
                        for k in _main_timing:
                            _main_timing[k] = 0.0
                        _main_timing_count = 0
                    # Buffer lock contention metrics.
                    buf = self.replay_buffer
                    self.logger.scalar("timing/lock_wait_sample_ms", buf.lock_wait_sample_ns / 1e6)
                    self.logger.scalar("timing/lock_wait_add_ms", buf.lock_wait_add_ns / 1e6)
                    self.logger.scalar("timing/lock_contention_count", buf.lock_contention_count)
                    buf.lock_wait_sample_ns = 0
                    buf.lock_wait_add_ns = 0
                    buf.lock_contention_count = 0
                    self.on_log()
                    # Flush logger in a background thread so wandb video
                    # encoding doesn't block env stepping on the main thread.
                    _log_step = self._step
                    threading.Thread(
                        target=self.logger.write,
                        args=(_log_step,),
                        daemon=True,
                    ).start()

        finally:
            if self._async_training and train_thread is not None:
                if _training_in_flight:
                    try:
                        self._train_result_queue.get(timeout=30)
                    except queue.Empty:
                        pass
                stop_event.set()
                self._train_resume.set()
                train_thread.join(timeout=10)
