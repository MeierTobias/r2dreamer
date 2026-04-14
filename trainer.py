import queue
import threading

import tools
import torch


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

        All compiled functions were already traced on the main thread
        (via the warmup update call).  This thread only executes the
        cached compiled code — no Dynamo retracing occurs.
        """
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
                try:
                    update_num = self._train_request_queue.get(timeout=0.01)
                except queue.Empty:
                    continue

                metrics = {}
                completed = 0
                for _ in range(update_num):
                    # Check for pause between individual updates so the
                    # main thread doesn't have to wait for the whole batch.
                    if not self._train_resume.is_set():
                        break
                    metrics = agent.update(self.replay_buffer)
                    completed += 1
                if completed > 0:
                    self._train_result_queue.put((completed, metrics))

        except Exception as e:
            self._train_exception = e
        finally:
            self._train_paused.set()

    def _pause_training(self):
        """Pause the training thread and wait until it has stopped."""
        self._train_resume.clear()
        self._train_paused.wait()

    def _resume_training(self):
        """Resume the training thread."""
        self._train_paused.clear()
        self._train_resume.set()

    def _check_train_exception(self):
        """Re-raise any exception from the training thread."""
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
        """
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

        try:
            while self._step < self.steps:
                # --- Collect training results (non-blocking) ---
                if _training_in_flight:
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
                    if _training_in_flight:
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
                    self._fps.reset(self._step)
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
                    if _training_in_flight:
                        _num, _metrics = self._train_result_queue.get()
                        train_metrics = _metrics
                        update_count += _num
                        _training_in_flight = False
                    if _training_started:
                        self._pause_training()
                    self._save_fn(self._step)
                    # Reset FPS baseline so save wall-clock time is excluded.
                    self._fps.reset(self._step)
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
                agent.sync_inference_if_needed()
                trans, done = stepper.step(act.detach(), done.detach())
                act, agent_state = agent.act(trans.clone(), agent_state, eval=False)

                # Store transition.
                trans["action"] = act * ~done.unsqueeze(-1)
                if "opponent_action" in trans:
                    trans["opponent_action"] = trans["opponent_action"] * ~done.unsqueeze(-1)
                trans["stoch"] = agent_state["stoch"]
                trans["deter"] = agent_state["deter"]
                trans["episode"] = episode_ids
                if "image" in trans:
                    video_cache.append(trans["image"][0])
                self.replay_buffer.add_transition(trans.detach())
                returns += trans["reward"][:, 0]

                # Bump episode IDs AFTER storing the transition.
                for _i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                    finished_ep_id = episode_ids[_i].item()
                    episode_ids[_i] = _next_episode_id
                    _next_episode_id += 1
                    self.on_episode_end(finished_ep_id, _i)

                # --- Dispatch training updates ---
                if self.replay_buffer.count() // stepper.env_num > self.batch_length + 1:
                    if not _training_started:
                        # =====================================================
                        # Phase 2: Warmup compile (sequential, main thread)
                        # =====================================================
                        # Run pretrain + one update on the main thread so that
                        # torch.compile traces all compiled functions here.
                        # The training thread will only execute cached code.
                        # Run 10 warmup updates on the main thread so that
                        # torch.compile traces all compiled functions here
                        # (primary GPU + all replicas).  The training thread
                        # will only execute cached compiled code afterwards.
                        _warmup_count = 10
                        print(f"Warmup: {_warmup_count} updates on main thread (triggers torch.compile)...")
                        for _ in range(_warmup_count):
                            train_metrics = agent.update(self.replay_buffer)
                        update_count += _warmup_count
                        # =====================================================
                        # Phase 3: Start training thread
                        # =====================================================
                        train_thread = threading.Thread(
                            target=self._training_loop,
                            args=(agent, stop_event),
                            daemon=True,
                            name="training",
                        )
                        train_thread.start()
                        _training_started = True
                        print("Training thread started.")
                        # Dispatch world-model pretrain updates (if any)
                        # to the training thread.
                        if self._should_pretrain() and self.pretrain > 0:
                            print(f"Pretrain: dispatching {self.pretrain} updates to training thread...")
                            self._train_request_queue.put(self.pretrain)
                            _training_in_flight = True
                    elif not _training_in_flight:
                        update_num = self._updates_needed(self._step)
                        if update_num > 0:
                            self._train_request_queue.put(update_num)
                            _training_in_flight = True

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
                    if self.video_pred_log and not _training_in_flight:
                        # video_pred reads inference copies on sim device.
                        # Only run when training is idle to avoid blocking.
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
                            self._fps.reset(self._step)
                    if self.params_hist_log:
                        for name, param in agent._named_params.items():
                            self.logger.histogram(name, tools.to_np(param))
                    _train_fps = self._fps.compute(self._step)
                    if _train_fps is not None:
                        self.logger.scalar("fps/train", _train_fps)
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
            if train_thread is not None:
                if _training_in_flight:
                    try:
                        self._train_result_queue.get(timeout=30)
                    except queue.Empty:
                        pass
                stop_event.set()
                self._train_resume.set()
                train_thread.join(timeout=10)
