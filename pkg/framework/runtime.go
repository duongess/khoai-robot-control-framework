package framework

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type RuntimeStatus string

const (
	RuntimeStopped RuntimeStatus = "stopped"
	RuntimeRunning RuntimeStatus = "running"
	RuntimePaused  RuntimeStatus = "paused"
	RuntimeStalled RuntimeStatus = "stalled"
	RuntimeError   RuntimeStatus = "error"
)

type runtimeWorker struct {
	id            int
	task          Task
	episodeID     uint64
	episodeStep   uint64
	state         State
	lastAction    Action
	lastReward    float32
	episodeReward float64
	// episodePolicyVersion is assigned by the first prediction of an episode.
	// Training may continue in parallel, but it cannot change this episode's
	// policy snapshot.
	episodePolicyVersion uint64
	lastInfo             map[string]float32
	outcome              Outcome
	actionSource         string
}

type runtimeMetrics struct {
	totalSteps            uint64
	totalEpisodes         uint64
	successes             uint64
	totalReward           float64
	trainingBatches       uint64
	policyVersion         uint64
	trainingStep          uint64
	actorLoss             float32
	criticLoss            float32
	alphaLoss             float32
	entropy               float32
	criticOneQ            float32
	criticTwoQ            float32
	alpha                 float32
	actorLogStdHorizontal float32
	actorLogStdVertical   float32
	actorLogStdGripper    float32
	startedAt             time.Time
	lastProgressAt        time.Time
	rateStartedAt         time.Time
	rateStartSteps        uint64
	rateStartEpisodes     uint64
	stepsPerSecond        float64
	episodesPerSecond     float64
	actionCount           uint64
	rawActionSum          []float64
	rawActionSquare       []float64
	filteredActionSum     []float64
	filteredActionSquare  []float64
	deadZoneRemoved       []uint64
	filterModified        []uint64
	phaseCounts           map[int]uint64
	failureReasons        map[string]uint64
	contactSamples        uint64
	attachmentSamples     uint64
}

// ActionStatistics makes the effect of task-side filtering visible. It allows
// a collapsed policy to be distinguished from a policy whose commands were
// intentionally removed by a dead zone.
type ActionStatistics struct {
	Samples                 uint64    `json:"samples"`
	RawMean                 []float64 `json:"raw_mean"`
	RawStd                  []float64 `json:"raw_std"`
	FilteredMean            []float64 `json:"filtered_mean"`
	FilteredStd             []float64 `json:"filtered_std"`
	DeadZoneRemovedFraction []float64 `json:"dead_zone_removed_fraction"`
	FilterModifiedFraction  []float64 `json:"filter_modified_fraction"`
}

type WorkerSnapshot struct {
	ID            int                `json:"id"`
	EpisodeID     uint64             `json:"episode_id"`
	EpisodeStep   uint64             `json:"episode_step"`
	State         State              `json:"state"`
	LastAction    Action             `json:"last_action"`
	LastReward    float32            `json:"last_reward"`
	EpisodeReward float64            `json:"episode_reward"`
	PolicyVersion uint64             `json:"policy_version"`
	Info          map[string]float32 `json:"info,omitempty"`
	Outcome       Outcome            `json:"outcome"`
	ActionSource  string             `json:"action_source"`
	Metadata      map[string]any     `json:"metadata,omitempty"`
}

type RuntimeSnapshot struct {
	Status                RuntimeStatus     `json:"status"`
	ActiveWorkers         int               `json:"active_workers"`
	TotalSteps            uint64            `json:"total_steps"`
	TotalEpisodes         uint64            `json:"total_episodes"`
	SuccessRate           float64           `json:"success_rate"`
	AverageReward         float64           `json:"average_reward"`
	ReplayBufferSize      int               `json:"replay_buffer_size"`
	TrainingBatches       uint64            `json:"training_batches"`
	PolicyVersion         uint64            `json:"policy_version"`
	TrainingStep          uint64            `json:"training_step"`
	ActorLoss             float32           `json:"actor_loss"`
	CriticLoss            float32           `json:"critic_loss"`
	AlphaLoss             float32           `json:"alpha_loss"`
	Entropy               float32           `json:"entropy"`
	CriticOneQ            float32           `json:"critic_one_q"`
	CriticTwoQ            float32           `json:"critic_two_q"`
	Alpha                 float32           `json:"alpha"`
	ActorLogStdHorizontal float32           `json:"actor_log_std_horizontal"`
	ActorLogStdVertical   float32           `json:"actor_log_std_vertical"`
	ActorLogStdGripper    float32           `json:"actor_log_std_gripper"`
	StepsPerSecond        float64           `json:"steps_per_second"`
	EpisodesPerSecond     float64           `json:"episodes_per_second"`
	LastError             string            `json:"last_error"`
	ActionStatistics      ActionStatistics  `json:"action_statistics"`
	PhaseOccupancy        map[string]uint64 `json:"phase_occupancy"`
	FailureReasons        map[string]uint64 `json:"failure_reasons"`
	ContactRate           float64           `json:"contact_rate"`
	AttachmentRate        float64           `json:"attachment_rate"`
	Workers               []WorkerSnapshot  `json:"workers"`
}

// Configure attaches the learner and runtime settings before Start.
func (r *Runtime) Configure(config RuntimeConfig, learner Learner) error {
	if r == nil {
		return errors.New("runtime is nil")
	}
	if learner == nil {
		return errors.New("learner is required")
	}
	if config.WorkerCount <= 0 || config.TickInterval <= 0 || config.ReplayCapacity <= 0 || config.RandomActionWarmupTransitions < 0 || config.WarmupTransitions < 0 || config.TrainingBatchSize <= 0 || config.TrainingInterval <= 0 {
		return errors.New("runtime configuration contains invalid values")
	}
	replay, err := NewReplayBuffer(config.ReplayCapacity, config.ReplaySampleSeed)
	if err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.status == RuntimeRunning || r.status == RuntimePaused {
		return errors.New("runtime cannot be configured while active")
	}
	r.config, r.learner, r.replay = config, learner, replay
	r.metrics.actionCount = 0
	r.metrics.rawActionSum = nil
	r.metrics.rawActionSquare = nil
	r.metrics.filteredActionSum = nil
	r.metrics.filteredActionSquare = nil
	r.metrics.deadZoneRemoved = nil
	r.metrics.filterModified = nil
	r.metrics.phaseCounts = make(map[int]uint64)
	r.metrics.failureReasons = make(map[string]uint64)
	r.metrics.contactSamples = 0
	r.metrics.attachmentSamples = 0
	return nil
}

// Start creates independent task instances and starts batched inference.
func (r *Runtime) Start(ctx context.Context) error {
	if r == nil {
		return errors.New("runtime is nil")
	}
	if ctx == nil {
		return errors.New("context is required")
	}
	r.mu.Lock()
	if r.learner == nil {
		r.mu.Unlock()
		return errors.New("runtime learner is not configured")
	}
	if r.status == RuntimeRunning {
		r.mu.Unlock()
		return nil
	}
	if r.status == RuntimePaused {
		r.status = RuntimeRunning
		r.mu.Unlock()
		return nil
	}
	registration, exists := r.tasks[r.activeTaskName]
	if !exists {
		r.mu.Unlock()
		return errors.New("runtime has no active task registration")
	}
	workers := make([]*runtimeWorker, r.config.WorkerCount)
	for index := range workers {
		task, err := registration.Factory.Create()
		if err != nil {
			r.mu.Unlock()
			return fmt.Errorf("create worker %d task: %w", index+1, err)
		}
		state, err := task.Reset()
		if err != nil {
			r.mu.Unlock()
			return fmt.Errorf("reset worker %d task: %w", index+1, err)
		}
		if len(state) != registration.Descriptor.StateDimension {
			r.mu.Unlock()
			return fmt.Errorf("worker %d reset returned state dimension %d, want %d", index+1, len(state), registration.Descriptor.StateDimension)
		}
		workers[index] = &runtimeWorker{id: index + 1, task: task, episodeID: 1, state: append(State(nil), state...), outcome: OutcomeRunning, actionSource: "pending"}
	}
	loopContext, cancel := context.WithCancel(ctx)
	r.workers, r.cancel, r.done, r.status, r.lastError = workers, cancel, make(chan struct{}), RuntimeRunning, ""
	now := time.Now()
	r.metrics.startedAt, r.metrics.lastProgressAt, r.metrics.rateStartedAt = now, now, now
	r.metrics.rateStartSteps, r.metrics.rateStartEpisodes = r.metrics.totalSteps, r.metrics.totalEpisodes
	r.metrics.stepsPerSecond, r.metrics.episodesPerSecond = 0, 0
	done := r.done
	r.mu.Unlock()
	go r.run(loopContext, done, registration.Descriptor)
	return nil
}

func (r *Runtime) run(ctx context.Context, done chan struct{}, descriptor TaskDescriptor) {
	defer close(done)
	ticker := time.NewTicker(r.config.TickInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			r.mu.RLock()
			active := r.status == RuntimeRunning
			r.mu.RUnlock()
			if active {
				r.cycle(ctx, descriptor)
			}
		}
	}
}

func (r *Runtime) cycle(ctx context.Context, descriptor TaskDescriptor) {
	r.mu.RLock()
	useRandomWarmup := r.metrics.totalSteps < uint64(r.config.RandomActionWarmupTransitions)
	warmupSeed := r.config.ReplaySampleSeed + int64(r.metrics.totalSteps)*7919
	groups := make(map[uint64][]int)
	for i, worker := range r.workers {
		groups[worker.episodePolicyVersion] = append(groups[worker.episodePolicyVersion], i)
	}
	learner := r.learner
	r.mu.RUnlock()

	type indexedPrediction struct {
		action  Action
		version uint64
	}
	predictions := make(map[int]indexedPrediction, len(r.workers))
	if useRandomWarmup {
		for index := range r.workers {
			predictions[index] = indexedPrediction{
				action:  randomAction(warmupSeed+int64(index)*104729, descriptor.ActionDimension, descriptor.ActionMin, descriptor.ActionMax),
				version: 0,
			}
		}
	}
	for requestedVersion, indexes := range groups {
		if useRandomWarmup {
			break
		}
		states := make([]State, len(indexes))
		r.mu.RLock()
		for position, index := range indexes {
			states[position] = append(State(nil), r.workers[index].state...)
		}
		r.mu.RUnlock()
		prediction, err := learner.PredictBatch(ctx, states, requestedVersion)
		if err != nil {
			// A snapshot is pinned for one episode, but the learner may evict
			// old snapshots to bound memory. That invalidates this episode only.
			if requestedVersion != 0 && isEvictedPolicySnapshot(err) {
				if resetErr := r.restartWorkersAfterSnapshotEviction(indexes, requestedVersion, descriptor); resetErr != nil {
					r.fail(resetErr)
				}
				return
			}
			r.fail(fmt.Errorf("predict actions for policy version %d: %w", requestedVersion, err))
			return
		}
		if len(prediction.Actions) != len(states) {
			r.fail(fmt.Errorf("predict actions returned %d actions for %d workers", len(prediction.Actions), len(states)))
			return
		}
		if requestedVersion != 0 && prediction.PolicyVersion != requestedVersion {
			r.fail(fmt.Errorf("learner returned policy version %d for pinned version %d", prediction.PolicyVersion, requestedVersion))
			return
		}
		for position, index := range indexes {
			predictions[index] = indexedPrediction{action: prediction.Actions[position], version: prediction.PolicyVersion}
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.status != RuntimeRunning {
		return
	}
	for index, worker := range r.workers {
		prediction, exists := predictions[index]
		if !exists {
			r.lastError = fmt.Sprintf("worker %d has no prediction", worker.id)
			r.status = RuntimeError
			return
		}
		if worker.episodePolicyVersion == 0 && prediction.version != 0 {
			worker.episodePolicyVersion = prediction.version
		}
		if prediction.version != 0 {
			r.metrics.policyVersion = prediction.version
		}
		action := prediction.action
		if len(action) != descriptor.ActionDimension {
			r.lastError = fmt.Sprintf("worker %d received action dimension %d, want %d", worker.id, len(action), descriptor.ActionDimension)
			r.status = RuntimeError
			return
		}
		for component, value := range action {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || value < descriptor.ActionMin || value > descriptor.ActionMax {
				r.lastError = fmt.Sprintf("worker %d received unsafe action[%d]=%v outside [%v, %v]", worker.id, component, value, descriptor.ActionMin, descriptor.ActionMax)
				r.status = RuntimeError
				return
			}
		}
		result, stepErr := worker.task.Step(action)
		if stepErr != nil {
			r.lastError = fmt.Sprintf("worker %d step: %v", worker.id, stepErr)
			r.status = RuntimeError
			return
		}
		if len(result.State) != descriptor.StateDimension {
			r.lastError = fmt.Sprintf("worker %d step returned state dimension %d, want %d", worker.id, len(result.State), descriptor.StateDimension)
			r.status = RuntimeError
			return
		}
		transition := Transition{Observation: worker.state, Action: action, Reward: result.Reward, NextObservation: result.State, Outcome: result.Outcome, Done: result.Done}
		r.replay.Add(transition)
		worker.actionSource = "policy"
		if useRandomWarmup {
			worker.actionSource = "random_warmup"
		}
		worker.lastAction, worker.lastReward, worker.lastInfo, worker.outcome = append(Action(nil), action...), result.Reward, cloneInfo(result.Info), result.Outcome
		r.recordAction(action, result.Info)
		if phase, ok := result.Info["phase_numeric"]; ok {
			r.metrics.phaseCounts[int(phase)]++
		}
		if result.Info["contact_detected"] > 0 {
			r.metrics.contactSamples++
		}
		if result.Info["object_attached"] > 0 {
			r.metrics.attachmentSamples++
		}
		worker.episodeStep++
		worker.episodeReward += float64(result.Reward)
		r.metrics.totalSteps++
		r.metrics.totalReward += float64(result.Reward)
		if result.Done {
			r.metrics.totalEpisodes++
			if result.Outcome == OutcomeSuccess {
				r.metrics.successes++
			} else if reason, ok := result.Info["failure_reason_code"]; ok && reason > 0 {
				r.metrics.failureReasons[fmt.Sprintf("reason_%d", int(reason))]++
			}
			state, resetErr := worker.task.Reset()
			if resetErr != nil {
				r.lastError = fmt.Sprintf("worker %d reset: %v", worker.id, resetErr)
				r.status = RuntimeError
				return
			}
			worker.state, worker.episodeID, worker.episodeStep, worker.episodeReward, worker.episodePolicyVersion, worker.lastInfo, worker.outcome, worker.actionSource = append(State(nil), state...), worker.episodeID+1, 0, 0, 0, nil, OutcomeRunning, "pending"
		} else {
			worker.state = append(State(nil), result.State...)
		}
	}
	r.updateRates(time.Now())
	shouldTrain := !r.trainingInFlight && r.replay.Len() >= r.config.WarmupTransitions && r.metrics.totalSteps%uint64(r.config.TrainingInterval) == 0
	if shouldTrain {
		sample, sampleErr := r.replay.Sample(r.config.TrainingBatchSize)
		if sampleErr == nil {
			r.trainingInFlight = true
			go r.train(ctx, sample)
		}
	}
}

// isEvictedPolicySnapshot identifies the one prediction failure that can be
// recovered by beginning a fresh episode. Other FailedPrecondition responses
// may indicate a configuration error, so they remain fatal.
func isEvictedPolicySnapshot(err error) bool {
	if status.Code(err) != codes.FailedPrecondition {
		return false
	}
	message := strings.ToLower(err.Error())
	return strings.Contains(message, "policy snapshot") && strings.Contains(message, "unavailable")
}

// restartWorkersAfterSnapshotEviction discards episodes whose pinned actor
// snapshot was evicted. It intentionally adds no terminal transition to the
// replay buffer: the environment did not produce a valid action for that
// state. The next cycle asks the learner for a current immutable snapshot.
func (r *Runtime) restartWorkersAfterSnapshotEviction(indexes []int, evictedVersion uint64, descriptor TaskDescriptor) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.status != RuntimeRunning {
		return nil
	}
	for _, index := range indexes {
		if index < 0 || index >= len(r.workers) {
			return fmt.Errorf("restart worker index %d after evicted policy snapshot: out of range", index)
		}
		worker := r.workers[index]
		// A concurrent control path may already have reset this worker. Do not
		// discard a newer episode in that case.
		if worker.episodePolicyVersion != evictedVersion {
			continue
		}
		state, err := worker.task.Reset()
		if err != nil {
			return fmt.Errorf("restart worker %d after evicted policy snapshot %d: %w", worker.id, evictedVersion, err)
		}
		if len(state) != descriptor.StateDimension {
			return fmt.Errorf("restart worker %d after evicted policy snapshot %d returned state dimension %d, want %d", worker.id, evictedVersion, len(state), descriptor.StateDimension)
		}
		worker.state = append(State(nil), state...)
		worker.episodeID++
		worker.episodeStep = 0
		worker.lastAction = nil
		worker.lastReward = 0
		worker.episodeReward = 0
		worker.episodePolicyVersion = 0
		worker.lastInfo = map[string]float32{"policy_snapshot_restarted": 1}
		worker.outcome = OutcomeRunning
		worker.actionSource = "snapshot_recovery"
	}
	return nil
}

func randomAction(seed int64, dimension int, minimum, maximum float32) Action {
	random := rand.New(rand.NewSource(seed))
	action := make(Action, dimension)
	for index := range action {
		action[index] = minimum + random.Float32()*(maximum-minimum)
	}
	return action
}

func (r *Runtime) recordAction(action Action, info map[string]float32) {
	if len(r.metrics.rawActionSum) != len(action) {
		r.metrics.rawActionSum = make([]float64, len(action))
		r.metrics.rawActionSquare = make([]float64, len(action))
		r.metrics.filteredActionSum = make([]float64, len(action))
		r.metrics.filteredActionSquare = make([]float64, len(action))
		r.metrics.deadZoneRemoved = make([]uint64, len(action))
		r.metrics.filterModified = make([]uint64, len(action))
	}
	for index, value := range action {
		raw := float64(value)
		filtered := raw
		if telemetryValue, ok := info[actionInfoKey("filtered_action", index)]; ok {
			filtered = float64(telemetryValue)
		}
		r.metrics.rawActionSum[index] += raw
		r.metrics.rawActionSquare[index] += raw * raw
		r.metrics.filteredActionSum[index] += filtered
		r.metrics.filteredActionSquare[index] += filtered * filtered
		if info[actionInfoKey("dead_zone_removed", index)] > 0 {
			r.metrics.deadZoneRemoved[index]++
		}
		if math.Abs(filtered-raw) > 1e-6 {
			r.metrics.filterModified[index]++
		}
	}
	r.metrics.actionCount++
}

func actionInfoKey(prefix string, index int) string {
	components := []string{"horizontal", "vertical", "gripper"}
	if index >= 0 && index < len(components) {
		return prefix + "_" + components[index]
	}
	return fmt.Sprintf("%s_%d", prefix, index)
}

func (r *Runtime) train(ctx context.Context, transitions []Transition) {
	result, err := r.learner.TrainBatch(ctx, transitions)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.trainingInFlight = false
	if err != nil {
		r.lastError = fmt.Sprintf("train batch: %v", err)
		return
	}
	r.metrics.trainingBatches++
	r.metrics.policyVersion, r.metrics.trainingStep = result.PolicyVersion, result.TrainingStep
	r.metrics.actorLoss, r.metrics.criticLoss, r.metrics.alphaLoss, r.metrics.entropy = result.ActorLoss, result.CriticLoss, result.AlphaLoss, result.Entropy
	r.metrics.criticOneQ, r.metrics.criticTwoQ, r.metrics.alpha = result.CriticOneQ, result.CriticTwoQ, result.Alpha
	r.metrics.actorLogStdHorizontal, r.metrics.actorLogStdVertical, r.metrics.actorLogStdGripper = result.ActorLogStdHorizontal, result.ActorLogStdVertical, result.ActorLogStdGripper
}

func (r *Runtime) Pause() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.status != RuntimeRunning {
		return errors.New("runtime is not running")
	}
	r.status = RuntimePaused
	return nil
}
func (r *Runtime) Resume() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.status != RuntimePaused {
		return errors.New("runtime is not paused")
	}
	r.status = RuntimeRunning
	r.metrics.lastProgressAt = time.Now()
	return nil
}
func (r *Runtime) Reset() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.status == RuntimeRunning {
		return errors.New("runtime must be paused before reset")
	}
	for _, worker := range r.workers {
		state, err := worker.task.Reset()
		if err != nil {
			return err
		}
		worker.state, worker.episodeID, worker.episodeStep, worker.episodeReward, worker.lastInfo, worker.outcome, worker.actionSource = append(State(nil), state...), worker.episodeID+1, 0, 0, nil, OutcomeRunning, "pending"
	}
	return nil
}

// SaveCheckpoint asks the configured learner to durably save its complete
// training state. It does not reset active episodes or the Go replay buffer.
func (r *Runtime) SaveCheckpoint(ctx context.Context, modelName string) (CheckpointResult, error) {
	if r == nil {
		return CheckpointResult{}, errors.New("runtime is nil")
	}
	if ctx == nil {
		return CheckpointResult{}, errors.New("context is required")
	}
	r.mu.Lock()
	learner := r.learner
	r.mu.Unlock()
	checkpointing, ok := learner.(CheckpointingLearner)
	if !ok {
		return CheckpointResult{}, errors.New("configured learner does not support model checkpoints")
	}
	return checkpointing.SaveCheckpoint(ctx, modelName)
}

// ApproveCurriculumReview advances one reviewable worker to its next
// curriculum reset distribution. It deliberately does not alter success
// counters or add a transition to replay: human review is evaluation control,
// not a fabricated reinforcement-learning reward.
func (r *Runtime) ApproveCurriculumReview(workerID int) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.status != RuntimePaused {
		return errors.New("runtime must be paused before approving a curriculum review")
	}
	if workerID <= 0 || workerID > len(r.workers) {
		return fmt.Errorf("worker %d does not exist", workerID)
	}
	worker := r.workers[workerID-1]
	reviewable, ok := worker.task.(ReviewableTask)
	if !ok {
		return errors.New("active task does not support curriculum review")
	}
	if err := reviewable.ApproveCurriculumReview(); err != nil {
		return err
	}
	state, err := worker.task.Reset()
	if err != nil {
		return fmt.Errorf("reset approved worker %d: %w", workerID, err)
	}
	worker.state = append(State(nil), state...)
	worker.episodeID++
	worker.episodeStep = 0
	worker.episodeReward = 0
	worker.episodePolicyVersion = 0
	worker.lastInfo = map[string]float32{"manual_curriculum_review": 1}
	worker.outcome = OutcomeRunning
	worker.actionSource = "pending"
	return nil
}
func (r *Runtime) Stop() {
	r.mu.Lock()
	cancel, done := r.cancel, r.done
	r.cancel, r.status = nil, RuntimeStopped
	r.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if done != nil {
		<-done
	}
}
func (r *Runtime) fail(err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.status == RuntimeRunning {
		r.status, r.lastError = RuntimeError, err.Error()
	}
}

func (r *Runtime) Snapshot() RuntimeSnapshot {
	r.mu.RLock()
	defer r.mu.RUnlock()
	status := r.status
	if status == RuntimeRunning && !r.metrics.lastProgressAt.IsZero() && time.Since(r.metrics.lastProgressAt) > 5*time.Second {
		status = RuntimeStalled
	}
	snapshot := RuntimeSnapshot{Status: status, TotalSteps: r.metrics.totalSteps, TotalEpisodes: r.metrics.totalEpisodes, TrainingBatches: r.metrics.trainingBatches, PolicyVersion: r.metrics.policyVersion, TrainingStep: r.metrics.trainingStep, ActorLoss: r.metrics.actorLoss, CriticLoss: r.metrics.criticLoss, AlphaLoss: r.metrics.alphaLoss, Entropy: r.metrics.entropy, CriticOneQ: r.metrics.criticOneQ, CriticTwoQ: r.metrics.criticTwoQ, Alpha: r.metrics.alpha, ActorLogStdHorizontal: r.metrics.actorLogStdHorizontal, ActorLogStdVertical: r.metrics.actorLogStdVertical, ActorLogStdGripper: r.metrics.actorLogStdGripper, StepsPerSecond: r.metrics.stepsPerSecond, EpisodesPerSecond: r.metrics.episodesPerSecond, LastError: r.lastError, ActiveWorkers: len(r.workers), ActionStatistics: r.actionStatistics(), PhaseOccupancy: phaseOccupancy(r.metrics.phaseCounts), FailureReasons: cloneCounts(r.metrics.failureReasons)}
	if r.replay != nil {
		snapshot.ReplayBufferSize = r.replay.Len()
	}
	if r.metrics.totalEpisodes > 0 {
		snapshot.SuccessRate = float64(r.metrics.successes) / float64(r.metrics.totalEpisodes)
	}
	if r.metrics.totalSteps > 0 {
		snapshot.AverageReward = r.metrics.totalReward / float64(r.metrics.totalSteps)
		snapshot.ContactRate = float64(r.metrics.contactSamples) / float64(r.metrics.totalSteps)
		snapshot.AttachmentRate = float64(r.metrics.attachmentSamples) / float64(r.metrics.totalSteps)
	}
	snapshot.Workers = make([]WorkerSnapshot, len(r.workers))
	for i, worker := range r.workers {
		var metadata map[string]any
		if telemetryTask, ok := worker.task.(TelemetryTask); ok {
			metadata = telemetryTask.TelemetryMetadata()
		}
		snapshot.Workers[i] = WorkerSnapshot{ID: worker.id, EpisodeID: worker.episodeID, EpisodeStep: worker.episodeStep, State: append(State(nil), worker.state...), LastAction: append(Action(nil), worker.lastAction...), LastReward: worker.lastReward, EpisodeReward: worker.episodeReward, PolicyVersion: worker.episodePolicyVersion, Info: cloneInfo(worker.lastInfo), Outcome: worker.outcome, ActionSource: worker.actionSource, Metadata: metadata}
	}
	return snapshot
}

func (r *Runtime) actionStatistics() ActionStatistics {
	statistics := ActionStatistics{Samples: r.metrics.actionCount}
	dimension := len(r.metrics.rawActionSum)
	statistics.RawMean = make([]float64, dimension)
	statistics.RawStd = make([]float64, dimension)
	statistics.FilteredMean = make([]float64, dimension)
	statistics.FilteredStd = make([]float64, dimension)
	statistics.DeadZoneRemovedFraction = make([]float64, dimension)
	statistics.FilterModifiedFraction = make([]float64, dimension)
	if r.metrics.actionCount == 0 {
		return statistics
	}
	count := float64(r.metrics.actionCount)
	for index := 0; index < dimension; index++ {
		statistics.RawMean[index] = r.metrics.rawActionSum[index] / count
		statistics.FilteredMean[index] = r.metrics.filteredActionSum[index] / count
		statistics.RawStd[index] = math.Sqrt(math.Max(0, r.metrics.rawActionSquare[index]/count-statistics.RawMean[index]*statistics.RawMean[index]))
		statistics.FilteredStd[index] = math.Sqrt(math.Max(0, r.metrics.filteredActionSquare[index]/count-statistics.FilteredMean[index]*statistics.FilteredMean[index]))
		statistics.DeadZoneRemovedFraction[index] = float64(r.metrics.deadZoneRemoved[index]) / count
		statistics.FilterModifiedFraction[index] = float64(r.metrics.filterModified[index]) / count
	}
	return statistics
}

func phaseOccupancy(counts map[int]uint64) map[string]uint64 {
	result := make(map[string]uint64, len(counts))
	for phase, count := range counts {
		result[fmt.Sprintf("phase_%d", phase)] = count
	}
	return result
}

func cloneCounts(counts map[string]uint64) map[string]uint64 {
	if len(counts) == 0 {
		return map[string]uint64{}
	}
	copy := make(map[string]uint64, len(counts))
	for key, value := range counts {
		copy[key] = value
	}
	return copy
}

func cloneInfo(info map[string]float32) map[string]float32 {
	if len(info) == 0 {
		return nil
	}
	copy := make(map[string]float32, len(info))
	for key, value := range info {
		copy[key] = value
	}
	return copy
}

func (r *Runtime) updateRates(now time.Time) {
	r.metrics.lastProgressAt = now
	if r.metrics.rateStartedAt.IsZero() {
		r.metrics.rateStartedAt, r.metrics.rateStartSteps, r.metrics.rateStartEpisodes = now, r.metrics.totalSteps, r.metrics.totalEpisodes
		return
	}
	elapsed := now.Sub(r.metrics.rateStartedAt)
	if elapsed < 250*time.Millisecond {
		return
	}
	r.metrics.stepsPerSecond = float64(r.metrics.totalSteps-r.metrics.rateStartSteps) / elapsed.Seconds()
	r.metrics.episodesPerSecond = float64(r.metrics.totalEpisodes-r.metrics.rateStartEpisodes) / elapsed.Seconds()
	r.metrics.rateStartedAt, r.metrics.rateStartSteps, r.metrics.rateStartEpisodes = now, r.metrics.totalSteps, r.metrics.totalEpisodes
}
