package framework

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"
)

type RuntimeStatus string

const (
	RuntimeStopped RuntimeStatus = "stopped"
	RuntimeRunning RuntimeStatus = "running"
	RuntimePaused  RuntimeStatus = "paused"
	RuntimeError   RuntimeStatus = "error"
)

type runtimeWorker struct {
	id          int
	task        Task
	episodeID   uint64
	episodeStep uint64
	state       State
	lastAction  Action
	lastReward  float32
	outcome     Outcome
}

type runtimeMetrics struct {
	totalSteps      uint64
	totalEpisodes   uint64
	successes       uint64
	totalReward     float64
	trainingBatches uint64
	policyVersion   uint64
	trainingStep    uint64
	actorLoss       float32
	criticLoss      float32
	alphaLoss       float32
	entropy         float32
	startedAt       time.Time
}

type WorkerSnapshot struct {
	ID          int     `json:"id"`
	EpisodeID   uint64  `json:"episode_id"`
	EpisodeStep uint64  `json:"episode_step"`
	State       State   `json:"state"`
	LastAction  Action  `json:"last_action"`
	LastReward  float32 `json:"last_reward"`
	Outcome     Outcome `json:"outcome"`
}

type RuntimeSnapshot struct {
	Status           RuntimeStatus    `json:"status"`
	ActiveWorkers    int              `json:"active_workers"`
	TotalSteps       uint64           `json:"total_steps"`
	TotalEpisodes    uint64           `json:"total_episodes"`
	SuccessRate      float64          `json:"success_rate"`
	AverageReward    float64          `json:"average_reward"`
	ReplayBufferSize int              `json:"replay_buffer_size"`
	TrainingBatches  uint64           `json:"training_batches"`
	PolicyVersion    uint64           `json:"policy_version"`
	TrainingStep     uint64           `json:"training_step"`
	ActorLoss        float32          `json:"actor_loss"`
	CriticLoss       float32          `json:"critic_loss"`
	AlphaLoss        float32          `json:"alpha_loss"`
	Entropy          float32          `json:"entropy"`
	LastError        string           `json:"last_error"`
	Workers          []WorkerSnapshot `json:"workers"`
}

// Configure attaches the learner and runtime settings before Start.
func (r *Runtime) Configure(config RuntimeConfig, learner Learner) error {
	if r == nil {
		return errors.New("runtime is nil")
	}
	if learner == nil {
		return errors.New("learner is required")
	}
	if config.WorkerCount <= 0 || config.TickInterval <= 0 || config.ReplayCapacity <= 0 || config.WarmupTransitions < 0 || config.TrainingBatchSize <= 0 || config.TrainingInterval <= 0 {
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
		workers[index] = &runtimeWorker{id: index + 1, task: task, episodeID: 1, state: append(State(nil), state...), outcome: OutcomeRunning}
	}
	loopContext, cancel := context.WithCancel(ctx)
	r.workers, r.cancel, r.done, r.status, r.lastError = workers, cancel, make(chan struct{}), RuntimeRunning, ""
	r.metrics.startedAt = time.Now()
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
	states := make([]State, len(r.workers))
	for i, worker := range r.workers {
		states[i] = append(State(nil), worker.state...)
	}
	learner := r.learner
	r.mu.RUnlock()
	prediction, err := learner.PredictBatch(ctx, states)
	if err != nil {
		r.fail(fmt.Errorf("predict actions: %w", err))
		return
	}
	if len(prediction.Actions) != len(states) {
		r.fail(fmt.Errorf("predict actions returned %d actions for %d workers", len(prediction.Actions), len(states)))
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.status != RuntimeRunning {
		return
	}
	r.metrics.policyVersion = prediction.PolicyVersion
	for index, worker := range r.workers {
		action := prediction.Actions[index]
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
		worker.lastAction, worker.lastReward, worker.outcome = append(Action(nil), action...), result.Reward, result.Outcome
		worker.episodeStep++
		r.metrics.totalSteps++
		r.metrics.totalReward += float64(result.Reward)
		if result.Done {
			r.metrics.totalEpisodes++
			if result.Outcome == OutcomeSuccess {
				r.metrics.successes++
			}
			state, resetErr := worker.task.Reset()
			if resetErr != nil {
				r.lastError = fmt.Sprintf("worker %d reset: %v", worker.id, resetErr)
				r.status = RuntimeError
				return
			}
			worker.state, worker.episodeID, worker.episodeStep, worker.outcome = append(State(nil), state...), worker.episodeID+1, 0, OutcomeRunning
		} else {
			worker.state = append(State(nil), result.State...)
		}
	}
	shouldTrain := r.replay.Len() >= r.config.WarmupTransitions && r.metrics.totalSteps%uint64(r.config.TrainingInterval) == 0
	if shouldTrain {
		sample, sampleErr := r.replay.Sample(r.config.TrainingBatchSize)
		if sampleErr == nil {
			go r.train(ctx, sample)
		}
	}
}

func (r *Runtime) train(ctx context.Context, transitions []Transition) {
	result, err := r.learner.TrainBatch(ctx, transitions)
	r.mu.Lock()
	defer r.mu.Unlock()
	if err != nil {
		r.lastError = fmt.Sprintf("train batch: %v", err)
		return
	}
	r.metrics.trainingBatches++
	r.metrics.policyVersion, r.metrics.trainingStep = result.PolicyVersion, result.TrainingStep
	r.metrics.actorLoss, r.metrics.criticLoss, r.metrics.alphaLoss, r.metrics.entropy = result.ActorLoss, result.CriticLoss, result.AlphaLoss, result.Entropy
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
		worker.state, worker.episodeID, worker.episodeStep, worker.outcome = append(State(nil), state...), worker.episodeID+1, 0, OutcomeRunning
	}
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
	snapshot := RuntimeSnapshot{Status: r.status, TotalSteps: r.metrics.totalSteps, TotalEpisodes: r.metrics.totalEpisodes, TrainingBatches: r.metrics.trainingBatches, PolicyVersion: r.metrics.policyVersion, TrainingStep: r.metrics.trainingStep, ActorLoss: r.metrics.actorLoss, CriticLoss: r.metrics.criticLoss, AlphaLoss: r.metrics.alphaLoss, Entropy: r.metrics.entropy, LastError: r.lastError, ActiveWorkers: len(r.workers)}
	if r.replay != nil {
		snapshot.ReplayBufferSize = r.replay.Len()
	}
	if r.metrics.totalEpisodes > 0 {
		snapshot.SuccessRate = float64(r.metrics.successes) / float64(r.metrics.totalEpisodes)
	}
	if r.metrics.totalSteps > 0 {
		snapshot.AverageReward = r.metrics.totalReward / float64(r.metrics.totalSteps)
	}
	snapshot.Workers = make([]WorkerSnapshot, len(r.workers))
	for i, worker := range r.workers {
		snapshot.Workers[i] = WorkerSnapshot{ID: worker.id, EpisodeID: worker.episodeID, EpisodeStep: worker.episodeStep, State: append(State(nil), worker.state...), LastAction: append(Action(nil), worker.lastAction...), LastReward: worker.lastReward, Outcome: worker.outcome}
	}
	return snapshot
}
