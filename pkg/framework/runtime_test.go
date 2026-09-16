package framework

import (
	"context"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type runtimeTestTask struct{ state State }

func (t *runtimeTestTask) Reset() (State, error) {
	t.state = State{0, 0}
	return append(State(nil), t.state...), nil
}
func (t *runtimeTestTask) Step(action Action) (StepResult, error) {
	t.state[0] += action[0]
	return StepResult{State: append(State(nil), t.state...), Reward: 1, Outcome: OutcomeRunning}, nil
}

type runtimeTestFactory struct{}

func (runtimeTestFactory) Create() (Task, error) { return &runtimeTestTask{}, nil }

type runtimeTestLearner struct {
	mu                     sync.Mutex
	predictions, trainings int
	requestedVersions      []uint64
	action                 Action
}

func (l *runtimeTestLearner) HealthCheck(context.Context) (HealthStatus, error) {
	return HealthStatus{Ready: true}, nil
}
func (l *runtimeTestLearner) PredictBatch(_ context.Context, states []State, requestedVersion uint64) (PredictionResult, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.predictions++
	l.requestedVersions = append(l.requestedVersions, requestedVersion)
	actions := make([]Action, len(states))
	for i := range actions {
		if l.action != nil {
			actions[i] = append(Action(nil), l.action...)
		} else {
			actions[i] = Action{0.1}
		}
	}
	if requestedVersion == 0 {
		requestedVersion = 1
	}
	return PredictionResult{Actions: actions, PolicyVersion: requestedVersion}, nil
}
func (l *runtimeTestLearner) TrainBatch(_ context.Context, transitions []Transition) (TrainingResult, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.trainings++
	return TrainingResult{Accepted: true, PolicyVersion: 2, TrainingStep: uint64(l.trainings)}, nil
}
func (l *runtimeTestLearner) Close() error { return nil }

type snapshotEvictionLearner struct {
	mu                sync.Mutex
	requestedVersions []uint64
	evicted           bool
}

func (l *snapshotEvictionLearner) HealthCheck(context.Context) (HealthStatus, error) {
	return HealthStatus{Ready: true}, nil
}

func (l *snapshotEvictionLearner) PredictBatch(_ context.Context, states []State, requestedVersion uint64) (PredictionResult, error) {
	l.mu.Lock()
	l.requestedVersions = append(l.requestedVersions, requestedVersion)
	if requestedVersion == 1 && !l.evicted {
		l.evicted = true
		l.mu.Unlock()
		return PredictionResult{}, status.Error(codes.FailedPrecondition, "policy snapshot 1 is unavailable; start a fresh episode")
	}
	version := requestedVersion
	if version == 0 {
		if l.evicted {
			version = 2
		} else {
			version = 1
		}
	}
	l.mu.Unlock()
	actions := make([]Action, len(states))
	for index := range actions {
		actions[index] = Action{0.1}
	}
	return PredictionResult{Actions: actions, PolicyVersion: version}, nil
}

func (l *snapshotEvictionLearner) TrainBatch(context.Context, []Transition) (TrainingResult, error) {
	return TrainingResult{Accepted: true}, nil
}

func (l *snapshotEvictionLearner) Close() error { return nil }

type blockingTrainingLearner struct {
	runtimeTestLearner
	started chan struct{}
	release chan struct{}
	once    sync.Once
}

type checkpointRuntimeLearner struct {
	runtimeTestLearner
	savedName string
}

func (l *checkpointRuntimeLearner) SaveCheckpoint(_ context.Context, modelName string) (CheckpointResult, error) {
	l.savedName = modelName
	return CheckpointResult{ModelName: modelName, PolicyVersion: 4, TrainingStep: 9}, nil
}

func (l *blockingTrainingLearner) TrainBatch(_ context.Context, _ []Transition) (TrainingResult, error) {
	l.mu.Lock()
	l.trainings++
	trainingStep := l.trainings
	l.mu.Unlock()
	l.once.Do(func() { close(l.started) })
	<-l.release
	return TrainingResult{Accepted: true, PolicyVersion: 2, TrainingStep: uint64(trainingStep)}, nil
}

func TestReplayBufferOverwritesAndSamplesCopies(t *testing.T) {
	buffer, err := NewReplayBuffer(2, 1)
	if err != nil {
		t.Fatal(err)
	}
	buffer.Add(Transition{Observation: State{1}})
	buffer.Add(Transition{Observation: State{2}})
	buffer.Add(Transition{Observation: State{3}})
	if buffer.Len() != 2 {
		t.Fatalf("buffer length = %d, want 2", buffer.Len())
	}
	sample, err := buffer.Sample(2)
	if err != nil {
		t.Fatal(err)
	}
	sample[0].Observation[0] = 99
	again, err := buffer.Sample(2)
	if err != nil {
		t.Fatal(err)
	}
	for _, transition := range again {
		if transition.Observation[0] == 99 {
			t.Fatal("sample mutation changed buffered transition")
		}
	}
}

func TestRuntimeDelegatesCheckpointWithoutResettingWorkers(t *testing.T) {
	learner := &checkpointRuntimeLearner{}
	runtime := NewRuntime()
	runtime.learner = learner

	result, err := runtime.SaveCheckpoint(context.Background(), "grasp-v1")
	if err != nil {
		t.Fatalf("SaveCheckpoint() error = %v", err)
	}
	if learner.savedName != "grasp-v1" || result.PolicyVersion != 4 || result.TrainingStep != 9 {
		t.Fatalf("SaveCheckpoint() result=%#v learner=%#v", result, learner)
	}
}

func TestRuntimeBatchesWorkersAndSupportsPauseResume(t *testing.T) {
	learner := &runtimeTestLearner{}
	runtime := NewRuntime()
	if err := runtime.RegisterTask(TaskRegistration{Descriptor: TaskDescriptor{Name: "test", StateDimension: 2, ActionDimension: 1, ActionMin: -1, ActionMax: 1}, Factory: runtimeTestFactory{}}); err != nil {
		t.Fatal(err)
	}
	config := DefaultRuntimeConfig()
	config.WorkerCount, config.TickInterval, config.RandomActionWarmupTransitions, config.WarmupTransitions, config.TrainingBatchSize, config.TrainingInterval = 2, time.Millisecond, 0, 2, 2, 1
	if err := runtime.Configure(config, learner); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := runtime.Start(ctx); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Second)
	for runtime.Snapshot().TotalSteps < 4 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if runtime.Snapshot().TotalSteps < 4 {
		t.Fatal("runtime did not step workers")
	}
	if err := runtime.Pause(); err != nil {
		t.Fatal(err)
	}
	paused := runtime.Snapshot().TotalSteps
	time.Sleep(5 * time.Millisecond)
	if runtime.Snapshot().TotalSteps != paused {
		t.Fatal("runtime stepped while paused")
	}
	if err := runtime.Resume(); err != nil {
		t.Fatal(err)
	}
	runtime.Stop()
	if learner.predictions == 0 {
		t.Fatal("runtime did not issue batched predictions")
	}
	if got := runtime.Snapshot().Workers[0].PolicyVersion; got != 1 {
		t.Fatalf("worker episode policy version = %d, want 1", got)
	}
	learner.mu.Lock()
	defer learner.mu.Unlock()
	if len(learner.requestedVersions) < 2 || learner.requestedVersions[0] != 0 {
		t.Fatalf("policy snapshot requests = %v, want initial latest then pinned version", learner.requestedVersions)
	}
	for _, version := range learner.requestedVersions[1:] {
		if version != 1 {
			t.Fatalf("episode changed policy snapshot: requests=%v", learner.requestedVersions)
		}
	}
}

func TestRuntimeRestartsOnlyEvictedPolicySnapshotEpisode(t *testing.T) {
	learner := &snapshotEvictionLearner{}
	runtime := NewRuntime()
	if err := runtime.RegisterTask(TaskRegistration{Descriptor: TaskDescriptor{Name: "snapshot-recovery", StateDimension: 2, ActionDimension: 1, ActionMin: -1, ActionMax: 1}, Factory: runtimeTestFactory{}}); err != nil {
		t.Fatal(err)
	}
	config := DefaultRuntimeConfig()
	config.WorkerCount, config.TickInterval, config.RandomActionWarmupTransitions, config.WarmupTransitions = 1, time.Millisecond, 0, 1000
	if err := runtime.Configure(config, learner); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := runtime.Start(ctx); err != nil {
		t.Fatal(err)
	}
	defer runtime.Stop()

	deadline := time.Now().Add(time.Second)
	for {
		snapshot := runtime.Snapshot()
		if snapshot.Status == RuntimeError {
			t.Fatalf("snapshot eviction stopped the runtime: %#v", snapshot)
		}
		if snapshot.Workers[0].EpisodeID >= 2 && snapshot.Workers[0].PolicyVersion == 2 && snapshot.TotalSteps >= 2 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("runtime did not recover from an evicted snapshot: %#v", snapshot)
		}
		time.Sleep(time.Millisecond)
	}
	learner.mu.Lock()
	defer learner.mu.Unlock()
	if len(learner.requestedVersions) < 3 || learner.requestedVersions[0] != 0 || learner.requestedVersions[1] != 1 || learner.requestedVersions[2] != 0 {
		t.Fatalf("policy requests = %v, want latest, evicted snapshot, then latest", learner.requestedVersions)
	}
}

func TestRuntimeRejectsUnsafePolicyActionsBeforeTaskStep(t *testing.T) {
	learner := &runtimeTestLearner{action: Action{2}}
	runtime := NewRuntime()
	if err := runtime.RegisterTask(TaskRegistration{Descriptor: TaskDescriptor{Name: "safe", StateDimension: 2, ActionDimension: 1, ActionMin: -1, ActionMax: 1}, Factory: runtimeTestFactory{}}); err != nil {
		t.Fatal(err)
	}
	config := DefaultRuntimeConfig()
	config.WorkerCount, config.TickInterval, config.RandomActionWarmupTransitions = 1, time.Millisecond, 0
	if err := runtime.Configure(config, learner); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := runtime.Start(ctx); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Second)
	for runtime.Snapshot().Status != RuntimeError && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if runtime.Snapshot().Status != RuntimeError {
		t.Fatalf("runtime did not reject unsafe action: %#v", runtime.Snapshot())
	}
}

func TestRuntimeUsesNonZeroRandomActionsBeforePolicyWarmup(t *testing.T) {
	learner := &runtimeTestLearner{}
	runtime := NewRuntime()
	if err := runtime.RegisterTask(TaskRegistration{Descriptor: TaskDescriptor{Name: "warmup", StateDimension: 2, ActionDimension: 3, ActionMin: -1, ActionMax: 1}, Factory: runtimeTestFactory{}}); err != nil {
		t.Fatal(err)
	}
	config := DefaultRuntimeConfig()
	config.WorkerCount, config.TickInterval, config.RandomActionWarmupTransitions, config.WarmupTransitions = 1, time.Millisecond, 8, 100
	if err := runtime.Configure(config, learner); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := runtime.Start(ctx); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Second)
	for runtime.Snapshot().TotalSteps < 4 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	runtime.Stop()
	snapshot := runtime.Snapshot()
	if snapshot.TotalSteps < 4 {
		t.Fatalf("random warm-up did not step: %#v", snapshot)
	}
	if learner.predictions != 0 {
		t.Fatalf("random warm-up called actor %d times", learner.predictions)
	}
	if snapshot.Workers[0].ActionSource != "random_warmup" {
		t.Fatalf("action source = %q, want random_warmup", snapshot.Workers[0].ActionSource)
	}
	stats := snapshot.ActionStatistics
	if stats.Samples == 0 || len(stats.RawStd) != 3 || stats.RawStd[0] == 0 || stats.RawStd[1] == 0 || stats.RawStd[2] == 0 {
		t.Fatalf("warm-up actions were not diverse: %#v", stats)
	}
}

func TestRuntimeSchedulesAtMostOneTrainingBatchAtATime(t *testing.T) {
	learner := &blockingTrainingLearner{started: make(chan struct{}), release: make(chan struct{})}
	runtime := NewRuntime()
	if err := runtime.RegisterTask(TaskRegistration{Descriptor: TaskDescriptor{Name: "backpressure", StateDimension: 2, ActionDimension: 1, ActionMin: -1, ActionMax: 1}, Factory: runtimeTestFactory{}}); err != nil {
		t.Fatal(err)
	}
	config := DefaultRuntimeConfig()
	config.WorkerCount, config.TickInterval, config.RandomActionWarmupTransitions, config.WarmupTransitions, config.TrainingBatchSize, config.TrainingInterval = 1, time.Millisecond, 0, 1, 1, 1
	if err := runtime.Configure(config, learner); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := runtime.Start(ctx); err != nil {
		t.Fatal(err)
	}
	select {
	case <-learner.started:
	case <-time.After(time.Second):
		runtime.Stop()
		t.Fatal("runtime did not start a training batch")
	}
	// Several physics ticks pass while the learner update is blocked. A second
	// update must not be queued behind the first one.
	time.Sleep(20 * time.Millisecond)
	learner.mu.Lock()
	trainings := learner.trainings
	learner.mu.Unlock()
	if trainings != 1 {
		close(learner.release)
		runtime.Stop()
		t.Fatalf("concurrent or queued training calls = %d, want 1", trainings)
	}
	close(learner.release)
	runtime.Stop()
}
