package framework

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestTerminalWorkersResetBeforeNextStepAndTrainingStopKeepsSimulationRunning(t *testing.T) {
	learner := &stoppingRuntimeLearner{}
	factory := &terminalRuntimeFactory{}
	runtime := NewRuntime()
	descriptor := TaskDescriptor{Name: "terminal", StateDimension: 2, ActionDimension: 1, ActionMin: -1, ActionMax: 1}
	if err := runtime.RegisterTask(TaskRegistration{Descriptor: descriptor, Factory: factory}); err != nil {
		t.Fatal(err)
	}
	config := DefaultRuntimeConfig()
	config.WorkerCount, config.TickInterval, config.RandomActionWarmupTransitions, config.WarmupTransitions, config.TrainingBatchSize, config.TrainingInterval = 2, time.Millisecond, 0, 1, 1, 1
	if err := runtime.Configure(config, learner); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := runtime.Start(ctx); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Second)
	for runtime.Snapshot().TotalEpisodes < 6 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	snapshot := runtime.Snapshot()
	runtime.Stop()
	if snapshot.Status != RuntimeRunning || snapshot.TotalEpisodes < 6 || learner.reports != 2 || !runtime.trainingStopped || learner.trainings != 0 {
		t.Fatalf("simulation/training state after stop: snapshot=%#v reports=%d trainingStopped=%v trainings=%d", snapshot, learner.reports, runtime.trainingStopped, learner.trainings)
	}
	for index, task := range factory.tasks {
		if task.resets <= 2 || task.steps < 2 || task.terminal {
			t.Fatalf("worker %d was not reset after terminal steps: %#v", index+1, task)
		}
	}
}

func TestResetRecoversErroredRuntimeAndResetsEveryWorker(t *testing.T) {
	factory := &terminalRuntimeFactory{}
	runtime := NewRuntime()
	if err := runtime.RegisterTask(TaskRegistration{Descriptor: TaskDescriptor{Name: "terminal", StateDimension: 2, ActionDimension: 1, ActionMin: -1, ActionMax: 1}, Factory: factory}); err != nil {
		t.Fatal(err)
	}
	config := DefaultRuntimeConfig()
	config.WorkerCount, config.TickInterval = 2, time.Hour
	if err := runtime.Configure(config, &runtimeTestLearner{}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := runtime.Start(ctx); err != nil {
		t.Fatal(err)
	}
	defer runtime.Stop()
	runtime.mu.Lock()
	runtime.status, runtime.lastError = RuntimeError, "worker 2 stepped terminal task"
	factory.tasks[1].terminal = true
	runtime.mu.Unlock()
	if err := runtime.Reset(); err != nil {
		t.Fatal(err)
	}
	if snapshot := runtime.Snapshot(); snapshot.Status != RuntimeRunning || snapshot.LastError != "" {
		t.Fatalf("reset did not recover runtime: %#v", snapshot)
	}
	if err := runtime.Reset(); err != nil {
		t.Fatalf("reset while running failed: %v", err)
	}
	for index, task := range factory.tasks {
		if task.terminal || task.resets != 3 {
			t.Fatalf("worker %d did not reset twice after start: %#v", index+1, task)
		}
	}
}

func TestCycleResetsAlreadyTerminalTaskBeforeStepping(t *testing.T) {
	factory := &terminalRuntimeFactory{}
	runtime := NewRuntime()
	descriptor := TaskDescriptor{Name: "terminal", StateDimension: 2, ActionDimension: 1, ActionMin: -1, ActionMax: 1}
	if err := runtime.RegisterTask(TaskRegistration{Descriptor: descriptor, Factory: factory}); err != nil {
		t.Fatal(err)
	}
	config := DefaultRuntimeConfig()
	config.WorkerCount, config.TickInterval, config.RandomActionWarmupTransitions, config.WarmupTransitions = 2, time.Hour, 0, 100000
	if err := runtime.Configure(config, &runtimeTestLearner{}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := runtime.Start(ctx); err != nil {
		t.Fatal(err)
	}
	defer runtime.Stop()
	runtime.mu.Lock()
	factory.tasks[1].terminal = true
	runtime.mu.Unlock()
	runtime.cycle(ctx, descriptor)
	if snapshot := runtime.Snapshot(); snapshot.Status != RuntimeRunning || snapshot.TotalEpisodes != 2 {
		t.Fatalf("terminal task was stepped before reset: %#v", snapshot)
	}
	if factory.tasks[1].resets != 3 || factory.tasks[1].steps != 1 {
		t.Fatalf("terminal task reset/step order is wrong: %#v", factory.tasks[1])
	}
}

func TestResetDiscardsPredictionForOldEpisode(t *testing.T) {
	learner := &blockingPredictionLearner{started: make(chan struct{}), release: make(chan struct{})}
	runtime := NewRuntime()
	descriptor := TaskDescriptor{Name: "test", StateDimension: 2, ActionDimension: 1, ActionMin: -1, ActionMax: 1}
	if err := runtime.RegisterTask(TaskRegistration{Descriptor: descriptor, Factory: runtimeTestFactory{}}); err != nil {
		t.Fatal(err)
	}
	config := DefaultRuntimeConfig()
	config.WorkerCount, config.TickInterval, config.RandomActionWarmupTransitions = 1, time.Hour, 0
	if err := runtime.Configure(config, learner); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := runtime.Start(ctx); err != nil {
		t.Fatal(err)
	}
	defer runtime.Stop()
	done := make(chan struct{})
	go func() { runtime.cycle(ctx, descriptor); close(done) }()
	<-learner.started
	if err := runtime.Reset(); err != nil {
		t.Fatal(err)
	}
	close(learner.release)
	<-done
	if snapshot := runtime.Snapshot(); snapshot.TotalSteps != 0 || snapshot.Workers[0].EpisodeStep != 0 {
		t.Fatalf("old prediction stepped reset episode: %#v", snapshot)
	}
}

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

type terminalRuntimeTask struct {
	terminal   bool
	returnDone bool
	resets     int
	steps      int
}

func (task *terminalRuntimeTask) Reset() (State, error) {
	task.terminal = false
	task.resets++
	return State{0, 0}, nil
}

func (task *terminalRuntimeTask) Step(Action) (StepResult, error) {
	if task.terminal {
		return StepResult{}, errors.New("stepped terminal task")
	}
	task.steps++
	task.terminal = true
	return StepResult{State: State{0, 0}, Reward: 1, Outcome: OutcomeSuccess, Done: task.returnDone}, nil
}

func (task *terminalRuntimeTask) IsTerminal() bool { return task.terminal }

type terminalRuntimeFactory struct{ tasks []*terminalRuntimeTask }

func (factory *terminalRuntimeFactory) Create() (Task, error) {
	task := &terminalRuntimeTask{returnDone: len(factory.tasks)%2 == 0}
	factory.tasks = append(factory.tasks, task)
	return task, nil
}

type stoppingRuntimeLearner struct {
	runtimeTestLearner
	reports int
}

func (learner *stoppingRuntimeLearner) RecordEpisodeResult(_ context.Context, _ EpisodeResult) (EpisodeStopResult, error) {
	learner.reports++
	return EpisodeStopResult{StopTraining: learner.reports >= 2, CompletedEpisodes: uint64(learner.reports)}, nil
}

type blockingPredictionLearner struct {
	runtimeTestLearner
	started chan struct{}
	release chan struct{}
}

func (learner *blockingPredictionLearner) PredictBatch(ctx context.Context, states []State, version uint64) (PredictionResult, error) {
	close(learner.started)
	<-learner.release
	return learner.runtimeTestLearner.PredictBatch(ctx, states, version)
}

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

func (l *checkpointRuntimeLearner) SaveCheckpoint(_ context.Context) (CheckpointResult, error) {
	l.savedName = "grasp-v1"
	return CheckpointResult{ModelName: l.savedName, PolicyVersion: 4, TrainingStep: 9}, nil
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

	result, err := runtime.SaveCheckpoint(context.Background())
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

func TestRuntimeRollingSuccessRateUsesLast1000Episodes(t *testing.T) {
	runtime := NewRuntime()
	for i := 0; i < 1000; i++ {
		runtime.recordEpisodeOutcome(i < 700)
	}
	if got, want := runtime.rollingSuccessRate(), 0.7; got != want {
		t.Fatalf("rolling success rate = %.3f, want %.3f", got, want)
	}
	for i := 0; i < 100; i++ {
		runtime.recordEpisodeOutcome(false)
	}
	if got, want := runtime.rollingSuccessRate(), 0.6; got != want {
		t.Fatalf("rolling success rate after 1100 episodes = %.3f, want %.3f", got, want)
	}
}
