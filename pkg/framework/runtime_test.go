package framework

import (
	"context"
	"sync"
	"testing"
	"time"
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
	action                 Action
}

func (l *runtimeTestLearner) HealthCheck(context.Context) (HealthStatus, error) {
	return HealthStatus{Ready: true}, nil
}
func (l *runtimeTestLearner) PredictBatch(_ context.Context, states []State) (PredictionResult, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.predictions++
	actions := make([]Action, len(states))
	for i := range actions {
		if l.action != nil {
			actions[i] = append(Action(nil), l.action...)
		} else {
			actions[i] = Action{0.1}
		}
	}
	return PredictionResult{Actions: actions, PolicyVersion: 1}, nil
}
func (l *runtimeTestLearner) TrainBatch(_ context.Context, transitions []Transition) (TrainingResult, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.trainings++
	return TrainingResult{Accepted: true, PolicyVersion: 2, TrainingStep: uint64(l.trainings)}, nil
}
func (l *runtimeTestLearner) Close() error { return nil }

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

func TestRuntimeBatchesWorkersAndSupportsPauseResume(t *testing.T) {
	learner := &runtimeTestLearner{}
	runtime := NewRuntime()
	if err := runtime.RegisterTask(TaskRegistration{Descriptor: TaskDescriptor{Name: "test", StateDimension: 2, ActionDimension: 1, ActionMin: -1, ActionMax: 1}, Factory: runtimeTestFactory{}}); err != nil {
		t.Fatal(err)
	}
	config := DefaultRuntimeConfig()
	config.WorkerCount, config.TickInterval, config.WarmupTransitions, config.TrainingBatchSize, config.TrainingInterval = 2, time.Millisecond, 2, 2, 1
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
}

func TestRuntimeRejectsUnsafePolicyActionsBeforeTaskStep(t *testing.T) {
	learner := &runtimeTestLearner{action: Action{2}}
	runtime := NewRuntime()
	if err := runtime.RegisterTask(TaskRegistration{Descriptor: TaskDescriptor{Name: "safe", StateDimension: 2, ActionDimension: 1, ActionMin: -1, ActionMax: 1}, Factory: runtimeTestFactory{}}); err != nil {
		t.Fatal(err)
	}
	config := DefaultRuntimeConfig()
	config.WorkerCount, config.TickInterval = 1, time.Millisecond
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
