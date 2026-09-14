package framework

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"regexp"
	"sync"
)

var taskNamePattern = regexp.MustCompile(`^[a-z][a-z0-9_-]*$`)

// Runtime stores public task registrations.
type Runtime struct {
	mu             sync.RWMutex
	tasks          map[string]TaskRegistration
	config         RuntimeConfig
	learner        Learner
	activeTaskName string
	workers        []*runtimeWorker
	replay         *ReplayBuffer
	status         RuntimeStatus
	cancel         context.CancelFunc
	done           chan struct{}
	metrics        runtimeMetrics
	lastError      string
}

// NewRuntime creates an empty task runtime.
func NewRuntime() *Runtime {
	return &Runtime{tasks: make(map[string]TaskRegistration), config: DefaultRuntimeConfig(), status: RuntimeStopped}
}

// RegisterTask validates and stores a task registration.
func (r *Runtime) RegisterTask(registration TaskRegistration) error {
	if r == nil {
		return errors.New("runtime is nil")
	}
	if err := validateTaskRegistration(registration); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.tasks == nil {
		r.tasks = make(map[string]TaskRegistration)
	}
	if _, exists := r.tasks[registration.Descriptor.Name]; exists {
		return fmt.Errorf("task %q is already registered", registration.Descriptor.Name)
	}

	r.tasks[registration.Descriptor.Name] = registration
	if r.activeTaskName == "" {
		r.activeTaskName = registration.Descriptor.Name
	}
	return nil
}

// ReplaceTask updates a registered task while the runtime is not stepping.
func (r *Runtime) ReplaceTask(registration TaskRegistration) error {
	if r == nil {
		return errors.New("runtime is nil")
	}
	if err := validateTaskRegistration(registration); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.status == RuntimeRunning {
		return errors.New("runtime must be paused before replacing a task")
	}
	if r.tasks == nil {
		r.tasks = make(map[string]TaskRegistration)
	}
	r.tasks[registration.Descriptor.Name] = registration
	r.activeTaskName = registration.Descriptor.Name
	for _, worker := range r.workers {
		task, err := registration.Factory.Create()
		if err != nil {
			return fmt.Errorf("replace worker %d task: %w", worker.id, err)
		}
		state, err := task.Reset()
		if err != nil {
			return fmt.Errorf("reset replacement worker %d task: %w", worker.id, err)
		}
		if len(state) != registration.Descriptor.StateDimension {
			return fmt.Errorf("replacement worker %d returned state dimension %d, want %d", worker.id, len(state), registration.Descriptor.StateDimension)
		}
		worker.task, worker.state, worker.episodeID, worker.episodeStep, worker.outcome = task, append(State(nil), state...), worker.episodeID+1, 0, OutcomeRunning
	}
	return nil
}

func validateTaskRegistration(registration TaskRegistration) error {
	descriptor := registration.Descriptor
	if !taskNamePattern.MatchString(descriptor.Name) {
		return errors.New("task name must start with a lowercase letter and contain only lowercase letters, digits, hyphens, or underscores")
	}
	if descriptor.StateDimension <= 0 {
		return errors.New("state dimension must be greater than zero")
	}
	if descriptor.ActionDimension <= 0 {
		return errors.New("action dimension must be greater than zero")
	}
	if math.IsNaN(float64(descriptor.ActionMin)) || math.IsNaN(float64(descriptor.ActionMax)) || math.IsInf(float64(descriptor.ActionMin), 0) || math.IsInf(float64(descriptor.ActionMax), 0) || descriptor.ActionMin >= descriptor.ActionMax {
		return errors.New("action minimum must be less than action maximum")
	}
	if isNilTaskFactory(registration.Factory) {
		return errors.New("task factory is required")
	}
	return nil
}

func isNilTaskFactory(factory TaskFactory) bool {
	if factory == nil {
		return true
	}
	value := reflect.ValueOf(factory)
	return value.Kind() == reflect.Ptr && value.IsNil()
}
