package framework

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"regexp"
)

var taskNamePattern = regexp.MustCompile(`^[a-z][a-z0-9_-]*$`)

// Runtime stores public task registrations.
type Runtime struct {
	tasks map[string]TaskRegistration
}

// NewRuntime creates an empty task runtime.
func NewRuntime() *Runtime {
	return &Runtime{tasks: make(map[string]TaskRegistration)}
}

// RegisterTask validates and stores a task registration.
func (r *Runtime) RegisterTask(registration TaskRegistration) error {
	if r == nil {
		return errors.New("runtime is nil")
	}
	if err := validateTaskRegistration(registration); err != nil {
		return err
	}
	if r.tasks == nil {
		r.tasks = make(map[string]TaskRegistration)
	}
	if _, exists := r.tasks[registration.Descriptor.Name]; exists {
		return fmt.Errorf("task %q is already registered", registration.Descriptor.Name)
	}

	r.tasks[registration.Descriptor.Name] = registration
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
