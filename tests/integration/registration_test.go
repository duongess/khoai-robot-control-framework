package integration

import (
	"testing"

	"github.com/duongess/khoai-robot-control-framework/pkg/framework"
)

type externalTask struct{}

func (externalTask) Reset() (framework.State, error) {
	return framework.State{0, 0}, nil
}

func (externalTask) Step(framework.Action) (framework.StepResult, error) {
	return framework.StepResult{}, nil
}

type externalTaskFactory struct{}

func (externalTaskFactory) Create() (framework.Task, error) {
	return externalTask{}, nil
}

func TestRegisterTaskAcceptsValidRegistration(t *testing.T) {
	runtime := framework.NewRuntime()

	err := runtime.RegisterTask(validRegistration("force-control"))

	if err != nil {
		t.Fatalf("RegisterTask() error = %v", err)
	}
}

func TestRegisterTaskRejectsInvalidDimensions(t *testing.T) {
	runtime := framework.NewRuntime()
	registration := validRegistration("invalid-dimensions")
	registration.Descriptor.StateDimension = 0

	if err := runtime.RegisterTask(registration); err == nil {
		t.Fatal("RegisterTask() error = nil")
	}
}

func TestRegisterTaskRejectsInvalidActionBounds(t *testing.T) {
	runtime := framework.NewRuntime()
	registration := validRegistration("invalid-bounds")
	registration.Descriptor.ActionMin = 1
	registration.Descriptor.ActionMax = -1

	if err := runtime.RegisterTask(registration); err == nil {
		t.Fatal("RegisterTask() error = nil")
	}
}

func TestRegisterTaskRejectsNilFactory(t *testing.T) {
	runtime := framework.NewRuntime()
	registration := validRegistration("nil-factory")
	registration.Factory = nil

	if err := runtime.RegisterTask(registration); err == nil {
		t.Fatal("RegisterTask() error = nil")
	}
}

func TestRegisterTaskRejectsDuplicateName(t *testing.T) {
	runtime := framework.NewRuntime()
	registration := validRegistration("force-control")
	if err := runtime.RegisterTask(registration); err != nil {
		t.Fatalf("RegisterTask() first error = %v", err)
	}

	if err := runtime.RegisterTask(registration); err == nil {
		t.Fatal("RegisterTask() duplicate error = nil")
	}
}

func validRegistration(name string) framework.TaskRegistration {
	return framework.TaskRegistration{
		Descriptor: framework.TaskDescriptor{
			Name:            name,
			StateDimension:  6,
			ActionDimension: 1,
			ActionMin:       -1,
			ActionMax:       1,
		},
		Factory: externalTaskFactory{},
	}
}
