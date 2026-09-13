package framework

// Task is an environment contract that external repositories can implement.
type Task interface {
	Reset() (State, error)
	Step(Action) (StepResult, error)
}

// TaskFactory creates independent task instances.
type TaskFactory interface {
	Create() (Task, error)
}

// TaskDescriptor describes a task's state and action spaces.
type TaskDescriptor struct {
	Name            string
	StateDimension  int
	ActionDimension int
	ActionMin       float32
	ActionMax       float32
}

// TaskRegistration couples a task descriptor with its factory.
type TaskRegistration struct {
	Descriptor TaskDescriptor
	Factory    TaskFactory
}
