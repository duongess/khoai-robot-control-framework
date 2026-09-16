package framework

// Task is an environment contract that external repositories can implement.
type Task interface {
	Reset() (State, error)
	Step(Action) (StepResult, error)
}

// ReviewableTask is optional. A human can approve a curriculum transition only
// while the runtime is paused; approval does not create a synthetic reward or
// successful replay transition.
type ReviewableTask interface {
	ApproveCurriculumReview() error
}

// TelemetryTask is optional. It lets an environment publish a small immutable
// metadata snapshot for an external visualizer without changing the policy
// state vector or coupling the framework to a task package.
type TelemetryTask interface {
	TelemetryMetadata() map[string]any
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
