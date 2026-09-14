package framework

// State is a task observation vector.
type State []float32

// Observation is retained as an alias for State.
type Observation = State

// Action is a task action vector.
type Action []float32

// Outcome describes a task-specific result category.
type Outcome string

const (
	OutcomeRunning Outcome = "running"
	OutcomeSuccess Outcome = "success"
	OutcomeFailure Outcome = "failure"
)

// StepResult contains the state produced by one task step.
type StepResult struct {
	State   State
	Reward  float32
	Outcome Outcome
	Done    bool
	Info    map[string]float32
}
