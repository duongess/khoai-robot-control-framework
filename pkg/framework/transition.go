package framework

// Transition records one task action and its resulting state.
type Transition struct {
	Observation     Observation
	Action          Action
	Reward          float32
	NextObservation Observation
	Outcome         Outcome
	Done            bool
}
