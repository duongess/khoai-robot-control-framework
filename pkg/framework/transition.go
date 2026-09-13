package framework

type Transition struct {
	Observation     Observation
	Action          Action
	Reward          float32
	NextObservation Observation
	Done            bool
}
