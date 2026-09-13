package framework

type Environment interface {
	Reset() (Observation, error)
	Step(Action) (Transition, error)
}
