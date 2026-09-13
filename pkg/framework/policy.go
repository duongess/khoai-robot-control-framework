package framework

type Policy interface {
	Act(Observation) (Action, error)
}
