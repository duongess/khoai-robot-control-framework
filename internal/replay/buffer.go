package replay

type Buffer interface {
	Add(interface{})
	Len() int
}
