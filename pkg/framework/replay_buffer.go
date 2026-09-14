package framework

import (
	"errors"
	"math/rand"
	"sync"
)

// ReplayBuffer is a fixed-capacity, concurrency-safe transition ring buffer.
type ReplayBuffer struct {
	mu          sync.RWMutex
	transitions []Transition
	capacity    int
	next        int
	random      *rand.Rand
}

func NewReplayBuffer(capacity int, seed int64) (*ReplayBuffer, error) {
	if capacity <= 0 {
		return nil, errors.New("replay buffer capacity must be greater than zero")
	}
	return &ReplayBuffer{transitions: make([]Transition, 0, capacity), capacity: capacity, random: rand.New(rand.NewSource(seed))}, nil
}

func (b *ReplayBuffer) Add(transition Transition) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	transition = copyTransition(transition)
	if len(b.transitions) < b.capacity {
		b.transitions = append(b.transitions, transition)
		return
	}
	b.transitions[b.next] = transition
	b.next = (b.next + 1) % b.capacity
}

func (b *ReplayBuffer) Len() int {
	if b == nil {
		return 0
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	return len(b.transitions)
}

func (b *ReplayBuffer) Sample(count int) ([]Transition, error) {
	if b == nil {
		return nil, errors.New("replay buffer is nil")
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if count <= 0 {
		return nil, errors.New("sample count must be greater than zero")
	}
	if count > len(b.transitions) {
		return nil, errors.New("sample count exceeds replay buffer size")
	}
	indices := b.random.Perm(len(b.transitions))[:count]
	sample := make([]Transition, count)
	for i, index := range indices {
		sample[i] = copyTransition(b.transitions[index])
	}
	return sample, nil
}

func (b *ReplayBuffer) Clear() {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.transitions = b.transitions[:0]
	b.next = 0
}

func copyTransition(transition Transition) Transition {
	transition.Observation = append(State(nil), transition.Observation...)
	transition.Action = append(Action(nil), transition.Action...)
	transition.NextObservation = append(State(nil), transition.NextObservation...)
	return transition
}
