package framework

import "time"

// Config configures a local learner connection.
type Config struct {
	Address        string
	ConnectTimeout time.Duration
	RequestTimeout time.Duration
}

// DefaultConfig returns safe local learner connection settings.
func DefaultConfig() Config {
	return Config{Address: DefaultLearnerAddress, ConnectTimeout: 5 * time.Second, RequestTimeout: 5 * time.Second}
}

// RuntimeConfig controls batched environment collection and training.
type RuntimeConfig struct {
	WorkerCount    int
	TickInterval   time.Duration
	ReplayCapacity int
	// RandomActionWarmupTransitions collects broad, bounded exploration before
	// actor actions are used. It is distinct from the replay warm-up threshold.
	RandomActionWarmupTransitions int
	WarmupTransitions             int
	TrainingBatchSize             int
	TrainingInterval              int
	ReplaySampleSeed              int64
}

// DefaultRuntimeConfig returns conservative local development defaults.
func DefaultRuntimeConfig() RuntimeConfig {
	return RuntimeConfig{
		WorkerCount:                   4,
		TickInterval:                  50 * time.Millisecond,
		ReplayCapacity:                10_000,
		RandomActionWarmupTransitions: 256,
		WarmupTransitions:             64,
		TrainingBatchSize:             32,
		// Four workers add four transitions per 50 ms tick. Training every 32
		// transitions leaves enough CPU time for control/inference and avoids
		// starving the dashboard on a local machine.
		TrainingInterval:              32,
		ReplaySampleSeed:              42,
	}
}
