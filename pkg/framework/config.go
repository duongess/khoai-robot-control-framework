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
	WorkerCount       int
	TickInterval      time.Duration
	ReplayCapacity    int
	WarmupTransitions int
	TrainingBatchSize int
	TrainingInterval  int
	ReplaySampleSeed  int64
}

// DefaultRuntimeConfig returns conservative local development defaults.
func DefaultRuntimeConfig() RuntimeConfig {
	return RuntimeConfig{WorkerCount: 4, TickInterval: 50 * time.Millisecond, ReplayCapacity: 10_000, WarmupTransitions: 64, TrainingBatchSize: 32, TrainingInterval: 4, ReplaySampleSeed: 42}
}
