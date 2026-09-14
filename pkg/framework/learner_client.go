package framework

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	learnerv1 "github.com/duongess/khoai-robot-control-framework/gen/go/learner/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const DefaultLearnerAddress = "127.0.0.1:50051"

// Learner is the framework-owned abstraction for the SAC transport.
type Learner interface {
	HealthCheck(context.Context) (HealthStatus, error)
	PredictBatch(context.Context, []State) (PredictionResult, error)
	TrainBatch(context.Context, []Transition) (TrainingResult, error)
	Close() error
}

type HealthStatus struct {
	Ready         bool
	PolicyVersion uint64
	TrainingStep  uint64
	Device        string
}

type PredictionResult struct {
	Actions       []Action
	PolicyVersion uint64
}

type TrainingResult struct {
	Accepted      bool
	SamplesSeen   uint64
	ActorLoss     float32
	CriticLoss    float32
	AlphaLoss     float32
	Entropy       float32
	PolicyVersion uint64
	TrainingStep  uint64
}

// LearnerClient is the gRPC adapter. Generated protobuf types stay private to it.
type LearnerClient struct {
	conn    *grpc.ClientConn
	client  learnerv1.LearnerServiceClient
	timeout time.Duration
}

func NewLearnerClient(ctx context.Context) (*LearnerClient, error) {
	return NewLearnerClientWithConfig(ctx, DefaultConfig())
}

func NewLearnerClientWithConfig(ctx context.Context, config Config) (*LearnerClient, error) {
	if ctx == nil {
		return nil, errors.New("context is required")
	}
	if config.Address == "" {
		config.Address = DefaultLearnerAddress
	}
	if config.ConnectTimeout <= 0 {
		config.ConnectTimeout = 5 * time.Second
	}
	if config.RequestTimeout <= 0 {
		config.RequestTimeout = 5 * time.Second
	}
	dialContext, cancel := context.WithTimeout(ctx, config.ConnectTimeout)
	defer cancel()
	conn, err := grpc.DialContext(dialContext, config.Address, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
	if err != nil {
		return nil, fmt.Errorf("connect to learner at %s: %w", config.Address, err)
	}
	return newLearnerClient(conn, config.RequestTimeout), nil
}

func (c *LearnerClient) HealthCheck(ctx context.Context) (HealthStatus, error) {
	callContext, cancel, err := c.requestContext(ctx)
	if err != nil {
		return HealthStatus{}, err
	}
	defer cancel()
	response, err := c.client.HealthCheck(callContext, &learnerv1.HealthCheckRequest{})
	if err != nil {
		return HealthStatus{}, fmt.Errorf("learner health check: %w", err)
	}
	return HealthStatus{Ready: response.GetReady(), PolicyVersion: response.GetPolicyVersion(), TrainingStep: response.GetTrainingStep(), Device: response.GetDevice()}, nil
}

func (c *LearnerClient) PredictBatch(ctx context.Context, states []State) (PredictionResult, error) {
	if len(states) == 0 {
		return PredictionResult{}, errors.New("predict batch requires at least one state")
	}
	request := &learnerv1.PredictBatchRequest{States: make([]*learnerv1.State, len(states))}
	for index, state := range states {
		if err := validateFinite("state", state); err != nil {
			return PredictionResult{}, fmt.Errorf("predict batch state %d: %w", index, err)
		}
		request.States[index] = &learnerv1.State{Values: append([]float32(nil), state...)}
	}
	callContext, cancel, err := c.requestContext(ctx)
	if err != nil {
		return PredictionResult{}, err
	}
	defer cancel()
	response, err := c.client.PredictBatch(callContext, request)
	if err != nil {
		return PredictionResult{}, fmt.Errorf("predict batch: %w", err)
	}
	if len(response.GetActions()) != len(states) {
		return PredictionResult{}, fmt.Errorf("predict batch returned %d actions for %d states", len(response.GetActions()), len(states))
	}
	actions := make([]Action, len(response.GetActions()))
	for index, action := range response.GetActions() {
		values := action.GetValues()
		if err := validateFinite("action", values); err != nil {
			return PredictionResult{}, fmt.Errorf("predict batch action %d: %w", index, err)
		}
		actions[index] = append(Action(nil), values...)
	}
	return PredictionResult{Actions: actions, PolicyVersion: response.GetPolicyVersion()}, nil
}

func (c *LearnerClient) TrainBatch(ctx context.Context, transitions []Transition) (TrainingResult, error) {
	if len(transitions) == 0 {
		return TrainingResult{}, errors.New("train batch requires at least one transition")
	}
	request := &learnerv1.TrainBatchRequest{Batch: &learnerv1.TransitionBatch{Transitions: make([]*learnerv1.Transition, len(transitions))}}
	for index, transition := range transitions {
		if err := validateTransition(transition); err != nil {
			return TrainingResult{}, fmt.Errorf("train batch transition %d: %w", index, err)
		}
		request.Batch.Transitions[index] = transitionToProto(transition)
	}
	callContext, cancel, err := c.requestContext(ctx)
	if err != nil {
		return TrainingResult{}, err
	}
	defer cancel()
	response, err := c.client.TrainBatch(callContext, request)
	if err != nil {
		return TrainingResult{}, fmt.Errorf("train batch: %w", err)
	}
	metrics := []float32{response.GetActorLoss(), response.GetCriticLoss(), response.GetAlphaLoss(), response.GetEntropy()}
	if err := validateFinite("training metric", metrics); err != nil {
		return TrainingResult{}, err
	}
	return TrainingResult{Accepted: response.GetAccepted(), SamplesSeen: response.GetSamplesSeen(), ActorLoss: response.GetActorLoss(), CriticLoss: response.GetCriticLoss(), AlphaLoss: response.GetAlphaLoss(), Entropy: response.GetEntropy(), PolicyVersion: response.GetPolicyVersion(), TrainingStep: response.GetTrainingStep()}, nil
}

func (c *LearnerClient) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

func newLearnerClient(conn *grpc.ClientConn, timeout time.Duration) *LearnerClient {
	return &LearnerClient{conn: conn, client: learnerv1.NewLearnerServiceClient(conn), timeout: timeout}
}

func (c *LearnerClient) requestContext(ctx context.Context) (context.Context, context.CancelFunc, error) {
	if ctx == nil {
		return nil, nil, errors.New("context is required")
	}
	if c == nil || c.client == nil {
		return nil, nil, errors.New("learner client is not initialized")
	}
	callContext, cancel := context.WithTimeout(ctx, c.timeout)
	return callContext, cancel, nil
}

func validateFinite(label string, values []float32) error {
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("%s contains a non-finite value", label)
		}
	}
	return nil
}

func validateTransition(transition Transition) error {
	if err := validateFinite("state", transition.Observation); err != nil {
		return err
	}
	if err := validateFinite("action", transition.Action); err != nil {
		return err
	}
	if err := validateFinite("next state", transition.NextObservation); err != nil {
		return err
	}
	if math.IsNaN(float64(transition.Reward)) || math.IsInf(float64(transition.Reward), 0) {
		return errors.New("reward is non-finite")
	}
	return nil
}

func transitionToProto(transition Transition) *learnerv1.Transition {
	return &learnerv1.Transition{State: &learnerv1.State{Values: append([]float32(nil), transition.Observation...)}, Action: &learnerv1.Action{Values: append([]float32(nil), transition.Action...)}, Reward: transition.Reward, NextState: &learnerv1.State{Values: append([]float32(nil), transition.NextObservation...)}, Terminated: transition.Done}
}
