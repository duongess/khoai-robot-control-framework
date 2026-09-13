package framework

import (
	"context"
	"errors"
	"fmt"
	"time"

	learnerv1 "github.com/duongess/khoai-robot-control-framework/gen/go/learner/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const DefaultLearnerAddress = "127.0.0.1:50051"

const learnerRequestTimeout = 5 * time.Second

type TrainBatchResult struct {
	Accepted      bool
	SamplesSeen   uint64
	PolicyVersion string
}

type LearnerClient struct {
	conn    *grpc.ClientConn
	client  learnerv1.LearnerServiceClient
	timeout time.Duration
}

func NewLearnerClient(ctx context.Context) (*LearnerClient, error) {
	if ctx == nil {
		return nil, errors.New("context is required")
	}

	dialCtx, cancel := context.WithTimeout(ctx, learnerRequestTimeout)
	defer cancel()

	conn, err := grpc.DialContext(
		dialCtx,
		DefaultLearnerAddress,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return nil, fmt.Errorf("connect to learner at %s: %w", DefaultLearnerAddress, err)
	}

	return newLearnerClient(conn, learnerRequestTimeout), nil
}

func (c *LearnerClient) HealthCheck(ctx context.Context) (bool, error) {
	callCtx, cancel, err := c.requestContext(ctx)
	if err != nil {
		return false, err
	}
	defer cancel()

	response, err := c.client.HealthCheck(callCtx, &learnerv1.HealthCheckRequest{})
	if err != nil {
		return false, fmt.Errorf("health check: %w", err)
	}

	return response.GetReady(), nil
}

func (c *LearnerClient) PredictBatch(ctx context.Context, states []State) ([]Action, error) {
	callCtx, cancel, err := c.requestContext(ctx)
	if err != nil {
		return nil, err
	}
	defer cancel()

	request := &learnerv1.PredictBatchRequest{States: make([]*learnerv1.State, len(states))}
	for index, state := range states {
		request.States[index] = &learnerv1.State{Values: append([]float32(nil), state...)}
	}

	response, err := c.client.PredictBatch(callCtx, request)
	if err != nil {
		return nil, fmt.Errorf("predict batch: %w", err)
	}
	if len(response.Actions) != len(states) {
		return nil, fmt.Errorf("predict batch: received %d actions for %d states", len(response.Actions), len(states))
	}

	actions := make([]Action, len(response.Actions))
	for index, action := range response.Actions {
		actions[index] = append(Action(nil), action.GetValues()...)
	}
	return actions, nil
}

func (c *LearnerClient) TrainBatch(ctx context.Context, transitions []Transition) (TrainBatchResult, error) {
	callCtx, cancel, err := c.requestContext(ctx)
	if err != nil {
		return TrainBatchResult{}, err
	}
	defer cancel()

	request := &learnerv1.TrainBatchRequest{
		Batch: &learnerv1.TransitionBatch{Transitions: make([]*learnerv1.Transition, len(transitions))},
	}
	for index, transition := range transitions {
		request.Batch.Transitions[index] = transitionToProto(transition)
	}

	response, err := c.client.TrainBatch(callCtx, request)
	if err != nil {
		return TrainBatchResult{}, fmt.Errorf("train batch: %w", err)
	}

	return TrainBatchResult{
		Accepted:      response.GetAccepted(),
		SamplesSeen:   response.GetSamplesSeen(),
		PolicyVersion: response.GetPolicyVersion(),
	}, nil
}

func (c *LearnerClient) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

func newLearnerClient(conn *grpc.ClientConn, timeout time.Duration) *LearnerClient {
	return &LearnerClient{
		conn:    conn,
		client:  learnerv1.NewLearnerServiceClient(conn),
		timeout: timeout,
	}
}

func (c *LearnerClient) requestContext(ctx context.Context) (context.Context, context.CancelFunc, error) {
	if ctx == nil {
		return nil, nil, errors.New("context is required")
	}
	callCtx, cancel := context.WithTimeout(ctx, c.timeout)
	return callCtx, cancel, nil
}

func transitionToProto(transition Transition) *learnerv1.Transition {
	return &learnerv1.Transition{
		State:      &learnerv1.State{Values: append([]float32(nil), transition.Observation...)},
		Action:     &learnerv1.Action{Values: append([]float32(nil), transition.Action...)},
		Reward:     transition.Reward,
		NextState:  &learnerv1.State{Values: append([]float32(nil), transition.NextObservation...)},
		Terminated: transition.Done,
	}
}
