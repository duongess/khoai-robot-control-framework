package framework

import (
	"context"
	"net"
	"testing"
	"time"

	learnerv1 "github.com/duongess/khoai-robot-control-framework/gen/go/learner/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
)

const testBufferSize = 1024 * 1024

type learnerServiceStub struct {
	learnerv1.UnimplementedLearnerServiceServer
	predictRequest *learnerv1.PredictBatchRequest
	trainRequest   *learnerv1.TrainBatchRequest
}

func (s *learnerServiceStub) HealthCheck(context.Context, *learnerv1.HealthCheckRequest) (*learnerv1.HealthCheckResponse, error) {
	return &learnerv1.HealthCheckResponse{Ready: true}, nil
}

func (s *learnerServiceStub) PredictBatch(_ context.Context, request *learnerv1.PredictBatchRequest) (*learnerv1.PredictBatchResponse, error) {
	s.predictRequest = request
	return &learnerv1.PredictBatchResponse{
		Actions: []*learnerv1.Action{{Values: []float32{0.5}}, {Values: []float32{0.5}}},
	}, nil
}

func (s *learnerServiceStub) TrainBatch(_ context.Context, request *learnerv1.TrainBatchRequest) (*learnerv1.TrainBatchResponse, error) {
	s.trainRequest = request
	return &learnerv1.TrainBatchResponse{Accepted: true, SamplesSeen: uint64(len(request.Batch.Transitions)), PolicyVersion: 7}, nil
}

func TestLearnerClientHealthCheck(t *testing.T) {
	client, _ := newTestLearnerClient(t)

	health, err := client.HealthCheck(context.Background())
	if err != nil {
		t.Fatalf("HealthCheck() error = %v", err)
	}
	if !health.Ready {
		t.Fatal("HealthCheck() ready = false")
	}
}

func TestLearnerClientPredictBatchMapsStatesAndActions(t *testing.T) {
	client, service := newTestLearnerClient(t)

	prediction, err := client.PredictBatch(context.Background(), []State{{1, 2}, {3, 4}}, 7)
	if err != nil {
		t.Fatalf("PredictBatch() error = %v", err)
	}
	if len(service.predictRequest.States) != 2 || service.predictRequest.States[0].Values[0] != 1 {
		t.Fatalf("PredictBatch() request = %#v", service.predictRequest)
	}
	if service.predictRequest.PolicyVersion != 7 {
		t.Fatalf("PredictBatch() requested policy version = %d, want 7", service.predictRequest.PolicyVersion)
	}
	if len(prediction.Actions) != 2 || prediction.Actions[0][0] != 0.5 || prediction.Actions[1][0] != 0.5 {
		t.Fatalf("PredictBatch() actions = %#v", prediction.Actions)
	}
}

func TestLearnerClientTrainBatchMapsTransition(t *testing.T) {
	client, service := newTestLearnerClient(t)

	result, err := client.TrainBatch(context.Background(), []Transition{{
		Observation:     State{1},
		Action:          Action{0.25},
		Reward:          2,
		NextObservation: State{3},
		Done:            true,
	}})
	if err != nil {
		t.Fatalf("TrainBatch() error = %v", err)
	}
	transition := service.trainRequest.Batch.Transitions[0]
	if !transition.Terminated || transition.Reward != 2 || transition.State.Values[0] != 1 || transition.NextState.Values[0] != 3 {
		t.Fatalf("TrainBatch() transition = %#v", transition)
	}
	if !result.Accepted || result.SamplesSeen != 1 || result.PolicyVersion != 7 {
		t.Fatalf("TrainBatch() result = %#v", result)
	}
}

func TestLearnerClientClose(t *testing.T) {
	client, _ := newTestLearnerClient(t)

	if err := client.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func newTestLearnerClient(t *testing.T) (*LearnerClient, *learnerServiceStub) {
	t.Helper()

	listener := bufconn.Listen(testBufferSize)
	server := grpc.NewServer()
	service := &learnerServiceStub{}
	learnerv1.RegisterLearnerServiceServer(server, service)
	go func() {
		if err := server.Serve(listener); err != nil {
			t.Errorf("Serve() error = %v", err)
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	t.Cleanup(cancel)
	conn, err := grpc.DialContext(
		ctx,
		"bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		t.Fatalf("DialContext() error = %v", err)
	}
	t.Cleanup(func() {
		_ = conn.Close()
		server.Stop()
		_ = listener.Close()
	})

	return newLearnerClient(conn, time.Second), service
}
