.PHONY: test build proto

build:
	go build ./...

test:
	go test ./...

proto:
	./scripts/generate-proto.sh
