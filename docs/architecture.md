# Architecture

The Go runtime owns task lifecycle (`Reset`, batched `Step`), replay, and the gRPC learner client. A local Python gRPC process owns SAC actor/critic updates. The protocol carries only numeric state, action, and transition messages.

The SAC critic remains an MLP. The actor is selected at startup: `mlp`, sparse degree-preserving `random_graph`, or sparse directed `fly_connectome` loaded from a local neuPrint cache. The graph actor uses an explicit sensory group, bounded recurrent message passing, and six explicit motor readouts. It does not query neuPrint at runtime and cannot introduce new edges. Task descriptor bounds are checked before `Task.Step`; a registered robot task remains responsible for physical safety.
