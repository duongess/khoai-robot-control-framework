# Architecture

The Go runtime owns task lifecycle (`Reset`, batched `Step`), replay, and the gRPC learner client. A local Python gRPC process owns SAC actor/critic updates. The protocol carries only numeric state, action, and transition messages.

The SAC critic remains an MLP. The actor is selected at startup: direct-action
`mlp`, dense six-parameter `parametric_mlp`, sparse degree-preserving
`random_graph`, or sparse directed `fly_connectome` loaded from a local neuPrint
cache. Both parametric actors produce the complete bounded vector
`theta = [a_x, b_x, a_y, b_y, a_g, b_g]` before evaluating the registered reflex;
the dense actor uses its MLP context and the graph actor uses its connectome
context. The graph actor uses an explicit sensory group, bounded recurrent
message passing, and six explicit motor readouts. It does not query neuPrint at
runtime and cannot introduce new edges. Task descriptor bounds are checked
before `Task.Step`; a registered robot task remains responsible for physical
safety.
