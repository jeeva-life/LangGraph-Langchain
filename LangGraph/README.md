# SUPER STEP:
It is considered as a single iteration over the graph nodes. Nodes that run in parallel are part of same super-step, while nodes that run sequentially belong to seperate super-steps.

The graph describes one super-step; one interaction between agents and tools to achieve an outcome

Every user interaction is a fresh "graph.invoke(state)" call

The reducer handles updating state during a super-step but not between super-steps
