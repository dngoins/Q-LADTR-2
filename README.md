# Q-LADTR-2
Description The main goal of this code is to simulate and visualize the behavior of a network of Unmanned Aerial Vehicles (UAVs) using Q-learning. The UAVs have different roles (searching or ferrying) and their positions and velocities are updated over time. The Q-learning algorithm is used to optimize UAV actions to maximize a reward function.

## Current learning fix

The notebook now uses a spatial Q-learning state instead of only the UAV status string. Q-values are stored in a dictionary keyed by grid position, role, network condition, goodput, RTT, and buffer bins. Movement is also directional (`up`, `down`, `left`, `right`, `stay`) so the agent can learn where to move, not just whether to move.

## Future work plan

1. **Spatial Q-learning with grid-based states** - Continue tuning the dictionary Q-table state key around `(grid_x, grid_y, role, network_condition, goodput, rtt, buffer)` and compare grid resolutions such as 10x10, 20x20, and adaptive cells.
2. **Directional movement actions** - Evaluate the new `up`, `down`, `left`, `right`, and `stay` actions against prior random movement and track whether learned policies converge toward high-connectivity areas.
3. **Multi-agent Q-learning** - Compare one shared Q-table against per-UAV and per-role Q-tables so search UAVs and ferry UAVs can specialize without overwriting each other's policies.
4. **Reward shaping** - Tune rewards for ground-station distance, nearby ferry/search connectivity, goodput, RTT, buffer pressure, packet delivery, drops, and collision avoidance.
5. **Algorithm comparison** - Benchmark tabular Q-learning against DQN, PPO, A2C, and MADDPG using the same environment metrics and episode seeds.
6. **Obstacle and no-fly-zone constraints** - Add map masks for obstacles/no-fly zones, prevent invalid transitions, and penalize policy attempts to enter restricted cells.
7. **Realistic wireless signal model** - Replace random network condition changes with path-loss, distance, interference, and line-of-sight based signal quality.
8. **Battery-aware routing** - Add battery level to the state, energy costs to movement/communication, and rewards for safe return or charging behavior.
9. **Packet-level simulation** - Model queues, packet arrivals, forwarding decisions, delay, drops, throughput, and packet delivery ratio instead of only status flags.
10. **Live dashboard** - Build a dashboard for UAV paths, Q-values by grid cell, reward curves, packet delivery ratio, drops, delay, and learned policy changes over time.
