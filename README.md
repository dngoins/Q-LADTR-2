# Q-LADTR-2
Description The main goal of this code is to simulate and visualize the behavior of a network of Unmanned Aerial Vehicles (UAVs) using Q-learning. The UAVs have different roles (searching or ferrying) and their positions and velocities are updated over time. The Q-learning algorithm is used to optimize UAV actions to maximize a reward function.

## Current learning fix

The notebook now uses a spatial Q-learning state instead of only the UAV status string. Q-values are stored in a dictionary keyed by grid position, role, network condition, goodput, RTT, and buffer bins. Movement is also directional (`up`, `down`, `left`, `right`, `stay`) so the agent can learn where to move, not just whether to move.

## Directional movement comparison

Future work item 2 is implemented as a notebook comparison workflow. Run `compare_movement_strategies(...)` to compare the learned directional movement actions against a `random_movement` baseline that uses the prior random velocity transition model. The returned DataFrame reports average reward, final average reward, learned state count, movement action count, movement action rate, and random movement transition count.

## Multi-agent Q-learning comparison

Future work item 3 is implemented with configurable Q-table scopes. Set `q_table_scope` to `shared`, `per_role`, or `per_uav` when calling `train_q_learning_agent(...)`, or run `compare_q_table_scopes(...)` to compare all three modes using the same seed and environment settings.

## Reward shaping profiles

Future work item 4 is implemented with configurable reward weights and per-component reward totals. Pass `reward_weights` into `train_q_learning_agent(...)`, or run `compare_reward_profiles(...)` to compare the baseline, connectivity-focused, and drop-avoidance-focused reward profiles.

## Validation graphs

Use the validation helpers to generate the evidence plots for convergence and coverage:

```python
episode_df, summary_df, agents = run_validation_experiments(
    seeds=(1, 2, 3, 4, 5),
    num_episodes=500,
    num_uavs=10,
    max_steps=50,
)

best_agent = agents[(1, "directional", "shared")]
plot_learning_validation_graphs(episode_df, summary_df, agent=best_agent)
```

The generated plots cover average reward, ferry UAV distance to the ground station, network condition, goodput, RTT, buffer pressure, packet drops, directional versus random movement, Q-table scope comparisons, per-seed final rewards, and learned policy arrows.

## Network condition modeling

The current notebook still uses a simplified network condition model. A stronger model would compute network quality from radio features such as RSSI/RSRP, SINR/SNR, path loss, interference, bandwidth, packet loss, and queueing delay, then map those values into `network_condition`, `goodput`, and `rtt`.

Useful options:

1. **Analytical model** - Use free-space or log-distance path loss, shadow fading, SINR, and Shannon capacity. This is the easiest way to make UAV distance and relay placement physically meaningful.
2. **3GPP-inspired model** - Use 3GPP TR 36.777 or TR 38.901 style air-to-ground path-loss and line-of-sight probability models for UAV-to-ground links.
3. **Public traces** - Use CRAWDAD wireless traces, IEEE DataPort wireless/UAV datasets, OpenCelliD cell tower locations, or WiFi RSSI datasets such as UCI Wireless Indoor Localization to calibrate RSSI/path-loss behavior.
4. **Simulator-generated traces** - Use ns-3 with LTE/5G/WiFi modules to generate repeatable RSSI, SINR, throughput, delay, and packet loss data for controlled UAV movement scenarios.

## Future work plan

1. **Spatial Q-learning with grid-based states** - Continue tuning the dictionary Q-table state key around `(grid_x, grid_y, role, network_condition, goodput, rtt, buffer)` and compare grid resolutions such as 10x10, 20x20, and adaptive cells.
2. **Directional movement actions** - Implemented a comparison workflow for learned `up`, `down`, `left`, `right`, and `stay` actions against the prior random movement model using `compare_movement_strategies(...)`.
3. **Multi-agent Q-learning** - Implemented configurable Q-table scopes with `q_table_scope="shared"`, `"per_role"`, or `"per_uav"` and `compare_q_table_scopes(...)`.
4. **Reward shaping** - Implemented configurable reward weights plus `reward_component_totals` and `compare_reward_profiles(...)`.
5. **Algorithm comparison** - Benchmark tabular Q-learning against DQN, PPO, A2C, and MADDPG using the same environment metrics and episode seeds.
6. **Obstacle and no-fly-zone constraints** - Add map masks for obstacles/no-fly zones, prevent invalid transitions, and penalize policy attempts to enter restricted cells.
7. **Realistic wireless signal model** - Replace random network condition changes with path-loss, distance, interference, and line-of-sight based signal quality.
8. **Battery-aware routing** - Add battery level to the state, energy costs to movement/communication, and rewards for safe return or charging behavior.
9. **Packet-level simulation** - Model queues, packet arrivals, forwarding decisions, delay, drops, throughput, and packet delivery ratio instead of only status flags.
10. **Live dashboard** - Build a dashboard for UAV paths, Q-values by grid cell, reward curves, packet delivery ratio, drops, delay, and learned policy changes over time.
