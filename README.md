# Q-LADTR-2

The main goal of this updated notebook is to simulate, train, validate, and benchmark a location-aware UAV network using Q-learning and comparison baselines. The UAVs operate as search or ferry UAVs, move through a grid, forward packets toward a ground station, and learn policies that should improve coverage and packet-delivery conditions over time.

## Relationship to the original Q-LADTR work

This repository is an update to the original Q-LADTR published work on location-aware multi-drone networking with intelligent packet forwarding. The original work framed the UAV network as a Q-learning routing problem where search UAVs, ferry UAVs, and a ground station cooperate to forward data under changing network conditions.

The updated notebook keeps that Q-LADTR objective, but strengthens the implementation so it can test whether UAVs actually learn useful spatial behavior. It adds spatial states, learned directional movement, validation metrics, oracle-scored grid targets, and algorithm baselines so the original routing idea can be evaluated against clearer evidence.

## What is new or updated

1. **Spatial learning instead of status-only learning** - The Q-table now includes grid position, role, network condition, goodput, RTT, and buffer bins. This addresses the main insight from the notebook review: UAVs cannot learn "best areas" unless area and connectivity are part of the state.
2. **Learned directional movement** - Movement is represented by explicit actions (`up`, `down`, `left`, `right`, `stay`) instead of random velocity changes, so Q-learning can learn where to move.
3. **Cleaner simulation behavior** - The update fixes UAV identity handling, avoids accidental Q-table mutation during action selection, and uses one UAV update per UAV per frame/step.
4. **Oracle and baseline comparisons** - Synthetic oracle data defines ideal search/ferry placements and per-cell network quality so random movement, directional Q-LADTR, and the min/max oracle upper bound can be compared on the same grid.
5. **Validation and new insights** - The notebook now reports convergence, reward trends, packet drops, policy arrows, runtime, oracle distance, role match rate, and algorithm comparisons. These outputs make it easier to explain when directional Q-LADTR improves over random behavior and why the oracle remains the expected upper bound.

## Current learning fix

The notebook now uses a spatial Q-learning state instead of only the UAV status string. Q-values are stored in a dictionary keyed by grid position, role, network condition, goodput, RTT, and buffer bins. Movement is also directional (`up`, `down`, `left`, `right`, `stay`) so the agent can learn where to move, not just whether to move.

## Directional movement comparison

The directional movement comparison is implemented as a notebook workflow. Run `compare_movement_strategies(...)` to compare the learned directional movement actions against a `random_movement` baseline that uses the prior random velocity transition model. The returned DataFrame reports average reward, final average reward, learned state count, movement action count, movement action rate, and random movement transition count.

## Multi-agent Q-learning comparison

Configurable Q-table scopes are implemented. Set `q_table_scope` to `shared`, `per_role`, or `per_uav` when calling `train_q_learning_agent(...)`, or run `compare_q_table_scopes(...)` to compare all three modes using the same seed and environment settings.

## Reward shaping profiles

Configurable reward weights and per-component reward totals are implemented. Pass `reward_weights` into `train_q_learning_agent(...)`, or run `compare_reward_profiles(...)` to compare the baseline, connectivity-focused, and drop-avoidance-focused reward profiles.

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

## Synthetic oracle dataset and min/max baseline

The repository includes deterministic synthetic oracle data for the 10x10 grid:

| File | Purpose |
| --- | --- |
| `data/synthetic_uav_ideal_placements.csv` | One ideal role and grid placement per UAV. The default 10-UAV layout uses a 6 search / 4 ferry ratio. |
| `data/synthetic_uav_network_conditions.csv` | Per-UAV, per-grid-cell synthetic network condition, goodput, RTT, buffer pressure, packet drop probability, and oracle score. |

Use these helpers to regenerate or benchmark against the oracle:

```python
oracle_df, ideal_placements = generate_synthetic_network_oracle(
    num_uavs=10,
    grid_size=10,
    search_ratio=0.6,
    save_prefix="synthetic_uav",
)

q_summary, q_scores, oracle_df, ideal_placements = run_oracle_benchmark(
    seeds=(1, 2, 3),
    num_episodes=500,
    num_uavs=10,
    grid_size=10,
    search_ratio=0.6,
    movement_strategies=("random_movement", "directional"),
)

minmax_summary, minmax_rows, _, _ = run_minmax_oracle_baseline(
    num_uavs=10,
    grid_size=10,
    search_ratio=0.6,
)

plot_oracle_benchmark(q_summary, minmax_summary)
plot_oracle_grid_view(oracle_df, ideal_placements, metric="ideal_score")
```

The oracle benchmark records elapsed runtime, seconds per episode, distance to oracle placement, role match rate, network condition, goodput, RTT, buffer pressure, and packet drops for both the `random_movement` baseline and directional Q-LADTR. The min/max oracle baseline is intentionally simple and fast: it directly picks the highest oracle-scored grid cell per UAV. It is useful as an upper-bound or sanity-check baseline before comparing Q-LADTR with DQN, PPO, A2C, or MADDPG. Use `plot_oracle_grid_view(...)` to show the oracle score heatmap with ideal search/ferry UAV placements overlaid on the grid.

## Algorithm comparison

The notebook includes a comparison section for the direct min/max oracle upper bound, the `random_movement` baseline, tabular directional Q-LADTR, and optional DQN, PPO, and A2C baselines. The notebook presents the results in that sequence: oracle target and ideal placements first, then random movement, then directional Q-LADTR, then the optional deep-RL algorithms. Q-LADTR, `random_movement`, and `minmax_oracle` run with the existing notebook dependencies. DQN, PPO, and A2C require Stable-Baselines3 and Gymnasium:

```python
%pip install stable-baselines3 gymnasium
```

After installing those packages, set:

```python
run_deep_rl_algorithms = True
```

The algorithm comparison cell combines min/max oracle, random movement, directional Q-LADTR, and optional DQN/PPO/A2C rows into one summary table and chart set using the same oracle scoring metrics. It also prints a final interpretation explaining which algorithm had the best oracle-position match, which learned policy performed best, which method was closest to oracle placements, and which ran fastest. Keep `future5_total_timesteps` small while testing, then increase it for stronger deep-RL training. MADDPG is not included because Stable-Baselines3 does not provide it; adding MADDPG will require a multi-agent RL library or custom implementation.

## Network condition modeling

The current notebook still uses a simplified network condition model. A stronger model would compute network quality from radio features such as RSSI/RSRP, SINR/SNR, path loss, interference, bandwidth, packet loss, and queueing delay, then map those values into `network_condition`, `goodput`, and `rtt`.

Useful options:

1. **Analytical model** - Use free-space or log-distance path loss, shadow fading, SINR, and Shannon capacity. This is the easiest way to make UAV distance and relay placement physically meaningful.
2. **3GPP-inspired model** - Use 3GPP TR 36.777 or TR 38.901 style air-to-ground path-loss and line-of-sight probability models for UAV-to-ground links.
3. **Public traces** - Use CRAWDAD wireless traces, IEEE DataPort wireless/UAV datasets, OpenCelliD cell tower locations, or WiFi RSSI datasets such as UCI Wireless Indoor Localization to calibrate RSSI/path-loss behavior.
4. **Simulator-generated traces** - Use ns-3 with LTE/5G/WiFi modules to generate repeatable RSSI, SINR, throughput, delay, and packet loss data for controlled UAV movement scenarios.

## Future work plan

1. **Obstacle and no-fly-zone constraints** - Add map masks for obstacles/no-fly zones, prevent invalid transitions, and penalize policy attempts to enter restricted cells.
2. **Realistic wireless signal model** - Replace random network condition changes with path-loss, distance, interference, and line-of-sight based signal quality.
3. **Battery-aware routing** - Add battery level to the state, energy costs to movement/communication, and rewards for safe return or charging behavior.
4. **Packet-level simulation** - Model queues, packet arrivals, forwarding decisions, delay, drops, throughput, and packet delivery ratio instead of only status flags.
5. **Live dashboard** - Build a dashboard for UAV paths, Q-values by grid cell, reward curves, packet delivery ratio, drops, delay, and learned policy changes over time.
