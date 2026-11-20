# ScrollSwarm

**A side-scrolling environment for cooperative multi-agent navigation and collection.**

ScrollSwarm is a small, visualizable testbed for studying how agents coordinate when they have limited local observations and can communicate only with nearby teammates.

The world scrolls from right to left. Agents occupy fixed horizontal columns and move vertically to avoid obstacles and collect targets. The initial environment is intentionally simple: it is designed to make the task and agent interactions easy to inspect, not to model realistic robot dynamics.

## The idea

At each step, randomized obstacles and targets enter the field. Each agent observes only its local field of view and chooses whether to move up, down, or stay. Agents can optionally exchange information with the agents in the neighboring columns.

The initial version uses compact vector observations. Pixel observations, variable agent speed, and energy constraints are possible extensions.

## Research progression

The environment is intended to support a sequence of experiments:

1. **Single-agent RL:** Can one agent avoid obstacles and collect targets?
2. **Multi-agent RL:** How does a team coordinate to improve collection and avoid collisions?
3. **Graph MARL:** What changes when agents communicate only with adjacent teammates?
4. **Robustness:** How does the team perform when one agent’s observation or message is perturbed?

## Initial design

- A configurable number of agents, starting with a small team
- Fixed x-axis columns and vertical movement
- A right-to-left scrolling world
- Randomized obstacle and target placement
- Local fields of view
- Neighbor communication along a chain graph
- Shared team-level task objective
- Reproducible episode generation through random seeds

The first version will use discrete `up`, `stay`, and `down` actions with constant scrolling speed. Continuous speed control and day/night energy constraints are potential later extensions.

## Scope and limitations

ScrollSwarm is an abstract 2D game-like environment. Results from it should be interpreted as results in this simplified setting; they do not establish performance on physical robots or realistic swarm dynamics.

## License

To be decided.