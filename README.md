# ScrollSwarm

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Managed with uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9.svg?logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![Renderer](https://img.shields.io/badge/renderer-pygame--2.6-green.svg?logo=python&logoColor=white)](https://www.pygame.org/)
[![Tests](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)](tests/)
[![Status](https://img.shields.io/badge/status-active%20research-informational.svg)](README.md)

**A side-scrolling environment for cooperative multi-agent navigation and collection.**

![ScrollSwarm Environment Preview](assets/screenshot.png)

ScrollSwarm is a lightweight, visualizable testbed for studying how agents coordinate when they have limited local observations and can communicate only with nearby teammates.

The world scrolls continuously from right to left at 1 pixel per timestep ($30\text{ FPS}$). Agents occupy fixed horizontal columns and move vertically to avoid obstacles and collect targets. The environment is designed to make multi-agent interactions, graph communication, and failure modes transparent and easy to inspect.

---

## Quickstart

ScrollSwarm uses [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency and environment management.

### Run Interactive Simulation
```bash
# Launch with default swarm (5 agents)
uv run python main.py

# Launch with custom swarm size (1 to 10 agents)
uv run python main.py -n 3
uv run python main.py -n 8
```

### Controls
| Key | Action |
| :--- | :--- |
| **`W` / `UP`** | Move focused agent up ($5\text{ px}$) |
| **`S` / `DOWN`** | Move focused agent down ($5\text{ px}$) |
| **`TAB`** / **`1`–`9`** | Switch active focused agent |
| **`A`** | Move all alive agents synchronously |
| **`R`** | Restart simulation & reset corridor |
| **`SPACE`** | Pause / Resume |
| **`ESC`** | Exit |

### Run Tests
```bash
uv run pytest
```

---

## The Idea

At each step, randomized obstacles and targets enter the field. Each agent observes only its local field of view and chooses whether to move up, down, or stay. Agents can optionally exchange information with the agents in the neighboring columns.

The initial version uses compact vector observations. Pixel observations, variable agent speed, and energy constraints are planned extensions.

---

## Research Progression

The environment is designed to support a structured progression of experiments:

1. **Single-Agent RL:** Can one agent avoid obstacles and collect targets?
2. **Multi-Agent RL:** How does a team coordinate to improve collection and avoid collisions?
3. **Graph MARL:** What changes when agents communicate only with adjacent teammates?
4. **Robustness:** How does the team perform when one agent’s observation or message is perturbed?

---

## Environment Design

- **Grid Geometry:** $2:1$ rectangular aspect ratio ($800 \times 400\text{ px}$ default).
- **Pixel-Level Granularity:** $1\text{ px/timestep}$ right-to-left scrolling.
- **Topography:** Continuous mountain ridges along boundary edges generated via discretized Ornstein-Uhlenbeck (mean-reverting Brownian motion) with guaranteed corridor clearance.
- **Entities:**
  - **Land:** Plain light green terrain.
  - **Obstacles:** Gray boundary mountains and dark green circular trees with brown trunks. Instant termination on collision.
  - **Targets:** Red 'X' marks collected on agent contact.
  - **Swarm:** $1 \le N \le 10$ triangle agents (pointing right) occupying fixed, evenly spaced horizontal columns.

---

## Scope and Limitations

ScrollSwarm is an abstract 2D game-like environment. Results from it should be interpreted as results in this simplified setting; they do not establish performance on physical robots or realistic swarm dynamics.

---

## License

To be decided.
