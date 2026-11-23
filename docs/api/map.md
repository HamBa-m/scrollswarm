# Map & Entities API

The `map` and `items` modules govern the scrolling airspace corridor, procedural multi-scale obstacle generation, target waypoints, and the guaranteed navigable passage invariant ($G_{\min} \ge 110\text{ px}$).

---

## Overview

- **Airspace Geometry:** Configurable $W \times H$ 2D corridor with ceiling and floor safety margins.
- **Scrolling Motion:** Global horizontal leftward drift ($v_{\text{scroll}} = 1\text{ px/step}$).
- **Procedural Obstacles:** Ground structures, overhead barriers, and oscillating mid-air debris.
- **Reconnaissance Targets:** Stochastic waypoints with proximity-dependent camera Bernoulli trials.

---

## Quick Usage

```python
from scrollswarm.map import ScrollingMap, MapConfig

config = MapConfig(
    width=800,
    height=400,
    scroll_speed=1,
    min_corridor_gap=110,
    seed=42,
)

scrolling_map = ScrollingMap(config)
scrolling_map.reset(seed=42)

# Step map world
scrolling_map.step()
print(f"Active obstacles: {len(scrolling_map.obstacles)}, Targets: {len(scrolling_map.targets)}")
```

---

## Class & Dataclass Reference

::: scrollswarm.map.MapConfig

::: scrollswarm.map.ScrollingMap

::: scrollswarm.items.ObstacleType

::: scrollswarm.items.Obstacle

::: scrollswarm.items.Target
