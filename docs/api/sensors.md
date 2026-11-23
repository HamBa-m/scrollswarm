# Sensors & Raycasting API

The `sensors` module implements the multi-modal perception suite equipped on each UAV, combining a 7-ray forward LiDAR rangefinder, nearest target tracking, and inter-drone neighbor proximity sensing.

---

## Overview

- **LiDAR Rangefinder:** 7 rays spanning $120^\circ$ FOV ($-60^\circ$ to $+60^\circ$), casting up to $R_{\max} = 400\text{ px}$.
- **Analytical Intersections:** Computes exact ray intersections with top/bottom corridor boundaries and polygon obstacle edges via vector cross-products.
- **Observation Construction:** `RaySensor.observe()` synthesizes the full normalized 16-dimensional continuous feature vector for policy conditioning.

---

## Quick Usage

```python
from scrollswarm.sensors import RaySensor
from scrollswarm.map import ScrollingMap, MapConfig

sensor = RaySensor(num_rays=7, max_range=400.0, fov_degrees=120.0)
scrolling_map = ScrollingMap(MapConfig())

# Cast LiDAR rays from coordinate (100, 200)
obs_hits, tgt_hits = sensor.cast_rays(origin_x=100.0, origin_y=200.0, scrolling_map=scrolling_map)

for i, hit in enumerate(obs_hits):
    print(f"Ray {i}: hit={hit.hit_type}, distance={hit.distance:.1f}px, norm={hit.norm_distance:.3f}")
```

---

## Class Reference

::: scrollswarm.sensors.RayHit

::: scrollswarm.sensors.RaySensor
