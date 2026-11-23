# Sprites & Rendering API

The `sprites` module provides procedural vector-rendered quadcopter UAV graphics with distinct per-agent color tints, aerodynamic fuselages, rotor disc blur, and camera pods.

---

## Overview

- **Procedural Graphics:** Generates crisp Pygame Surfaces on-the-fly without external image assets.
- **Surface Caching:** Caches generated surfaces by `(color, size)` to minimize rendering overhead.
- **Quadcopter Features:** Dark carbon-fiber rotor arms, translucent spinning rotor blur discs, central fuselage, and optical reconnaissance camera pod.

---

## Quick Usage

```python
import pygame
from scrollswarm.sprites import get_quadcopter_sprite

pygame.init()

# Generate quadcopter sprite tinted Dodger Blue
drone_surface = get_quadcopter_sprite(color=(30, 144, 255), size=(24, 24))
print(f"Sprite size: {drone_surface.get_size()}")
```

---

## Function Reference

::: scrollswarm.sprites.get_quadcopter_sprite

::: scrollswarm.sprites.get_rocket_sprite
