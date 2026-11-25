"""
ScrollSwarm: A side-scrolling multi-agent reinforcement learning testbed.
"""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from scrollswarm.env import ScrollSwarmEnv
from scrollswarm.map import (
    ScrollingMap,
    MapConfig,
    COLOR_SPACE_VOID,
    COLOR_RING_HAZE,
    COLOR_RING_SHEATH,
    COLOR_RING_RIM,
    COLOR_ASTEROID_FILL,
    COLOR_ASTEROID_OUTLINE,
    COLOR_ASTEROID_SPECULAR,
    COLOR_LAND,
    COLOR_MOUNTAIN,
    COLOR_MOUNTAIN_RIM,
    COLOR_TREE_LEAVES,
    COLOR_TREE_RIM,
    COLOR_TREE_TRUNK,
    COLOR_TARGET,
)
from scrollswarm.agent import (
    Agent,
    Action,
    AGENT_PALETTE,
    COLOR_AGENT_DEFAULT,
    COLOR_AGENT_DEAD,
    create_agents,
)
from scrollswarm.sensors import RaySensor, RayHit
from scrollswarm.items import Asteroid, Tree, Target
from scrollswarm.comm import CommunicationGraph, TopologyType
from scrollswarm.sprites import get_rocket_sprite, get_quadcopter_sprite
from scrollswarm.geometry import (
    get_available_backends,
    resolve_backend,
    is_numba_available,
    is_cpp_available,
)

__all__ = [
    "ScrollSwarmEnv",
    "ScrollingMap",
    "MapConfig",
    "Agent",
    "Action",
    "create_agents",
    "AGENT_PALETTE",
    "COLOR_AGENT_DEFAULT",
    "COLOR_AGENT_DEAD",
    "RaySensor",
    "RayHit",
    "Asteroid",
    "Tree",
    "Target",
    "CommunicationGraph",
    "TopologyType",
    "get_rocket_sprite",
    "get_quadcopter_sprite",
    "get_available_backends",
    "resolve_backend",
    "is_numba_available",
    "is_cpp_available",
    "COLOR_SPACE_VOID",
    "COLOR_RING_HAZE",
    "COLOR_RING_SHEATH",
    "COLOR_RING_RIM",
    "COLOR_ASTEROID_FILL",
    "COLOR_ASTEROID_OUTLINE",
    "COLOR_ASTEROID_SPECULAR",
    "COLOR_LAND",
    "COLOR_MOUNTAIN",
    "COLOR_MOUNTAIN_RIM",
    "COLOR_TREE_LEAVES",
    "COLOR_TREE_RIM",
    "COLOR_TREE_TRUNK",
    "COLOR_TARGET",
]
