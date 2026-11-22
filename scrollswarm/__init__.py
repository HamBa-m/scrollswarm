"""
ScrollSwarm: A side-scrolling multi-agent reinforcement learning testbed.
"""

from scrollswarm.env import ScrollSwarmEnv
from scrollswarm.map import (
    ScrollingMap,
    MapConfig,
    RockCrag,
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
from scrollswarm.items import Tree, Target
from scrollswarm.comm import CommunicationGraph, TopologyType

__all__ = [
    "ScrollSwarmEnv",
    "ScrollingMap",
    "MapConfig",
    "RockCrag",
    "Agent",
    "Action",
    "create_agents",
    "AGENT_PALETTE",
    "COLOR_AGENT_DEFAULT",
    "COLOR_AGENT_DEAD",
    "RaySensor",
    "RayHit",
    "Tree",
    "Target",
    "CommunicationGraph",
    "TopologyType",
    "COLOR_LAND",
    "COLOR_MOUNTAIN",
    "COLOR_MOUNTAIN_RIM",
    "COLOR_TREE_LEAVES",
    "COLOR_TREE_RIM",
    "COLOR_TREE_TRUNK",
    "COLOR_TARGET",
]
