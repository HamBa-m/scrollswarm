"""
ScrollSwarm: A side-scrolling multi-agent reinforcement learning testbed.
"""

from objects.env import ScrollSwarmEnv
from objects.map import ScrollingMap, MapConfig
from objects.agent import Agent, Action, create_agents
from objects.sensors import RaySensor, RayHit
from objects.items import Tree, Target

__all__ = [
    "ScrollSwarmEnv",
    "ScrollingMap",
    "MapConfig",
    "Agent",
    "Action",
    "create_agents",
    "RaySensor",
    "RayHit",
    "Tree",
    "Target",
]
