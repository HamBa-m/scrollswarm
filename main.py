"""
ScrollSwarm - 2D Interactive Multi-Agent UAV Swarm Reconnaissance Simulation

Features:
  - 1 to 10 autonomous UAV quadcopters (default 5, configurable via --num-agents / -n)
  - Procedural quadcopter UAV sprites with per-agent team colors
  - 1 px/step scrolling, 30 FPS, atmospheric aerial survey corridor
  - Multi-scale procedural obstacles: ground-standing buildings, overhead towers, floating debris
  - Guaranteed navigable passage invariant (>= 110 px)
  - Targets with stochastic photo acquisition under action uncertainty
  - Interactive selection (TAB or keys 1-9 to switch focused UAV)

Controls:
  - UP Arrow / W   : Climb focused UAV (thrust k = 5 px)
  - DOWN Arrow / S : Dive focused UAV (thrust k = 5 px)
  - TAB            : Cycle active focused UAV
  - 1-9            : Directly select UAV #1 to #9
  - A              : Step all alive UAVs synchronously
  - V              : Toggle 7-ray LiDAR rangefinder visualization
  - C              : Toggle communication graph links
  - R              : Restart simulation
  - SPACE          : Pause / Resume
  - ESC            : Quit
"""

import argparse
import sys
from typing import List
import numpy as np
import pygame

from scrollswarm.map import (
    ScrollingMap,
    MapConfig,
    COLOR_AIRSPACE,
    COLOR_GRID_LINE,
    COLOR_CEILING,
    COLOR_GROUND,
    COLOR_BOUNDARY_RIM,
    COLOR_BUILDING_FILL,
    COLOR_BUILDING_OUTLINE,
    COLOR_BUILDING_SPECULAR,
    COLOR_HAZARD_FILL,
    COLOR_HAZARD_OUTLINE,
    COLOR_TARGET,
    COLOR_TARGET_INSPECTED,
)
from scrollswarm.items import ObstacleType
from scrollswarm.agent import (
    Agent,
    Action,
    COLOR_AGENT_DEAD,
    create_agents,
)
from scrollswarm.sensors import RaySensor
from scrollswarm.comm import CommunicationGraph, TopologyType
from scrollswarm.sprites import get_quadcopter_sprite


def draw_corridor_boundaries(
    surface: pygame.Surface, scrolling_map: ScrollingMap
) -> None:
    """Renders the atmospheric airspace, survey grid lanes, and corridor boundaries."""
    w = scrolling_map.width
    h = scrolling_map.height

    # Atmospheric sky
    surface.fill(COLOR_AIRSPACE)

    # Subtle survey grid lanes
    for offset_lane in (60, 120, 180, 240, 300, 360):
        pygame.draw.line(
            surface, COLOR_GRID_LINE, (0, offset_lane), (w, offset_lane), width=1
        )

    # Corridor ceiling and ground planes
    ceil_y = int(scrolling_map.ceiling_y)
    floor_y = int(scrolling_map.floor_y)
    pygame.draw.rect(surface, COLOR_CEILING, (0, 0, w, ceil_y))
    pygame.draw.rect(surface, COLOR_GROUND, (0, floor_y, w, h - floor_y))
    pygame.draw.line(surface, COLOR_BOUNDARY_RIM, (0, ceil_y), (w, ceil_y), 2)
    pygame.draw.line(surface, COLOR_BOUNDARY_RIM, (0, floor_y), (w, floor_y), 2)


# Backward compatibility aliases
draw_cosmic_boundaries = draw_corridor_boundaries
draw_topography = draw_corridor_boundaries


def draw_obstacles(surface: pygame.Surface, scrolling_map: ScrollingMap) -> None:
    """Renders multi-scale procedural obstacles (buildings, overhead towers, floating debris)."""
    for obs in scrolling_map.obstacles:
        verts = [(int(vx), int(vy)) for vx, vy in obs.vertices]
        if len(verts) >= 3:
            fill_color = (
                COLOR_HAZARD_FILL
                if obs.obstacle_type == ObstacleType.FLOATING
                else COLOR_BUILDING_FILL
            )
            edge_color = (
                COLOR_HAZARD_OUTLINE
                if obs.obstacle_type == ObstacleType.FLOATING
                else COLOR_BUILDING_OUTLINE
            )
            pygame.draw.polygon(surface, fill_color, verts)
            pygame.draw.polygon(surface, edge_color, verts, width=2)
            if obs.obstacle_type != ObstacleType.FLOATING:
                pygame.draw.line(
                    surface, COLOR_BUILDING_SPECULAR, verts[0], verts[2], 1
                )


# Backward compatibility aliases
draw_asteroids = draw_obstacles
draw_trees = draw_obstacles


def draw_targets(surface: pygame.Surface, scrolling_map: ScrollingMap) -> None:
    """Renders all inspection targets with photo radius halos."""
    for target in scrolling_map.targets:
        cx, cy, s = int(target.x), int(target.y), target.size
        if target.collected:
            pygame.draw.circle(surface, COLOR_TARGET_INSPECTED, (cx, cy), s)
            pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 2)
        else:
            # Inspection photo radius guide
            pygame.draw.circle(
                surface, (255, 215, 0), (cx, cy), int(target.inspection_radius), width=1
            )
            pygame.draw.circle(surface, COLOR_TARGET, (cx, cy), s)
            pygame.draw.circle(surface, (255, 240, 150), (cx, cy), s + 3, width=1)
            pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 2)


def draw_sensor_rays(
    surface: pygame.Surface, agent: Agent, scrolling_map: ScrollingMap
) -> None:
    """Renders the 7 forward LiDAR rangefinder beams with high-contrast laser styling."""
    if not agent.alive:
        return
    if agent.sensor is None:
        agent.sensor = RaySensor(max_range=float(scrolling_map.width) / 2.0)

    obs_hits, tgt_hits = agent.sensor.cast_rays(agent.x, agent.y, scrolling_map)

    # Obstacle rangefinder beams
    for hit in obs_hits:
        start_pt = (int(agent.x), int(agent.y))
        end_pt = (int(hit.endpoint[0]), int(hit.endpoint[1]))

        if hit.hit_type != "none":
            pygame.draw.line(surface, (255, 75, 20), start_pt, end_pt, 2)
            pygame.draw.circle(surface, (255, 40, 20), end_pt, 4)
        else:
            pygame.draw.line(surface, (40, 90, 140), start_pt, end_pt, 1)

    # Highlight detected targets along rays
    for hit in tgt_hits:
        if hit.hit_type == "target":
            end_pt = (int(hit.endpoint[0]), int(hit.endpoint[1]))
            pygame.draw.circle(surface, (0, 230, 120), end_pt, 6, width=2)


def draw_comm_links(
    surface: pygame.Surface, comm_graph: CommunicationGraph, agents: List[Agent]
) -> None:
    """Renders active graph communication links between living UAVs."""
    living_agents = [a for a in agents if a.alive]
    edges = comm_graph.get_communication_edges(living_agents)
    for p1, p2 in edges:
        pygame.draw.line(
            surface,
            (70, 140, 230),
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            1,
        )


def draw_agents(
    surface: pygame.Surface,
    agents: List[Agent],
    selected_idx: int,
    font: pygame.font.Font,
) -> None:
    """Renders all UAVs with quadcopter sprites, focus rings, and ID labels."""
    for idx, agent in enumerate(agents):
        color = agent.color if agent.alive else COLOR_AGENT_DEAD
        sprite = get_quadcopter_sprite(color, size=(24, 24))
        sw, sh = sprite.get_size()
        surface.blit(sprite, (int(agent.x - sw // 2), int(agent.y - sh // 2)))

        # Highlight focused UAV
        if idx == selected_idx and agent.alive:
            pygame.draw.circle(
                surface, (255, 255, 255), (int(agent.x), int(agent.y)), 16, width=2
            )
            pygame.draw.circle(
                surface, (30, 144, 255), (int(agent.x), int(agent.y)), 18, width=1
            )
        elif not agent.alive:
            pygame.draw.circle(
                surface, (180, 50, 50), (int(agent.x), int(agent.y)), 13, width=1
            )

        # ID label above UAV
        lbl = font.render(str(agent.agent_id + 1), True, (220, 225, 235))
        surface.blit(lbl, (int(agent.x - 3), int(agent.y - sh // 2 - 12)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ScrollSwarm 2D UAV Swarm Reconnaissance Simulation"
    )
    parser.add_argument(
        "-n",
        "--num-agents",
        type=int,
        default=5,
        help="Number of UAVs in the swarm (1 to 10, default: 5)",
    )
    parser.add_argument(
        "-c",
        "--comm-topology",
        type=str,
        default="chain",
        choices=["chain", "ring", "full", "radius", "capped"],
        help="Graph communication topology (default: chain)",
    )
    args = parser.parse_args()

    num_agents = max(1, min(10, args.num_agents))
    comm_graph = CommunicationGraph(topology=args.comm_topology)

    pygame.init()
    pygame.font.init()

    config = MapConfig(width=800, height=400, fps=30)
    screen = pygame.display.set_mode((config.width, config.height))
    pygame.display.set_caption(
        f"ScrollSwarm - UAV Swarm Reconnaissance ({num_agents} Drones)"
    )
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 13)
    id_font = pygame.font.SysFont("Consolas", 10, bold=True)

    scrolling_map = ScrollingMap(config)
    agents = create_agents(
        num_agents=num_agents,
        map_width=config.width,
        scrolling_map=scrolling_map,
        step_size=5,
    )

    selected_idx = 0
    show_sensors = True
    show_comm = True
    running = True
    paused = False

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_v:
                    show_sensors = not show_sensors
                elif event.key == pygame.K_c:
                    show_comm = not show_comm
                elif event.key == pygame.K_TAB:
                    selected_idx = (selected_idx + 1) % len(agents)
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    idx = event.key - pygame.K_1
                    if idx < len(agents):
                        selected_idx = idx
                elif event.key == pygame.K_r:
                    scrolling_map.reset()
                    agents = create_agents(
                        num_agents=num_agents,
                        map_width=config.width,
                        scrolling_map=scrolling_map,
                        step_size=5,
                    )
                    selected_idx = min(selected_idx, len(agents) - 1)

        # Simulation step
        if not paused:
            keys = pygame.key.get_pressed()
            focused_action = Action.STAY
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                focused_action = Action.UP
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                focused_action = Action.DOWN

            all_move = keys[pygame.K_a]
            for idx, agent in enumerate(agents):
                if not agent.alive:
                    continue
                if idx == selected_idx or all_move:
                    agent.step(focused_action, config.height)

            # 1. Advance map scrolling
            scrolling_map.step()

            # 2. Collision checks
            for agent in agents:
                if agent.alive:
                    agent.check_collision_boundary(scrolling_map)
                    agent.check_collision_obstacles(scrolling_map.obstacles)

            # 3. Stochastic target inspection under action uncertainty
            living_agents = [a for a in agents if a.alive]
            for target in scrolling_map.targets:
                if target.collected:
                    continue
                target.sample_inspection(living_agents, scrolling_map.rng)

        # Rendering
        screen.fill(COLOR_AIRSPACE)
        draw_corridor_boundaries(screen, scrolling_map)
        draw_obstacles(screen, scrolling_map)
        draw_targets(screen, scrolling_map)

        if show_sensors and agents[selected_idx].alive:
            draw_sensor_rays(screen, agents[selected_idx], scrolling_map)

        if show_comm:
            draw_comm_links(screen, comm_graph, agents)

        draw_agents(screen, agents, selected_idx, id_font)

        # HUD Overlay with translucent dark backdrop
        hud_bg = pygame.Surface((config.width - 24, 42), pygame.SRCALPHA)
        pygame.draw.rect(
            hud_bg, (16, 22, 32, 190), (0, 0, config.width - 24, 42), border_radius=5
        )
        screen.blit(hud_bg, (12, 18))

        alive_count = sum(1 for a in agents if a.alive)
        targets_inspected = sum(1 for t in scrolling_map.targets if t.collected)
        focused_agent = agents[selected_idx]
        status_str = "AIRBORNE" if focused_agent.alive else "CRASHED"
        status_color = (80, 230, 130) if focused_agent.alive else (255, 90, 90)

        # Telemetry line 1
        hud_prefix = f"UAVs Active: {alive_count}/{num_agents} | Targets Photographed: {targets_inspected} | Focused: UAV #{focused_agent.agent_id + 1} ("
        txt_prefix = font.render(hud_prefix, True, (240, 245, 250))
        txt_status = font.render(status_str, True, status_color)
        hud_suffix = f", Photos: {focused_agent.score})"
        txt_suffix = font.render(hud_suffix, True, (240, 245, 250))

        hud_line2 = f"[W/S] Climb/Dive | [TAB/1-9] Focus | [A] All | [V] LiDAR | [C] Comm ({args.comm_topology}) | [R] Restart"
        txt_line2 = font.render(hud_line2, True, (175, 185, 200))

        cur_x = 22
        screen.blit(txt_prefix, (cur_x, 22))
        cur_x += txt_prefix.get_width()
        screen.blit(txt_status, (cur_x, 22))
        cur_x += txt_status.get_width()
        screen.blit(txt_suffix, (cur_x, 22))

        screen.blit(txt_line2, (22, 38))

        if paused:
            pause_surface = font.render("-- PAUSED --", True, (220, 50, 50))
            screen.blit(pause_surface, (config.width // 2 - 40, 68))

        pygame.display.flip()
        clock.tick(config.fps)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
