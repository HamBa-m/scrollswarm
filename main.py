"""
ScrollSwarm - 2D Interactive Multi-Agent Saturn Ring Surfing Simulation

Features:
  - 1 to 10 rocket probes (default 5, configurable via --num-agents / -n)
  - SVG vector rocket probe sprites with per-agent team colors
  - 1 px/step scrolling, 30 FPS, cosmic void, radiation sheath boundaries
  - Dynamic polygon asteroids executing Keplerian drift and Hill's epicyclic vertical oscillations
  - Collision causes instant structural termination
  - Interactive selection (TAB or keys 1-9 to switch focused probe)

Controls:
  - UP Arrow / W   : Move focused probe up (thrust k = 5 px)
  - DOWN Arrow / S : Move focused probe down (thrust k = 5 px)
  - TAB            : Cycle active focused probe
  - 1-9            : Directly select Probe #1 to #9
  - A              : Step all alive probes synchronously
  - V              : Toggle 7-ray LiDAR rangefinder visualization
  - C              : Toggle communication graph links
  - R              : Restart simulation
  - SPACE          : Pause / Resume
  - ESC            : Quit
"""

import argparse
import sys
from typing import List
import pygame

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
    COLOR_TARGET,
)
from scrollswarm.agent import (
    Agent,
    Action,
    COLOR_AGENT_DEAD,
    create_agents,
)
from scrollswarm.sensors import RaySensor
from scrollswarm.comm import CommunicationGraph, TopologyType
from scrollswarm.sprites import get_rocket_sprite


def draw_cosmic_boundaries(surface: pygame.Surface, scrolling_map: ScrollingMap) -> None:
    """Renders the Saturnian cosmic void, ring dust lanes, and sheath boundary planes."""
    w = scrolling_map.width
    h = scrolling_map.height

    # Deep space cosmic void
    surface.fill(COLOR_SPACE_VOID)

    # Saturnian champagne ring dust streams
    for offset_lane in (75, 140, 205, 270, 335):
        pygame.draw.line(surface, (18, 24, 38), (0, offset_lane), (w, offset_lane), width=1)

    # Sheath boundary planes (flat limits with luminous rims)
    ceil_y = int(scrolling_map.ceiling_y)
    floor_y = int(scrolling_map.floor_y)
    pygame.draw.rect(surface, COLOR_RING_SHEATH, (0, 0, w, ceil_y))
    pygame.draw.rect(surface, COLOR_RING_SHEATH, (0, floor_y, w, h - floor_y))
    pygame.draw.line(surface, COLOR_RING_RIM, (0, ceil_y), (w, ceil_y), 2)
    pygame.draw.line(surface, COLOR_RING_RIM, (0, floor_y), (w, floor_y), 2)


# Backward compatibility alias
draw_topography = draw_cosmic_boundaries


def draw_asteroids(surface: pygame.Surface, scrolling_map: ScrollingMap) -> None:
    """Renders dynamic polygon asteroids and icy moonlets with rock-ice facets."""
    for ast in getattr(scrolling_map, "asteroids", []):
        verts = [(int(vx), int(vy)) for vx, vy in ast.vertices]
        if len(verts) >= 3:
            pygame.draw.polygon(surface, COLOR_ASTEROID_FILL, verts)
            pygame.draw.polygon(surface, COLOR_ASTEROID_OUTLINE, verts, width=2)
            # Frosted ice specular facet line
            pygame.draw.line(surface, COLOR_ASTEROID_SPECULAR, verts[0], verts[2], 1)


# Backward compatibility alias
draw_trees = draw_asteroids


def draw_targets(surface: pygame.Surface, scrolling_map: ScrollingMap) -> None:
    """Renders all uncollected collection targets."""
    for target in scrolling_map.targets:
        if target.collected:
            continue
        cx, cy, s = int(target.x), int(target.y), target.size
        # Golden target with outer glow and white core
        pygame.draw.circle(surface, COLOR_TARGET, (cx, cy), s)
        pygame.draw.circle(surface, (255, 235, 120), (cx, cy), s + 3, width=1)
        pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 2)


def draw_sensor_rays(surface: pygame.Surface, agent: Agent, scrolling_map: ScrollingMap) -> None:
    """Renders the 7 forward sensor beams with high-contrast laser styling."""
    if not agent.alive:
        return
    if agent.sensor is None:
        agent.sensor = RaySensor(max_range=float(scrolling_map.width) / 2.0)

    obs_hits, tgt_hits = agent.sensor.cast_rays(agent.x, agent.y, scrolling_map)

    # Draw obstacle detection beams
    for hit in obs_hits:
        start_pt = (int(agent.x), int(agent.y))
        end_pt = (int(hit.endpoint[0]), int(hit.endpoint[1]))

        if hit.hit_type != "none":
            # High-alert orange-red beam striking an asteroid or boundary
            pygame.draw.line(surface, (255, 70, 0), start_pt, end_pt, 2)
            pygame.draw.circle(surface, (255, 40, 20), end_pt, 4)
        else:
            # Subtle cyan beam when path is clear
            pygame.draw.line(surface, (30, 80, 130), start_pt, end_pt, 1)

    # Highlight detected targets along rays
    for hit in tgt_hits:
        if hit.hit_type == "target":
            end_pt = (int(hit.endpoint[0]), int(hit.endpoint[1]))
            # Vivid emerald ring
            pygame.draw.circle(surface, (0, 220, 100), end_pt, 6, width=2)


def draw_comm_links(surface: pygame.Surface, comm_graph: CommunicationGraph, agents: List[Agent]) -> None:
    """Renders active graph communication links between living agents."""
    edges = comm_graph.get_communication_edges(agents)
    for p1, p2 in edges:
        # Faint cyan/blue communication data link
        pygame.draw.line(surface, (70, 140, 230), (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 1)


def draw_agents(surface: pygame.Surface, agents: List[Agent], selected_idx: int, font: pygame.font.Font) -> None:
    """Renders all agents with rocket probe sprites, focus ring, and ID tags."""
    for idx, agent in enumerate(agents):
        color = agent.color if agent.alive else COLOR_AGENT_DEAD
        sprite = get_rocket_sprite(color, size=(24, 24))
        sw, sh = sprite.get_size()
        surface.blit(sprite, (int(agent.x - sw // 2), int(agent.y - sh // 2)))

        # Highlight currently selected probe with distinct dual focus rings
        if idx == selected_idx and agent.alive:
            pygame.draw.circle(surface, (255, 255, 255), (int(agent.x), int(agent.y)), 16, width=2)
            pygame.draw.circle(surface, (30, 144, 255), (int(agent.x), int(agent.y)), 18, width=1)
        elif not agent.alive:
            # Subtle red indicator for terminated probes
            pygame.draw.circle(surface, (180, 50, 50), (int(agent.x), int(agent.y)), 13, width=1)

        # ID label above probe
        lbl = font.render(str(agent.agent_id), True, (220, 225, 235))
        surface.blit(lbl, (int(agent.x - 3), int(agent.y - sh // 2 - 12)))


def main() -> None:
    parser = argparse.ArgumentParser(description="ScrollSwarm 2D Simulation (Saturn Ring Surfing)")
    parser.add_argument(
        "-n", "--num-agents",
        type=int,
        default=5,
        help="Number of agents in the swarm (1 to 10, default: 5)"
    )
    parser.add_argument(
        "-c", "--comm-topology",
        type=str,
        default="chain",
        choices=["chain", "ring", "full", "radius", "capped"],
        help="Graph communication topology (default: chain)"
    )
    args = parser.parse_args()

    num_agents = max(1, min(10, args.num_agents))
    comm_graph = CommunicationGraph(topology=args.comm_topology)

    pygame.init()
    pygame.font.init()

    config = MapConfig(width=800, height=400, fps=30)
    screen = pygame.display.set_mode((config.width, config.height))
    pygame.display.set_caption(f"ScrollSwarm - Saturn Ring Surfing ({num_agents} Probes)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 13)
    id_font = pygame.font.SysFont("Consolas", 10, bold=True)

    scrolling_map = ScrollingMap(config)
    agents = create_agents(num_agents=num_agents, map_width=config.width, scrolling_map=scrolling_map, step_size=5)

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
                    # Cycle through agents
                    selected_idx = (selected_idx + 1) % len(agents)
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    idx = event.key - pygame.K_1
                    if idx < len(agents):
                        selected_idx = idx
                elif event.key == pygame.K_r:
                    # Restart simulation
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

            # Move agents: focused agent gets keyboard command; others stay (or sync move if 'A' pressed)
            all_move = keys[pygame.K_a]
            for idx, agent in enumerate(agents):
                if not agent.alive:
                    continue
                if idx == selected_idx or all_move:
                    agent.step(focused_action, config.height)

            # 1. Advance map scrolling by 1 pixel
            scrolling_map.step()

            # 2. Check collisions & target collection for each agent
            for agent in agents:
                if agent.alive:
                    agent.check_collision_mountain(scrolling_map)
                    agent.check_collision_obstacles(scrolling_map.asteroids)
                    agent.check_target_collection(scrolling_map.targets)

        # Rendering
        screen.fill(COLOR_SPACE_VOID)
        draw_cosmic_boundaries(screen, scrolling_map)
        draw_asteroids(screen, scrolling_map)
        draw_targets(screen, scrolling_map)

        if show_sensors and agents[selected_idx].alive:
            draw_sensor_rays(screen, agents[selected_idx], scrolling_map)

        if show_comm:
            draw_comm_links(screen, comm_graph, agents)

        draw_agents(screen, agents, selected_idx, id_font)

        # HUD Overlay with translucent dark backdrop for maximum legibility
        hud_bg = pygame.Surface((config.width - 24, 42), pygame.SRCALPHA)
        pygame.draw.rect(hud_bg, (16, 22, 32, 190), (0, 0, config.width - 24, 42), border_radius=5)
        screen.blit(hud_bg, (12, 18))

        alive_count = sum(1 for a in agents if a.alive)
        team_score = sum(a.score for a in agents)
        focused_agent = agents[selected_idx]
        status_str = "ALIVE" if focused_agent.alive else "TERMINATED"
        status_color = (80, 230, 130) if focused_agent.alive else (255, 90, 90)

        # Line 1: Swarm & Focused Agent Telemetry
        hud_prefix = f"Probes Active: {alive_count}/{num_agents} | Team Score: {team_score} | Focused: Probe #{focused_agent.agent_id} ("
        txt_prefix = font.render(hud_prefix, True, (240, 245, 250))
        txt_status = font.render(status_str, True, status_color)
        hud_suffix = f", Score: {focused_agent.score})"
        txt_suffix = font.render(hud_suffix, True, (240, 245, 250))

        hud_line2 = (
            f"[W/S] Thrust | [TAB/1-9] Focus | [A] All | [V] LiDAR | [C] Comm ({args.comm_topology}) | [R] Restart"
        )
        txt_line2 = font.render(hud_line2, True, (175, 185, 200))

        # Blit telemetry with highlighted status
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
