"""
ScrollSwarm - 2D Interactive Multi-Agent Side-Scroller

Features:
  - 1 to 10 agents (default 5, configurable via --num-agents / -n)
  - Triangle agents pointing RIGHT
  - 1 px/step scrolling, 30 FPS, light green plain land, boundary mountain ridges
  - Collision causes instant termination
  - Interactive selection (TAB or keys 1-9 to switch focused agent)

Controls:
  - UP Arrow / W   : Move focused agent up (k = 5 px)
  - DOWN Arrow / S : Move focused agent down (k = 5 px)
  - TAB            : Cycle active focused agent
  - 1-9            : Directly select Agent #1 to #9
  - A              : Step all alive agents synchronously
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
    COLOR_AGENT_DEAD,
    create_agents,
)
from scrollswarm.sensors import RaySensor
from scrollswarm.comm import CommunicationGraph, TopologyType


def draw_topography(surface: pygame.Surface, scrolling_map: ScrollingMap) -> None:
    """Renders the top and bottom mountain boundary polygons with dark rim contours."""
    w = scrolling_map.width
    h = scrolling_map.height

    # Top mountain polygon: (0,0) -> (x, top_heights[x]) -> (w-1, 0)
    top_points = [(0, 0)]
    for x in range(w):
        top_points.append((x, int(scrolling_map.top_heights[x])))
    top_points.append((w - 1, 0))
    pygame.draw.polygon(surface, COLOR_MOUNTAIN, top_points)

    # Top rim line
    top_rim = [(x, int(scrolling_map.top_heights[x])) for x in range(w)]
    pygame.draw.lines(surface, COLOR_MOUNTAIN_RIM, False, top_rim, width=2)

    # Bottom mountain polygon: (0, h) -> (x, h - bot_heights[x]) -> (w-1, h)
    bot_points = [(0, h)]
    for x in range(w):
        bot_points.append((x, int(h - scrolling_map.bot_heights[x])))
    bot_points.append((w - 1, h))
    pygame.draw.polygon(surface, COLOR_MOUNTAIN, bot_points)

    # Bottom rim line
    bot_rim = [(x, int(h - scrolling_map.bot_heights[x])) for x in range(w)]
    pygame.draw.lines(surface, COLOR_MOUNTAIN_RIM, False, bot_rim, width=2)


def draw_trees(surface: pygame.Surface, scrolling_map: ScrollingMap) -> None:
    """Renders all active trees with dark outline, vibrant canopy, and cedar trunk."""
    for tree in scrolling_map.trees:
        pos = (int(tree.x), int(tree.y))
        # Dark canopy rim for sharp contrast against land
        pygame.draw.circle(surface, COLOR_TREE_RIM, pos, tree.outer_radius + 1)
        # Outer vibrant canopy
        pygame.draw.circle(surface, COLOR_TREE_LEAVES, pos, tree.outer_radius)
        # Inner trunk
        pygame.draw.circle(surface, COLOR_TREE_TRUNK, pos, tree.inner_radius)


def draw_targets(surface: pygame.Surface, scrolling_map: ScrollingMap) -> None:
    """Renders all uncollected targets as bold scarlet X marks with center dot."""
    for target in scrolling_map.targets:
        if target.collected:
            continue
        cx, cy, s = int(target.x), int(target.y), target.size
        # Bold 3px scarlet cross
        pygame.draw.line(surface, COLOR_TARGET, (cx - s, cy - s), (cx + s, cy + s), width=3)
        pygame.draw.line(surface, COLOR_TARGET, (cx - s, cy + s), (cx + s, cy - s), width=3)
        # Center anchor dot
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
            # High-alert orange-red beam striking an obstacle
            pygame.draw.line(surface, (255, 70, 0), start_pt, end_pt, 2)
            pygame.draw.circle(surface, (255, 40, 20), end_pt, 4)
        else:
            # High-contrast deep navy beam when path is clear
            pygame.draw.line(surface, (50, 100, 160), start_pt, end_pt, 1)

    # Highlight detected targets along rays
    for hit in tgt_hits:
        if hit.hit_type == "target":
            end_pt = (int(hit.endpoint[0]), int(hit.endpoint[1]))
            # Vivid emerald ring
            pygame.draw.circle(surface, (0, 180, 60), end_pt, 6, width=2)


def draw_comm_links(surface: pygame.Surface, comm_graph: CommunicationGraph, agents: List[Agent]) -> None:
    """Renders active graph communication links between living agents."""
    edges = comm_graph.get_communication_edges(agents)
    for p1, p2 in edges:
        # Faint cyan/blue communication data link
        pygame.draw.line(surface, (70, 140, 230), (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 1)


def draw_agents(surface: pygame.Surface, agents: List[Agent], selected_idx: int, font: pygame.font.Font) -> None:
    """Renders all agents (pointing RIGHT) with outline and ID tags."""
    for idx, agent in enumerate(agents):
        pts = [(int(vx), int(vy)) for vx, vy in agent.vertices]
        color = agent.color if agent.alive else COLOR_AGENT_DEAD
        pygame.draw.polygon(surface, color, pts)

        # Highlight currently selected agent with a white outline
        outline_color = (255, 255, 255) if idx == selected_idx else (40, 40, 40)
        outline_width = 2 if idx == selected_idx else 1
        pygame.draw.polygon(surface, outline_color, pts, width=outline_width)

        # Small ID label above agent
        lbl = font.render(str(agent.agent_id), True, (20, 20, 20))
        surface.blit(lbl, (int(agent.x - 3), int(agent.y - agent.height - 4)))


def main() -> None:
    parser = argparse.ArgumentParser(description="ScrollSwarm 2D Simulation")
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
    pygame.display.set_caption(f"ScrollSwarm - Swarm Size: {num_agents}")
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
                    agent.check_collision_trees(scrolling_map.trees)
                    agent.check_target_collection(scrolling_map.targets)

        # Rendering
        screen.fill(COLOR_LAND)
        draw_topography(screen, scrolling_map)
        draw_trees(screen, scrolling_map)
        draw_targets(screen, scrolling_map)

        if show_sensors and agents[selected_idx].alive:
            draw_sensor_rays(screen, agents[selected_idx], scrolling_map)

        if show_comm:
            draw_comm_links(screen, comm_graph, agents)

        draw_agents(screen, agents, selected_idx, id_font)

        # HUD Overlay with translucent dark backdrop for maximum legibility
        hud_bg = pygame.Surface((config.width - 24, 46), pygame.SRCALPHA)
        pygame.draw.rect(hud_bg, (22, 28, 36, 215), (0, 0, config.width - 24, 46), border_radius=6)
        screen.blit(hud_bg, (12, 6))

        alive_count = sum(1 for a in agents if a.alive)
        team_score = sum(a.score for a in agents)
        focused_agent = agents[selected_idx]
        status_str = "ALIVE" if focused_agent.alive else "TERMINATED"
        status_color = (80, 230, 130) if focused_agent.alive else (255, 90, 90)

        # Line 1: Swarm & Focused Agent Telemetry
        hud_prefix = f"Agents Alive: {alive_count}/{num_agents} | Team Score: {team_score} | Focused: Agent #{focused_agent.agent_id} ("
        txt_prefix = font.render(hud_prefix, True, (240, 245, 250))
        txt_status = font.render(status_str, True, status_color)
        hud_suffix = f", Score: {focused_agent.score})"
        txt_suffix = font.render(hud_suffix, True, (240, 245, 250))

        hud_line2 = (
            f"[W/S] Move | [TAB/1-9] Select | [A] All | [V] Rays | [C] Comm ({args.comm_topology}) | [R] Reset"
        )
        txt_line2 = font.render(hud_line2, True, (175, 185, 200))

        # Blit telemetry with highlighted status
        cur_x = 22
        screen.blit(txt_prefix, (cur_x, 10))
        cur_x += txt_prefix.get_width()
        screen.blit(txt_status, (cur_x, 10))
        cur_x += txt_status.get_width()
        screen.blit(txt_suffix, (cur_x, 10))

        screen.blit(txt_line2, (22, 28))

        if paused:
            pause_surface = font.render("-- PAUSED --", True, (200, 0, 0))
            screen.blit(pause_surface, (config.width // 2 - 40, 8))

        pygame.display.flip()
        clock.tick(config.fps)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
