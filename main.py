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

from objects.map import (
    ScrollingMap,
    MapConfig,
    COLOR_LAND,
    COLOR_MOUNTAIN,
    COLOR_TREE_LEAVES,
    COLOR_TREE_TRUNK,
    COLOR_TARGET,
)
from objects.agent import (
    Agent,
    Action,
    COLOR_AGENT_DEAD,
    create_agents,
)
from objects.sensors import RaySensor


def draw_topography(surface: pygame.Surface, scrolling_map: ScrollingMap) -> None:
    """Renders the top and bottom mountain boundary polygons."""
    w = scrolling_map.width
    h = scrolling_map.height

    top_points = [(0, 0)]
    for x in range(w):
        top_points.append((x, int(scrolling_map.top_heights[x])))
    top_points.append((w - 1, 0))
    pygame.draw.polygon(surface, COLOR_MOUNTAIN, top_points)

    bot_points = [(0, h)]
    for x in range(w):
        bot_points.append((x, int(h - scrolling_map.bot_heights[x])))
    bot_points.append((w - 1, h))
    pygame.draw.polygon(surface, COLOR_MOUNTAIN, bot_points)


def draw_trees(surface: pygame.Surface, scrolling_map: ScrollingMap) -> None:
    """Renders all active trees with dark green canopy and brown trunk."""
    for tree in scrolling_map.trees:
        pos = (int(tree.x), int(tree.y))
        pygame.draw.circle(surface, COLOR_TREE_LEAVES, pos, tree.outer_radius)
        pygame.draw.circle(surface, COLOR_TREE_TRUNK, pos, tree.inner_radius)


def draw_targets(surface: pygame.Surface, scrolling_map: ScrollingMap) -> None:
    """Renders all uncollected targets as red X marks."""
    for target in scrolling_map.targets:
        if target.collected:
            continue
        cx, cy, s = int(target.x), int(target.y), target.size
        pygame.draw.line(surface, COLOR_TARGET, (cx - s, cy - s), (cx + s, cy + s), width=2)
        pygame.draw.line(surface, COLOR_TARGET, (cx - s, cy + s), (cx + s, cy - s), width=2)


def draw_sensor_rays(surface: pygame.Surface, agent: Agent, scrolling_map: ScrollingMap) -> None:
    """Renders the 7 forward sensor beams for the agent."""
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
            # Orange/red beam when striking an obstacle
            pygame.draw.line(surface, (230, 140, 50), start_pt, end_pt, 1)
            pygame.draw.circle(surface, (220, 40, 40), end_pt, 3)
        else:
            # Faint cyan beam when path is clear
            pygame.draw.line(surface, (160, 210, 240), start_pt, end_pt, 1)

    # Highlight detected targets along rays
    for hit in tgt_hits:
        if hit.hit_type == "target":
            end_pt = (int(hit.endpoint[0]), int(hit.endpoint[1]))
            pygame.draw.circle(surface, (40, 200, 40), end_pt, 5, width=1)


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
    args = parser.parse_args()

    num_agents = max(1, min(10, args.num_agents))

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

        draw_agents(screen, agents, selected_idx, id_font)

        # HUD Overlay
        alive_count = sum(1 for a in agents if a.alive)
        team_score = sum(a.score for a in agents)
        focused_agent = agents[selected_idx]
        status_str = "ALIVE" if focused_agent.alive else "DEAD"

        hud_line1 = (
            f"Agents Alive: {alive_count}/{num_agents} | Team Score: {team_score} | "
            f"Focused: Agent #{focused_agent.agent_id} ({status_str}, Score: {focused_agent.score})"
        )
        hud_line2 = (
            "[W/UP, S/DOWN] Move Focused | [TAB / 1-9] Select | [A] Move All | [V] Rays | [R] Reset | [SPACE] Pause"
        )

        txt_line1 = font.render(hud_line1, True, (20, 20, 20))
        txt_line2 = font.render(hud_line2, True, (50, 50, 50))
        screen.blit(txt_line1, (12, 8))
        screen.blit(txt_line2, (12, 24))

        if paused:
            pause_surface = font.render("-- PAUSED --", True, (200, 0, 0))
            screen.blit(pause_surface, (config.width // 2 - 40, 8))

        pygame.display.flip()
        clock.tick(config.fps)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
