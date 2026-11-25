import os
import shutil
import pygame
from scrollswarm.env import ScrollSwarmEnv


def main():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    env = ScrollSwarmEnv(num_agents=5, max_cycles=500, render_mode="rgb_array")
    env.reset(seed=42)

    # Step forward a few frames so agents and obstacles are in dynamic flow
    for step_i in range(35):
        actions = {a: 1 for a in env.agents}  # Action.STAY
        # Gentle vertical spreading
        if step_i < 14:
            if "agent_1" in actions:
                actions["agent_1"] = 0  # UP
            if "agent_3" in actions:
                actions["agent_3"] = 2  # DOWN
        env.step(actions)

    env.render()
    screen_surf = env._screen
    pygame.image.save(screen_surf, "assets/screenshot.png")
    shutil.copyfile("assets/screenshot.png", "docs/assets/screenshot.png")
    print("Screenshot saved to assets/screenshot.png and docs/assets/screenshot.png")
    env.close()


if __name__ == "__main__":
    main()
