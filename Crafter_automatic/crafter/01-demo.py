import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import pygame
from PIL import Image


DEFAULT_MODEL_PATH = "saves/01-basic_reward/ppo_survival_agent.zip"


def load_basic_reward_module():
    module_path = pathlib.Path(__file__).with_name("01-basic_reward.py")
    spec = importlib.util.spec_from_file_location("basic_reward", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def to_screen_surface(image, window):
    if tuple(image.shape[:2]) != tuple(window):
        image = Image.fromarray(image)
        image = image.resize(window, resample=Image.NEAREST)
        image = np.array(image)
    return pygame.surfarray.make_surface(image.transpose((1, 0, 2)))


def load_model(model_path):
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        from stable_baselines3 import PPO

        return PPO.load(model_path), False, "PPO"

    return MaskablePPO.load(model_path), True, "MaskablePPO"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=132)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--window", type=int, nargs=2, default=(900, 1100))
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model_path = pathlib.Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    basic_reward = load_basic_reward_module()
    env = basic_reward.BasicSurvivalRewardEnv(
        episode_steps=args.episode_steps,
        seed=args.seed,
        render_size=args.window,
    )
    model, use_action_masks, algorithm_name = load_model(model_path)

    pygame.init()
    screen = pygame.display.set_mode(args.window)
    pygame.display.set_caption(f"Crafter {algorithm_name} Demo - {model_path}")
    clock = pygame.time.Clock()

    try:
        for episode in range(1, args.episodes + 1):
            obs, _info = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        return

                if use_action_masks:
                    action, _state = model.predict(
                        obs,
                        deterministic=args.deterministic,
                        action_masks=env.action_masks(),
                    )
                else:
                    action, _state = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

                image = env.render()
                screen.blit(to_screen_surface(image, args.window), (0, 0))
                pygame.display.flip()
                clock.tick(args.fps)

            print(
                f"Episode {episode}: return={total_reward:.3f}, "
                f"steps={env.env._step}, inventory={info['inventory']}"
            )
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
