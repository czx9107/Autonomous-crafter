import argparse
import datetime
import math
import pathlib
import shutil
import sys
import zipfile
from dataclasses import dataclass

import imageio.v2 as imageio
import numpy as np
import torch as th
from torch import nn

import crafter
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


SURVIVAL_KEYS = ("health", "food", "drink", "energy")
MOVE_ACTION_NAMES = ("move_left", "move_right", "move_up", "move_down")
INVENTORY_OBS_KEYS = tuple(crafter.constants.items.keys())
POSITION_OBS_KEYS = ("x", "y")
STATS_OBS_SIZE = len(INVENTORY_OBS_KEYS) + len(POSITION_OBS_KEYS)
REWARD_COMPONENT_GROUPS = (
    "terminal",
    "low_status",
    "alive",
    "potential",
)
MAP_CHANNELS = 1 + len(crafter.constants.materials) + len(crafter.constants.objects)
UNKNOWN_MAP_TOKEN_ID = MAP_CHANNELS
MAP_EMBEDDING_DIM = 8
OBS_HISTORY_LENGTH = 4
NEW_MAP_HISTORY_LENGTH = 1
NEW_MAP_SIZE = 25
MEMORY_MAP_SIZE = 32
MEMORY_MAP_CHANNELS = 6
DEFAULT_OUTPUT_DIR = "saves/01-basic_reward"
MODEL_BASENAME = "ppo_survival_agent"
PPO_ALGORITHM_NAME = "MaskablePPO"
POLICY_NET_ARCH = {"pi": [512, 512], "vf": [512, 512]}
MAP_FEATURES_DIM = 128
MEMORY_FEATURES_DIM = 64
STATS_FEATURES_DIM = 64
PPO_ENT_COEF = 0.00
PPO_CLIP_RANGE = 0.2
PPO_LEARNING_RATE = 1.5e-4
PPO_ENABLE_LR_DECAY = False
PPO_LR_DECAY_START_FRACTION = 0.5
PPO_LR_DECAY_FINAL_FRACTION = 1.0 / 3.0
PPO_GAMMA = 0.997
PPO_N_EPOCHS = 10
PPO_BATCH_SIZE = 1024
MONSTER_SPAWN_RATE = 0.0
PPO_HYPERPARAM_FLAGS = {
    "--seed": "seed",
    "--n-steps": "n_steps",
    "--batch-size": "batch_size",
    "--learning-rate": "learning_rate",
}

MATERIAL_ID_BY_NAME = {
    name: index + 1 for index, name in enumerate(crafter.constants.materials)
}
OBJECT_ID_BY_NAME = {
    name: 1 + len(crafter.constants.materials) + index
    for index, name in enumerate(crafter.constants.objects)
}
WALKABLE_TOKEN_IDS = {
    MATERIAL_ID_BY_NAME[name]
    for name in crafter.constants.walkable
    if name in MATERIAL_ID_BY_NAME
}
WATER_MEMORY_TOKEN_IDS = {
    MATERIAL_ID_BY_NAME[name]
    for name in ("water", "grass_water", "sand_water")
    if name in MATERIAL_ID_BY_NAME
}
FOOD_MEMORY_TOKEN_IDS = {
    token_id
    for token_id in (
        MATERIAL_ID_BY_NAME.get("apple_tree"),
        OBJECT_ID_BY_NAME.get("cow"),
    )
    if token_id is not None
}
HOSTILE_MEMORY_TOKEN_IDS = {
    token_id
    for token_id in (
        OBJECT_ID_BY_NAME.get("zombie"),
        OBJECT_ID_BY_NAME.get("skeleton"),
    )
    if token_id is not None
}


def get_map_num_embeddings(observation_space):
    return int(np.max(observation_space.spaces["map"].high)) + 1


def get_expected_feature_dim(observation_space):
    features_dim = MAP_FEATURES_DIM + STATS_FEATURES_DIM
    if "memory" in observation_space.spaces:
        features_dim += MEMORY_FEATURES_DIM
    return features_dim


def get_expected_map_cnn_in_channels(observation_space):
    return MAP_EMBEDDING_DIM * observation_space.spaces["map"].shape[0]


def get_expected_map_head_in_features(observation_space):
    map_shape = observation_space.spaces["map"].shape
    map_history = map_shape[0]
    map_cnn = nn.Sequential(
        nn.Conv2d(MAP_EMBEDDING_DIM * map_history, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
    )
    with th.no_grad():
        sample = th.zeros(
            1,
            MAP_EMBEDDING_DIM * map_history,
            *map_shape[1:],
            dtype=th.float32,
        )
        return int(map_cnn(sample).shape[1])


def get_expected_memory_cnn_in_channels(observation_space):
    if "memory" not in observation_space.spaces:
        return None
    return int(observation_space.spaces["memory"].shape[0])


@dataclass(frozen=True)
class SurvivalRewardConfig:
    """Weights for the editable survival-only reward."""

    potential_gamma: float = PPO_GAMMA

    rate = 1.5
    health_delta: float = 2.0 * rate
    food_delta: float = 1.0 * rate
    drink_delta: float = 1.0 * rate
    energy_delta: float = 0.0
    food_drink_balance_delta: float = 0.0

    alive_bonus: float = 0.04
    status_bonus: float = 0.0
    action_cost: float = 0.00
    death_penalty: float = 30.0
    max_steps_bonus: float = 40.0
    invalid_action_penalty: float = 0.0

    enable_low_status_penalty: int = 1
    food_low_status_penalty: float = 0.1
    drink_low_status_penalty: float = 0.1
    health_low_status_penalty: float = 0.1
    energy_low_status_penalty: float = 0.0
    low_status_penalty: float = 0.25

    food_low_status_threshold: float = 0.4
    drink_low_status_threshold: float = 0.4
    health_low_status_threshold: float = 0.6
    energy_low_status_threshold: float = 0.3
    low_status_threshold: float = 0.5

    low_status_power: float = 2.0


class CrafterMapStatsExtractor(BaseFeaturesExtractor):
    """Encode the symbolic map with CNN and scalar survival/inventory with MLP."""

    def __init__(self, observation_space):
        self.has_memory = "memory" in observation_space.spaces
        features_dim = MAP_FEATURES_DIM + STATS_FEATURES_DIM
        if self.has_memory:
            features_dim += MEMORY_FEATURES_DIM
        super().__init__(observation_space, features_dim=features_dim)
        map_shape = observation_space.spaces["map"].shape
        map_history = map_shape[0]
        map_embeddings = get_map_num_embeddings(observation_space)
        stats_size = int(np.prod(observation_space.spaces["stats"].shape))

        self.map_embedding = nn.Embedding(map_embeddings, MAP_EMBEDDING_DIM)
        self.map_cnn = nn.Sequential(
            nn.Conv2d(MAP_EMBEDDING_DIM * map_history, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.zeros(
                1,
                MAP_EMBEDDING_DIM * map_history,
                *map_shape[1:],
                dtype=th.float32,
            )
            cnn_flattened = self.map_cnn(sample).shape[1]
        self.map_head = nn.Sequential(
            nn.Linear(cnn_flattened, MAP_FEATURES_DIM),
            nn.ReLU(),
        )
        if self.has_memory:
            memory_shape = observation_space.spaces["memory"].shape
            self.memory_cnn = nn.Sequential(
                nn.Conv2d(memory_shape[0], 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with th.no_grad():
                memory_sample = th.zeros(1, *memory_shape, dtype=th.float32)
                memory_flattened = self.memory_cnn(memory_sample).shape[1]
            self.memory_head = nn.Sequential(
                nn.Linear(memory_flattened, MEMORY_FEATURES_DIM),
                nn.ReLU(),
            )
        else:
            self.memory_cnn = None
            self.memory_head = None
        self.stats_mlp = nn.Sequential(
            nn.Linear(stats_size, STATS_FEATURES_DIM),
            nn.ReLU(),
            nn.Linear(STATS_FEATURES_DIM, STATS_FEATURES_DIM),
            nn.ReLU(),
        )

    def forward(self, observations):
        map_tokens = observations["map"].long()
        if map_tokens.dim() == 3:
            map_tokens = map_tokens.unsqueeze(1)
        batch_size, map_history, height, width = map_tokens.shape
        embedded_map = self.map_embedding(map_tokens)
        embedded_map = embedded_map.permute(0, 1, 4, 2, 3).reshape(
            batch_size,
            map_history * MAP_EMBEDDING_DIM,
            height,
            width,
        )
        map_features = self.map_head(self.map_cnn(embedded_map))
        stats = observations["stats"].float().view(observations["stats"].shape[0], -1)
        stats_features = self.stats_mlp(stats)
        features = [map_features]
        if self.has_memory:
            memory_features = self.memory_head(self.memory_cnn(observations["memory"].float()))
            features.append(memory_features)
        features.append(stats_features)
        return th.cat(features, dim=1)


def get_survival_reward(prev_inventory, curr_inventory, done, truncated=False, config=None):
    """Return reward using only health, food, drink, and energy.

    This is intentionally exposed at module level so later reward experiments can
    edit one function without touching the PPO training loop.
    """
    config = config or SurvivalRewardConfig()
    weights = {
        "health": config.health_delta,
        "food": config.food_delta,
        "drink": config.drink_delta,
        "energy": config.energy_delta,
    }
    low_status_thresholds = {
        "health": config.health_low_status_threshold,
        "food": config.food_low_status_threshold,
        "drink": config.drink_low_status_threshold,
        "energy": config.energy_low_status_threshold,
    }

    reward = 0.0
    components = {}
    prev_potential = get_status_potential(prev_inventory, config)
    curr_potential = get_status_potential(curr_inventory, config)
    for key in SURVIVAL_KEYS:
        max_value = float(crafter.constants.items[key]["max"])
        prev = float(prev_inventory[key]) / max_value
        curr = float(curr_inventory[key]) / max_value
        components[f"{key}_potential_prev"] = float(weights[key] * prev)
        components[f"{key}_potential_curr"] = float(weights[key] * curr)
        components[f"{key}_level"] = curr
    prev_balance_level = get_food_drink_balance_level(prev_inventory)
    curr_balance_level = get_food_drink_balance_level(curr_inventory)
    components["food_drink_balance_level_prev"] = float(prev_balance_level)
    components["food_drink_balance_level"] = float(curr_balance_level)
    components["food_drink_balance_potential_prev"] = float(
        config.food_drink_balance_delta * prev_balance_level
    )
    components["food_drink_balance_potential_curr"] = float(
        config.food_drink_balance_delta * curr_balance_level
    )
    potential_decay = (config.potential_gamma - 1.0) * curr_potential
    potential_delta = curr_potential - prev_potential
    potential_reward = config.potential_gamma * curr_potential - prev_potential
    components["status_potential_prev"] = float(prev_potential)
    components["status_potential_curr"] = float(curr_potential)
    components["status_potential_decay"] = float(potential_decay)
    components["status_potential_delta"] = float(potential_delta)
    components["status_potential_reward"] = float(potential_reward)
    reward += potential_reward

    low_status_penalty = 0.0
    low_status_penalty_weights = {
        "health": config.health_low_status_penalty,
        "food": config.food_low_status_penalty,
        "drink": config.drink_low_status_penalty,
        "energy": config.energy_low_status_penalty,
    }
    components["enable_low_status_penalty"] = int(config.enable_low_status_penalty)
    if config.enable_low_status_penalty and config.low_status_power > 0:
        for key in SURVIVAL_KEYS:
            threshold = float(low_status_thresholds.get(key, config.low_status_threshold))
            penalty_weight = float(low_status_penalty_weights.get(key, config.low_status_penalty))
            if threshold <= 0 or penalty_weight <= 0:
                components[f"{key}_low_penalty"] = 0.0
                continue
            level = components[f"{key}_level"]
            pressure = max(0.0, (threshold - level) / threshold)
            key_penalty = -penalty_weight * (pressure ** config.low_status_power)
            components[f"{key}_low_penalty"] = float(key_penalty)
            low_status_penalty += key_penalty
    components["low_status_penalty"] = float(low_status_penalty)
    reward += low_status_penalty

    alive_bonus = config.alive_bonus if not done or truncated else 0.0
    components["alive_bonus"] = float(alive_bonus)
    reward += alive_bonus

    status_bonus = config.status_bonus * np.mean(
        [components[f"{key}_level"] for key in SURVIVAL_KEYS]
    )
    components["status_bonus"] = float(status_bonus)
    reward += status_bonus

    action_cost = -config.action_cost
    components["action_cost"] = action_cost
    reward += action_cost

    death_penalty = 0.0
    if done and curr_inventory["health"] <= 0:
        death_penalty = -config.death_penalty
        reward += death_penalty
    components["death_penalty"] = death_penalty

    max_steps_bonus = 0.0
    if truncated and curr_inventory["health"] > 0:
        max_steps_bonus = config.max_steps_bonus
        reward += max_steps_bonus
    components["max_steps_bonus"] = max_steps_bonus
    components["survival_reward"] = float(reward)
    return float(reward), components


def get_status_potential(inventory, config=None):
    """Return a compact potential for survival status quality."""
    config = config or SurvivalRewardConfig()
    weights = {
        "health": config.health_delta,
        "food": config.food_delta,
        "drink": config.drink_delta,
        "energy": config.energy_delta,
    }
    potential = 0.0
    for key in SURVIVAL_KEYS:
        max_value = float(crafter.constants.items[key]["max"])
        potential += weights[key] * (float(inventory[key]) / max_value)
    potential += config.food_drink_balance_delta * get_food_drink_balance_level(inventory)
    return float(potential)


def get_food_drink_balance_level(inventory):
    """Return the weaker food/drink status as a compact balance signal."""
    food_max = float(crafter.constants.items["food"]["max"])
    drink_max = float(crafter.constants.items["drink"]["max"])
    food_level = float(inventory["food"]) / food_max
    drink_level = float(inventory["drink"]) / drink_max
    return float(min(food_level, drink_level))


def get_survival_recovery_events(prev_inventory, curr_inventory, action_name, player):
    """Describe every positive recovery of the four survival values."""
    events = []
    for key in SURVIVAL_KEYS:
        delta = int(curr_inventory[key] - prev_inventory[key])
        if delta <= 0:
            continue
        cause = get_recovery_cause(key, action_name, player)
        events.append(f"recover {key} {delta} value by {cause}")
    return events


def get_recovery_cause(key, action_name, player):
    if key in ("health", "food") and action_name.startswith("eat_"):
        return f"eating {action_name.removeprefix('eat_')}"
    if key == "health":
        return "sleeping" if player.sleeping else "natural regeneration"
    if key == "drink":
        if action_name == "drink":
            return "drinking water"
        return "drinking"
    if key == "energy":
        return "sleeping"
    return action_name


def has_items(inventory, required):
    return all(inventory[item] >= amount for item, amount in required.items())


def is_action_valid(env, action_name):
    """Check action preconditions before stepping the underlying environment."""
    player = env.player
    world = env.world
    inventory = player.inventory

    if player.sleeping and inventory["energy"] < crafter.constants.items["energy"]["max"]:
        return action_name == "sleep"

    if action_name == "noop":
        return True

    if action_name.startswith("move_"):
        directions = {
            "move_left": (-1, 0),
            "move_right": (1, 0),
            "move_up": (0, -1),
            "move_down": (0, 1),
        }
        target = player.pos + np.array(directions[action_name])
        material, obj = world[target]
        return obj is None and material in player.walkable

    if action_name == "sleep":
        return inventory["energy"] < crafter.constants.items["energy"]["max"]

    if action_name == "drink":
        return inventory["water_bucket"] > 0

    if action_name.startswith("eat_"):
        item = action_name.removeprefix("eat_")
        return inventory[item] > 0

    target = player.pos + np.array(player.facing)
    material, obj = world[target]

    if action_name == "do":
        if obj is not None:
            return isinstance(
                obj,
                (crafter.objects.Zombie, crafter.objects.Skeleton, crafter.objects.Cow),
            )
        if material == "water":
            return True
        collect_info = crafter.constants.collect.get(material)
        if not collect_info:
            return False
        return has_items(inventory, collect_info["require"])

    if action_name.startswith("place_"):
        name = action_name.removeprefix("place_")
        place_info = crafter.constants.place[name]
        return (
            obj is None
            and material in place_info["where"]
            and has_items(inventory, place_info["uses"])
        )

    if action_name.startswith("make_"):
        name = action_name.removeprefix("make_")
        make_info = crafter.constants.make[name]
        nearby, _ = world.nearby(player.pos, 1)
        return (
            all(util in nearby for util in make_info["nearby"])
            and has_items(inventory, make_info["uses"])
        )

    return True


def get_action_mask(env):
    """Return a bool mask for currently executable actions."""
    return np.array(
        [is_action_valid(env, action_name) for action_name in env.action_names],
        dtype=bool,
    )


def is_successful_eat(action_name, prev_inventory, curr_inventory):
    if not action_name.startswith("eat_"):
        return False
    item = action_name.removeprefix("eat_")
    return curr_inventory.get(item, 0) < prev_inventory.get(item, 0)


def is_successful_drink(action_name, prev_inventory, curr_inventory):
    drink_recovered = curr_inventory.get("drink", 0) > prev_inventory.get("drink", 0)
    used_water_bucket = (
        action_name == "drink"
        and curr_inventory.get("water_bucket", 0) < prev_inventory.get("water_bucket", 0)
    )
    return drink_recovered or used_water_bucket


def infer_death_cause(prev_inventory, curr_inventory, player, world):
    if curr_inventory["health"] > 0:
        return None
    if curr_inventory["food"] <= 0:
        return "starvation"
    if curr_inventory["drink"] <= 0:
        return "dehydration"
    if has_nearby_hostile(player, world):
        return "monster_attack"
    if prev_inventory["health"] > curr_inventory["health"]:
        return "monster_attack"
    return "other"


def has_nearby_hostile(player, world, radius=4):
    try:
        _materials, objs = world.nearby(player.pos, radius)
    except Exception:
        return False
    return any(isinstance(obj, (crafter.objects.Zombie, crafter.objects.Skeleton)) for obj in objs)


def empty_reward_component_totals():
    return {key: 0.0 for key in REWARD_COMPONENT_GROUPS}


def empty_move_counts():
    return {key: 0 for key in MOVE_ACTION_NAMES}


def group_reward_components(components):
    return {
        "terminal": float(components.get("death_penalty", 0.0))
        + float(components.get("max_steps_bonus", 0.0)),
        "low_status": float(components.get("low_status_penalty", 0.0)),
        "alive": float(components.get("alive_bonus", 0.0)),
        "potential": float(components.get("status_potential_reward", 0.0)),
    }


class BasicSurvivalRewardEnv(gym.Env):
    """Gymnasium-compatible wrapper around crafter.Env.

    The underlying Crafter reward is disabled. The reward returned to PPO comes
    only from get_survival_reward().
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(
        self,
        episode_steps=200,
        seed=None,
        area=(64, 64),
        view=(9, 12),
        render_size=(320, 384),
        reward_config=None,
        print_recovery_probes=False,
        monster_spawn_rate=MONSTER_SPAWN_RATE,
        use_new_map=False,
    ):
        super().__init__()
        self.episode_steps = episode_steps
        self.render_size = tuple(render_size)
        self.reward_config = reward_config or SurvivalRewardConfig()
        self.print_recovery_probes = print_recovery_probes
        self.use_new_map = bool(use_new_map)
        self.env = crafter.Env(
            area=area,
            view=view,
            reward=False,
            length=episode_steps,
            seed=seed,
            boss=False,
            footprints=True,
            other_agents=True,
            monster_spawn_rate=monster_spawn_rate,
            view_type="symbolic",
        )
        self.action_space = spaces.Discrete(self.env.action_space.n)
        self._map_obs_shape = (
            (NEW_MAP_SIZE, NEW_MAP_SIZE)
            if self.use_new_map
            else self.env.observation_space.shape
        )
        self._map_history_length = (
            NEW_MAP_HISTORY_LENGTH if self.use_new_map else OBS_HISTORY_LENGTH
        )
        self._area = tuple(area)
        self._extra_obs_keys = INVENTORY_OBS_KEYS
        observation_spaces = {
            "map": spaces.Box(
                low=0,
                high=UNKNOWN_MAP_TOKEN_ID if self.use_new_map else MAP_CHANNELS - 1,
                shape=(self._map_history_length,) + tuple(self._map_obs_shape),
                dtype=np.int64,
            ),
            "stats": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(OBS_HISTORY_LENGTH, STATS_OBS_SIZE),
                dtype=np.float32,
            ),
        }
        if not self.use_new_map:
            observation_spaces["memory"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(MEMORY_MAP_CHANNELS, MEMORY_MAP_SIZE, MEMORY_MAP_SIZE),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(observation_spaces)
        self._last_inventory = None
        self._observation_history = []
        self._memory_map = None
        self._token_memory_map = None
        self._episode_eat_count = 0
        self._episode_drink_count = 0
        self._episode_visited_positions = set()
        self._episode_reward_component_totals = empty_reward_component_totals()
        self._episode_move_counts = empty_move_counts()

    @property
    def action_names(self):
        return self.env.action_names

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env._seed = int(seed)
        obs = self.env.reset()
        self._last_inventory = self.env.player.inventory.copy()
        self._reset_memory_state()
        self._update_memory_state(obs)
        self._episode_eat_count = 0
        self._episode_drink_count = 0
        self._reset_episode_diagnostics()
        current_obs = self.encode_observation(obs, self._last_inventory)
        self._reset_observation_history(current_obs)
        info = {
            "inventory": self._last_inventory.copy(),
            "action_names": self.action_names,
        }
        return self._stack_observation_history(), info

    def step(self, action):
        action_name = self.action_names[int(action)]
        action_valid = is_action_valid(self.env, action_name)
        obs, _base_reward, done, info = self.env.step(int(action))
        curr_inventory = info["inventory"].copy()
        terminated = bool(self.env.player.health <= 0)
        truncated = bool(done and not terminated)
        reward, components = get_survival_reward(
            self._last_inventory,
            curr_inventory,
            done,
            truncated,
            self.reward_config,
        )
        invalid_action_penalty = 0.0
        if not action_valid:
            invalid_action_penalty = -self.reward_config.invalid_action_penalty
            reward += invalid_action_penalty
        components["invalid_action_penalty"] = invalid_action_penalty
        components["survival_reward"] = float(reward)
        self._add_episode_reward_components(components)
        recovery_events = get_survival_recovery_events(
            self._last_inventory,
            curr_inventory,
            action_name,
            self.env.player,
        )
        if is_successful_eat(action_name, self._last_inventory, curr_inventory):
            self._episode_eat_count += 1
        if is_successful_drink(action_name, self._last_inventory, curr_inventory):
            self._episode_drink_count += 1
        if action_name in self._episode_move_counts:
            self._episode_move_counts[action_name] += 1
        death_cause = infer_death_cause(
            self._last_inventory,
            curr_inventory,
            self.env.player,
            self.env.world,
        )
        self._last_inventory = curr_inventory
        self._mark_visited_position()

        info = dict(info)
        info["base_reward_ignored"] = _base_reward
        info["survival_reward_components"] = components
        info["action_name"] = action_name
        info["action_valid"] = action_valid
        info["reward"] = reward
        info["survival_success"] = bool(truncated and curr_inventory["health"] > 0)
        info["episode_eat_count"] = int(self._episode_eat_count)
        info["episode_drink_count"] = int(self._episode_drink_count)
        info["episode_visited_area"] = int(len(self._episode_visited_positions))
        info["episode_reward_components"] = dict(self._episode_reward_component_totals)
        info["episode_move_counts"] = dict(self._episode_move_counts)
        info["death_cause"] = death_cause

        if self.print_recovery_probes:
            for event in recovery_events:
                print(f"[recovery] step={self.env._step} action={action_name}: {event}")
        self._update_memory_state(obs)
        self._append_observation_history(self.encode_observation(obs, curr_inventory))
        return self._stack_observation_history(), reward, terminated, truncated, info

    def action_masks(self):
        # Keep masks outside the observation history. They must describe only the
        # currently executable actions in the live underlying environment state.
        return get_action_mask(self.env)

    def encode_observation(self, obs, inventory):
        if self.use_new_map:
            map_obs = self._get_local_token_map()
        else:
            map_ids = obs.astype(np.int64)
            map_obs = np.clip(map_ids, 0, MAP_CHANNELS - 1)
        extra_obs = []
        for key in self._extra_obs_keys:
            max_value = float(crafter.constants.items[key]["max"])
            extra_obs.append(float(inventory[key]) / max_value)
        extra_obs.extend(self._get_position_observation())
        observation = {
            "map": map_obs.astype(np.int64),
            "stats": np.array(extra_obs, dtype=np.float32),
        }
        if not self.use_new_map:
            observation["memory"] = self._get_local_memory().astype(np.float32)
        return observation

    def _get_position_observation(self):
        player_pos = np.array(self.env.player.pos, dtype=np.float32)
        area = np.maximum(np.array(self._area, dtype=np.float32) - 1.0, 1.0)
        position = np.clip(player_pos / area, 0.0, 1.0)
        return [float(position[0]), float(position[1])]

    def _reset_episode_diagnostics(self):
        self._episode_reward_component_totals = empty_reward_component_totals()
        self._episode_move_counts = empty_move_counts()
        self._episode_visited_positions = set()
        self._mark_visited_position()

    def _mark_visited_position(self):
        player_pos = np.array(self.env.player.pos)
        self._episode_visited_positions.add((int(player_pos[0]), int(player_pos[1])))

    def _add_episode_reward_components(self, components):
        grouped = group_reward_components(components)
        for key, value in grouped.items():
            self._episode_reward_component_totals[key] += float(value)

    def _reset_observation_history(self, observation):
        self._observation_history = [
            {
                "map": observation["map"].copy(),
                "stats": observation["stats"].copy(),
            }
            for _ in range(OBS_HISTORY_LENGTH)
        ]

    def _append_observation_history(self, observation):
        self._observation_history.append(
            {
                "map": observation["map"].copy(),
                "stats": observation["stats"].copy(),
            }
        )
        self._observation_history = self._observation_history[-OBS_HISTORY_LENGTH:]

    def _stack_observation_history(self):
        if self.use_new_map:
            map_stack = self._observation_history[-1]["map"][None, ...]
        else:
            map_stack = np.stack(
                [observation["map"] for observation in self._observation_history],
                axis=0,
            )
        observation = {
            "map": map_stack.astype(np.int64),
            "stats": np.stack(
                [observation["stats"] for observation in self._observation_history],
                axis=0,
            ).astype(np.float32),
        }
        if not self.use_new_map:
            observation["memory"] = self._get_local_memory().astype(np.float32)
        return observation

    def _reset_memory_state(self):
        if self.use_new_map:
            self._reset_token_memory_map()
        else:
            self._reset_memory_map()

    def _update_memory_state(self, obs):
        if self.use_new_map:
            self._update_token_memory_map(obs)
        else:
            self._update_memory_map(obs)

    def _reset_memory_map(self):
        self._memory_map = np.zeros(
            (MEMORY_MAP_CHANNELS, self._area[0], self._area[1]),
            dtype=np.float32,
        )

    def _reset_token_memory_map(self):
        self._token_memory_map = np.full(
            self._area,
            UNKNOWN_MAP_TOKEN_ID,
            dtype=np.int64,
        )

    def _update_memory_map(self, obs):
        if self._memory_map is None:
            self._reset_memory_map()
        obs_shape = np.array(obs.shape)
        bias = obs_shape // 2
        player_pos = np.array(self.env.player.pos)
        for x in range(obs.shape[0]):
            for y in range(obs.shape[1]):
                world_pos = player_pos + np.array((x, y)) - bias
                wx, wy = int(world_pos[0]), int(world_pos[1])
                if not (0 <= wx < self._area[0] and 0 <= wy < self._area[1]):
                    continue
                token = int(obs[x, y])
                self._memory_map[0, wx, wy] = 1.0
                if token in WATER_MEMORY_TOKEN_IDS:
                    self._memory_map[2, wx, wy] = 1.0
                if token in FOOD_MEMORY_TOKEN_IDS:
                    self._memory_map[3, wx, wy] = 1.0
                if token in HOSTILE_MEMORY_TOKEN_IDS:
                    self._memory_map[4, wx, wy] = 1.0
                if token > 0 and token not in WALKABLE_TOKEN_IDS:
                    self._memory_map[5, wx, wy] = 1.0
        px, py = int(player_pos[0]), int(player_pos[1])
        if 0 <= px < self._area[0] and 0 <= py < self._area[1]:
            self._memory_map[1, px, py] = 1.0

    def _update_token_memory_map(self, obs):
        if self._token_memory_map is None:
            self._reset_token_memory_map()
        obs_shape = np.array(obs.shape)
        bias = obs_shape // 2
        player_pos = np.array(self.env.player.pos)
        for x in range(obs.shape[0]):
            for y in range(obs.shape[1]):
                world_pos = player_pos + np.array((x, y)) - bias
                wx, wy = int(world_pos[0]), int(world_pos[1])
                if not (0 <= wx < self._area[0] and 0 <= wy < self._area[1]):
                    continue
                token = int(np.clip(obs[x, y], 0, MAP_CHANNELS - 1))
                self._token_memory_map[wx, wy] = token

    def _get_local_token_map(self):
        if self._token_memory_map is None:
            self._reset_token_memory_map()
        player_pos = np.array(self.env.player.pos)
        half = NEW_MAP_SIZE // 2
        local = np.full(
            (NEW_MAP_SIZE, NEW_MAP_SIZE),
            UNKNOWN_MAP_TOKEN_ID,
            dtype=np.int64,
        )
        for x in range(NEW_MAP_SIZE):
            for y in range(NEW_MAP_SIZE):
                world_pos = player_pos + np.array((x - half, y - half))
                wx, wy = int(world_pos[0]), int(world_pos[1])
                if 0 <= wx < self._area[0] and 0 <= wy < self._area[1]:
                    local[x, y] = self._token_memory_map[wx, wy]
        return local

    def _get_local_memory(self):
        if self._memory_map is None:
            self._reset_memory_map()
        player_pos = np.array(self.env.player.pos)
        half = MEMORY_MAP_SIZE // 2
        local = np.zeros(
            (MEMORY_MAP_CHANNELS, MEMORY_MAP_SIZE, MEMORY_MAP_SIZE),
            dtype=np.float32,
        )
        for x in range(MEMORY_MAP_SIZE):
            for y in range(MEMORY_MAP_SIZE):
                world_pos = player_pos + np.array((x - half, y - half))
                wx, wy = int(world_pos[0]), int(world_pos[1])
                if 0 <= wx < self._area[0] and 0 <= wy < self._area[1]:
                    local[:, x, y] = self._memory_map[:, wx, wy]
        return local

    def render(self):
        return self.env.render(self.render_size)

    def close(self):
        return None


class TrainingRenderCallback:
    """Save rendered frames and report episode diagnostics on separate cadences."""

    def __init__(
        self,
        render_every=5,
        report_every=5,
        output_dir="save/01-basic_reward/renders",
    ):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer
                self.output_dir = pathlib.Path(outer.output_dir)
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.episode_indices = None
                self.next_episode_index = 1
                self.current_frames = None
                self.completed_episodes = 0
                self.successful_episodes = 0
                self.recent_successes = []
                self.total_eat_count = 0
                self.recent_eat_counts = []
                self.total_drink_count = 0
                self.recent_drink_counts = []
                self.total_visited_area = 0
                self.recent_visited_areas = []
                self.total_reward_components = empty_reward_component_totals()
                self.recent_reward_components = empty_reward_component_totals()
                self.total_move_counts = empty_move_counts()
                self.recent_move_counts = empty_move_counts()
                self.total_supply_table = self.outer.empty_supply_table()
                self.recent_supply_table = self.outer.empty_supply_table()
                self.total_deaths = 0
                self.recent_deaths = 0
                self.total_death_causes = self.outer.empty_death_counts()
                self.recent_death_causes = self.outer.empty_death_counts()

            def _on_training_start(self):
                self._init_episode_tracking()

            def _init_episode_tracking(self):
                num_envs = int(getattr(self.training_env, "num_envs", 1))
                self.episode_indices = list(range(1, num_envs + 1))
                self.next_episode_index = num_envs + 1
                self.current_frames = [[] for _ in range(num_envs)]

            def _on_step(self):
                if self.episode_indices is None:
                    self._init_episode_tracking()

                record_envs = [
                    env_index
                    for env_index, episode_index in enumerate(self.episode_indices)
                    if self.outer.should_record(episode_index)
                ]
                if record_envs:
                    rendered_frames = self.outer.render_training_envs(self, record_envs)
                    for env_index, frame in zip(record_envs, rendered_frames):
                        self.current_frames[env_index].append(frame)

                dones = self.locals.get("dones", [])
                infos = self.locals.get("infos", [{} for _ in range(len(dones))])
                for env_index, done in enumerate(dones):
                    if not done:
                        continue
                    episode_index = self.episode_indices[env_index]
                    success = bool(infos[env_index].get("survival_success", False))
                    eat_count = int(infos[env_index].get("episode_eat_count", 0))
                    drink_count = int(infos[env_index].get("episode_drink_count", 0))
                    visited_area = int(infos[env_index].get("episode_visited_area", 0))
                    reward_components = infos[env_index].get(
                        "episode_reward_components",
                        empty_reward_component_totals(),
                    )
                    move_counts = infos[env_index].get(
                        "episode_move_counts",
                        empty_move_counts(),
                    )
                    death_cause = infos[env_index].get("death_cause")
                    self.completed_episodes += 1
                    self.successful_episodes += int(success)
                    self.recent_successes.append(success)
                    self.total_eat_count += eat_count
                    self.recent_eat_counts.append(eat_count)
                    self.total_drink_count += drink_count
                    self.recent_drink_counts.append(drink_count)
                    self.total_visited_area += visited_area
                    self.recent_visited_areas.append(visited_area)
                    self.outer.add_reward_components(
                        self.total_reward_components,
                        reward_components,
                    )
                    self.outer.add_reward_components(
                        self.recent_reward_components,
                        reward_components,
                    )
                    self.outer.add_move_counts(self.total_move_counts, move_counts)
                    self.outer.add_move_counts(self.recent_move_counts, move_counts)
                    supply_key = self.outer.supply_table_key(eat_count, drink_count)
                    self.total_supply_table[supply_key] += 1
                    self.recent_supply_table[supply_key] += 1
                    if death_cause:
                        self.total_deaths += 1
                        self.recent_deaths += 1
                        if death_cause in self.total_death_causes:
                            self.total_death_causes[death_cause] += 1
                        if death_cause in self.recent_death_causes:
                            self.recent_death_causes[death_cause] += 1
                    if self.current_frames[env_index]:
                        self.outer.save_frames(
                            self.current_frames[env_index],
                            episode_index,
                            self.num_timesteps,
                            self.output_dir,
                            env_index,
                        )
                    self.current_frames[env_index] = []
                    self.episode_indices[env_index] = self.next_episode_index
                    self.next_episode_index += 1
                    if self.outer.should_report_success(self.completed_episodes):
                        self.outer.report_success_rate(
                            self.completed_episodes,
                            self.successful_episodes,
                            self.recent_successes,
                            self.total_eat_count,
                            self.recent_eat_counts,
                            self.total_drink_count,
                            self.recent_drink_counts,
                            self.total_visited_area,
                            self.recent_visited_areas,
                            self.total_reward_components,
                            self.recent_reward_components,
                            self.total_move_counts,
                            self.recent_move_counts,
                            self.total_supply_table,
                            self.recent_supply_table,
                            self.total_deaths,
                            self.recent_deaths,
                            self.total_death_causes,
                            self.recent_death_causes,
                            self.num_timesteps,
                        )
                        self.recent_successes = []
                        self.recent_eat_counts = []
                        self.recent_drink_counts = []
                        self.recent_visited_areas = []
                        self.recent_reward_components = empty_reward_component_totals()
                        self.recent_move_counts = empty_move_counts()
                        self.recent_supply_table = self.outer.empty_supply_table()
                        self.recent_deaths = 0
                        self.recent_death_causes = self.outer.empty_death_counts()
                return True

        self.render_every = render_every
        self.report_every = report_every
        self.output_dir = output_dir
        self.callback = _Callback(self)

    def should_record(self, episode_index):
        if self.render_every <= 0:
            return False
        return episode_index % self.render_every == 0

    def should_report_success(self, completed_episodes):
        if self.report_every <= 0:
            return False
        return completed_episodes % self.report_every == 0

    def empty_death_counts(self):
        return {
            "starvation": 0,
            "dehydration": 0,
            "monster_attack": 0,
        }

    def empty_supply_table(self):
        return {
            "no_eat_no_drink": 0,
            "eat_no_drink": 0,
            "no_eat_drink": 0,
            "eat_drink": 0,
        }

    def supply_table_key(self, eat_count, drink_count):
        ate = eat_count > 0
        drank = drink_count > 0
        if ate and drank:
            return "eat_drink"
        if ate:
            return "eat_no_drink"
        if drank:
            return "no_eat_drink"
        return "no_eat_no_drink"

    def report_success_rate(
        self,
        completed_episodes,
        successful_episodes,
        recent_successes,
        total_eat_count,
        recent_eat_counts,
        total_drink_count,
        recent_drink_counts,
        total_visited_area,
        recent_visited_areas,
        total_reward_components,
        recent_reward_components,
        total_move_counts,
        recent_move_counts,
        total_supply_table,
        recent_supply_table,
        total_deaths,
        recent_deaths,
        total_death_causes,
        recent_death_causes,
        timesteps,
    ):
        recent_rate = np.mean(recent_successes) if recent_successes else 0.0
        total_rate = successful_episodes / completed_episodes if completed_episodes else 0.0
        recent_avg_eats = np.mean(recent_eat_counts) if recent_eat_counts else 0.0
        total_avg_eats = total_eat_count / completed_episodes if completed_episodes else 0.0
        recent_avg_drinks = np.mean(recent_drink_counts) if recent_drink_counts else 0.0
        total_avg_drinks = total_drink_count / completed_episodes if completed_episodes else 0.0
        recent_avg_visited = np.mean(recent_visited_areas) if recent_visited_areas else 0.0
        total_avg_visited = total_visited_area / completed_episodes if completed_episodes else 0.0
        recent_episodes = len(recent_successes)
        recent_supply_ratios = self.supply_ratios(recent_supply_table, recent_episodes)
        total_supply_ratios = self.supply_ratios(total_supply_table, completed_episodes)
        recent_reward_ratios = self.reward_component_ratios(recent_reward_components)
        total_reward_ratios = self.reward_component_ratios(total_reward_components)
        recent_reward_averages = self.average_reward_components(
            recent_reward_components,
            recent_episodes,
        )
        total_reward_averages = self.average_reward_components(
            total_reward_components,
            completed_episodes,
        )
        recent_move_ratios = self.move_ratios(recent_move_counts)
        total_move_ratios = self.move_ratios(total_move_counts)
        recent_avg_moves = (
            sum(recent_move_counts.values()) / recent_episodes
            if recent_episodes
            else 0.0
        )
        total_avg_moves = (
            sum(total_move_counts.values()) / completed_episodes
            if completed_episodes
            else 0.0
        )
        recent_death_ratios = self.death_ratios(recent_death_causes, recent_deaths)
        total_death_ratios = self.death_ratios(total_death_causes, total_deaths)
        print(
            ""
            f"[success] episodes={completed_episodes} timesteps={timesteps} "
            f"recent_success_rate={recent_rate:.3f} total_success_rate={total_rate:.3f}"
        )
        print(
            "[episode_stats]\n"
            f"  recent_avg_eats={recent_avg_eats:.3f} total_avg_eats={total_avg_eats:.3f}\n"
            f"  recent_avg_drinks={recent_avg_drinks:.3f} total_avg_drinks={total_avg_drinks:.3f}\n"
            f"  recent_avg_visited_area={recent_avg_visited:.3f} "
            f"total_avg_visited_area={total_avg_visited:.3f}\n"
            f"  recent_avg_move_actions={recent_avg_moves:.3f} "
            f"total_avg_move_actions={total_avg_moves:.3f}\n"
            "  recent_move_ratios left/right/up/down:\n"
            f"    {recent_move_ratios['move_left']:.3f}/"
            f"{recent_move_ratios['move_right']:.3f}/"
            f"{recent_move_ratios['move_up']:.3f}/"
            f"{recent_move_ratios['move_down']:.3f}\n"
            "  total_move_ratios left/right/up/down:\n"
            f"    {total_move_ratios['move_left']:.3f}/"
            f"{total_move_ratios['move_right']:.3f}/"
            f"{total_move_ratios['move_up']:.3f}/"
            f"{total_move_ratios['move_down']:.3f}\n"
            "  recent_reward_components avg_per_episode/abs_share "
            "(terminal,low_status,alive,potential):\n"
            f"    terminal={recent_reward_averages['terminal']:.3f}/"
            f"{recent_reward_ratios['terminal']:.3f}, "
            f"low_status={recent_reward_averages['low_status']:.3f}/"
            f"{recent_reward_ratios['low_status']:.3f}, "
            f"alive={recent_reward_averages['alive']:.3f}/"
            f"{recent_reward_ratios['alive']:.3f}, "
            f"potential={recent_reward_averages['potential']:.3f}/"
            f"{recent_reward_ratios['potential']:.3f}\n"
            "  total_reward_components avg_per_episode/abs_share "
            "(terminal,low_status,alive,potential):\n"
            f"    terminal={total_reward_averages['terminal']:.3f}/"
            f"{total_reward_ratios['terminal']:.3f}, "
            f"low_status={total_reward_averages['low_status']:.3f}/"
            f"{total_reward_ratios['low_status']:.3f}, "
            f"alive={total_reward_averages['alive']:.3f}/"
            f"{total_reward_ratios['alive']:.3f}, "
            f"potential={total_reward_averages['potential']:.3f}/"
            f"{total_reward_ratios['potential']:.3f}\n"
            "  recent_supply_table rows=eat(no,yes) cols=drink(no,yes):\n"
            f"    [{recent_supply_ratios['no_eat_no_drink']:.3f}, "
            f"{recent_supply_ratios['no_eat_drink']:.3f}]\n"
            f"    [{recent_supply_ratios['eat_no_drink']:.3f}, "
            f"{recent_supply_ratios['eat_drink']:.3f}]\n"
            "  total_supply_table rows=eat(no,yes) cols=drink(no,yes):\n"
            f"    [{total_supply_ratios['no_eat_no_drink']:.3f}, "
            f"{total_supply_ratios['no_eat_drink']:.3f}]\n"
            f"    [{total_supply_ratios['eat_no_drink']:.3f}, "
            f"{total_supply_ratios['eat_drink']:.3f}]\n"
            f"  recent_deaths={recent_deaths} recent_death_ratios="
            f"starvation:{recent_death_ratios['starvation']:.3f},"
            f"dehydration:{recent_death_ratios['dehydration']:.3f},"
            f"monster_attack:{recent_death_ratios['monster_attack']:.3f}\n"
            f"  total_deaths={total_deaths} total_death_ratios="
            f"starvation:{total_death_ratios['starvation']:.3f},"
            f"dehydration:{total_death_ratios['dehydration']:.3f},"
            f"monster_attack:{total_death_ratios['monster_attack']:.3f}\n"
        )

    def add_reward_components(self, target, source):
        for key in REWARD_COMPONENT_GROUPS:
            target[key] += float(source.get(key, 0.0))

    def add_move_counts(self, target, source):
        for key in MOVE_ACTION_NAMES:
            target[key] += int(source.get(key, 0))

    def move_ratios(self, move_counts):
        denominator = sum(int(move_counts.get(key, 0)) for key in MOVE_ACTION_NAMES)
        if denominator <= 0:
            return {key: 0.0 for key in MOVE_ACTION_NAMES}
        return {
            key: int(move_counts.get(key, 0)) / denominator
            for key in MOVE_ACTION_NAMES
        }

    def reward_component_ratios(self, components):
        denominator = sum(abs(float(components.get(key, 0.0))) for key in REWARD_COMPONENT_GROUPS)
        if denominator <= 0:
            return {key: 0.0 for key in REWARD_COMPONENT_GROUPS}
        return {
            key: abs(float(components.get(key, 0.0))) / denominator
            for key in REWARD_COMPONENT_GROUPS
        }

    def average_reward_components(self, components, episodes):
        if episodes <= 0:
            return {key: 0.0 for key in REWARD_COMPONENT_GROUPS}
        return {
            key: float(components.get(key, 0.0)) / episodes
            for key in REWARD_COMPONENT_GROUPS
        }

    def supply_ratios(self, supply_table, episodes):
        if episodes <= 0:
            return {key: 0.0 for key in self.empty_supply_table()}
        return {
            key: supply_table.get(key, 0) / episodes
            for key in self.empty_supply_table()
        }

    def death_ratios(self, death_causes, deaths):
        if deaths <= 0:
            return {cause: 0.0 for cause in self.empty_death_counts()}
        return {
            cause: death_causes.get(cause, 0) / deaths
            for cause in self.empty_death_counts()
        }

    def render_training_envs(self, callback, env_indices):
        try:
            return callback.training_env.env_method("render", indices=env_indices)
        except TypeError:
            frames = callback.training_env.env_method("render")
            return [frames[env_index] for env_index in env_indices]

    def save_frames(self, frames, episode, timesteps, output_dir, env_index):
        filename = output_dir / f"episode_{episode:04d}_env_{env_index:02d}_step_{timesteps}.gif"
        imageio.mimsave(filename, frames, duration=0.12)
        print(
            f"[render] training_episode={episode} env={env_index} timesteps={timesteps} "
            f"saved={filename}"
        )


def make_env(
    seed=None,
    episode_steps=200,
    reward_config=None,
    print_recovery_probes=False,
    use_new_map=False,
):
    return BasicSurvivalRewardEnv(
        episode_steps=episode_steps,
        seed=seed,
        reward_config=reward_config,
        print_recovery_probes=print_recovery_probes,
        use_new_map=use_new_map,
    )


def make_env_factory(rank, args, reward_config):
    def _init():
        return make_env(
            seed=args.seed + rank if args.seed is not None else None,
            episode_steps=args.episode_steps,
            reward_config=reward_config,
            print_recovery_probes=args.print_recovery_probes and rank == 0,
            use_new_map=args.new_map,
        )

    return _init


def build_training_env(args, reward_config, output_dir):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    if args.n_envs == 1:
        return Monitor(
            make_env(
                seed=args.seed,
                episode_steps=args.episode_steps,
                reward_config=reward_config,
                print_recovery_probes=args.print_recovery_probes,
                use_new_map=args.new_map,
            ),
            filename=str(output_dir / "monitor.csv"),
        )

    factories = [
        make_env_factory(rank, args, reward_config)
        for rank in range(args.n_envs)
    ]
    vec_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv
    vec_env = vec_cls(factories)
    return VecMonitor(vec_env, filename=str(output_dir / "monitor.csv"))


def archive_previous_run(output_dir, keep=3):
    """Archive monitor CSVs, renders, and model zips into output_dir/old.

    Future experiment scripts can reuse this layout:
    saves/<experiment_name>/
      old/
        YYYYmmdd_HHMMSS.zip
      monitor.csv
      renders/
      <model>.zip
    """
    output_dir = pathlib.Path(output_dir)
    old_dir = output_dir / "old"
    archive_sources = []

    for pattern in ("*.csv", "*.zip"):
        archive_sources.extend(
            path for path in output_dir.glob(pattern)
            if path.is_file() and path.parent != old_dir
        )
    renders_dir = output_dir / "renders"
    if renders_dir.exists() and any(renders_dir.iterdir()):
        archive_sources.append(renders_dir)

    if not archive_sources:
        old_dir.mkdir(parents=True, exist_ok=True)
        return None

    old_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = old_dir / f"run_{timestamp}.zip"

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source in archive_sources:
            if source.is_dir():
                for file_path in source.rglob("*"):
                    if file_path.is_file():
                        zf.write(file_path, file_path.relative_to(output_dir))
            else:
                zf.write(source, source.relative_to(output_dir))

    for source in archive_sources:
        if source.is_dir():
            shutil.rmtree(source)
        elif source.exists():
            source.unlink()

    archives = sorted(old_dir.glob("run_*.zip"), key=lambda path: path.stat().st_mtime)
    while len(archives) > keep:
        archives.pop(0).unlink()

    print(f"[archive] previous run saved to {archive_path}")
    return archive_path


def collect_provided_options(argv):
    return {
        token.split("=", 1)[0]
        for token in argv
        if token.startswith("--")
    }


def values_match(saved_value, requested_value):
    if isinstance(saved_value, float) or isinstance(requested_value, float):
        return math.isclose(float(saved_value), float(requested_value), rel_tol=1e-12, abs_tol=1e-12)
    return saved_value == requested_value


def schedule_value(value):
    if callable(value):
        return value(1.0)
    return value


def build_learning_rate(initial_lr, total_timesteps):
    if not PPO_ENABLE_LR_DECAY:
        return initial_lr

    start_fraction = float(PPO_LR_DECAY_START_FRACTION)
    final_fraction = float(PPO_LR_DECAY_FINAL_FRACTION)
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("PPO_LR_DECAY_START_FRACTION must be in [0.0, 1.0).")
    if final_fraction < 0.0:
        raise ValueError("PPO_LR_DECAY_FINAL_FRACTION must be non-negative.")

    decay_start_step = int(total_timesteps * start_fraction)
    final_lr = float(initial_lr) * final_fraction
    print(
        "[lr_schedule] "
        f"initial={initial_lr:g} start_step={decay_start_step} "
        f"final={final_lr:g} total_timesteps={total_timesteps}"
    )

    def schedule(progress_remaining):
        elapsed_fraction = 1.0 - float(progress_remaining)
        if elapsed_fraction <= start_fraction:
            return float(initial_lr)
        decay_progress = (elapsed_fraction - start_fraction) / (1.0 - start_fraction)
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        return float(initial_lr) + (final_lr - float(initial_lr)) * decay_progress

    return schedule


def validate_resume_hyperparams(model, args):
    conflicts = []
    for flag, attr in PPO_HYPERPARAM_FLAGS.items():
        if flag not in args.provided_options:
            continue
        saved_value = getattr(model, attr)
        requested_value = getattr(args, attr)
        if not values_match(saved_value, requested_value):
            conflicts.append((flag, attr, saved_value, requested_value))

    if conflicts:
        lines = [
            "Resume hyperparameter conflict detected.",
            f"The checkpoint keeps its saved {PPO_ALGORITHM_NAME} hyperparameters; conflicting CLI overrides are not allowed.",
        ]
        for flag, attr, saved_value, requested_value in conflicts:
            lines.append(
                f"  {flag} / {attr}: checkpoint={saved_value!r}, requested={requested_value!r}"
            )
        raise SystemExit("\n".join(lines))


def validate_resume_fixed_hyperparams(model):
    expected = {
        "ent_coef": PPO_ENT_COEF,
        "clip_range": PPO_CLIP_RANGE,
        "gamma": PPO_GAMMA,
        "n_epochs": PPO_N_EPOCHS,
    }
    conflicts = []
    for attr, expected_value in expected.items():
        saved_value = schedule_value(getattr(model, attr))
        if not values_match(saved_value, expected_value):
            conflicts.append((attr, saved_value, expected_value))

    if conflicts:
        lines = [
            "Resume fixed-hyperparameter conflict detected.",
            "Edit the checkpoint-compatible constants or start a fresh run for the new PPO settings.",
        ]
        for attr, saved_value, expected_value in conflicts:
            lines.append(
                f"  {attr}: checkpoint={saved_value!r}, current={expected_value!r}"
            )
        raise SystemExit("\n".join(lines))


def observation_space_signature(space):
    if hasattr(space, "spaces"):
        return {
            key: (tuple(subspace.shape), str(subspace.dtype))
            for key, subspace in space.spaces.items()
        }
    return (tuple(space.shape), str(space.dtype))


def validate_resume_observation_space(model, env):
    saved_signature = observation_space_signature(model.observation_space)
    current_signature = observation_space_signature(env.observation_space)
    if saved_signature != current_signature:
        raise SystemExit(
            "Resume observation-space conflict detected.\n"
            f"  checkpoint observation_space={saved_signature}\n"
            f"  current observation_space={current_signature}\n"
            "This script uses Dict observations. Old-map and --new-map checkpoints "
            "have different observation spaces and cannot be resumed across modes."
        )


def validate_resume_policy_arch(model, env):
    saved_arch = getattr(model.policy, "net_arch", None)
    if saved_arch != POLICY_NET_ARCH:
        raise SystemExit(
            "Resume policy-architecture conflict detected.\n"
            f"  checkpoint policy.net_arch={saved_arch!r}\n"
            f"  current policy.net_arch={POLICY_NET_ARCH!r}\n"
            "Start a fresh run for the current policy MLP, or resume from a checkpoint "
            "created with the same policy architecture."
        )
    saved_extractor = model.policy.features_extractor
    saved_extractor_name = type(saved_extractor).__name__
    saved_map_embedding = getattr(saved_extractor, "map_embedding", None)
    saved_map_cnn = getattr(saved_extractor, "map_cnn", None)
    saved_map_head = getattr(saved_extractor, "map_head", None)
    saved_memory_cnn = getattr(saved_extractor, "memory_cnn", None)
    current_observation_space = env.observation_space
    expected_features_dim = get_expected_feature_dim(current_observation_space)
    expected_map_embeddings = get_map_num_embeddings(current_observation_space)
    expected_map_cnn_in = get_expected_map_cnn_in_channels(current_observation_space)
    expected_memory_cnn_in = get_expected_memory_cnn_in_channels(current_observation_space)
    expected_map_head_in = get_expected_map_head_in_features(current_observation_space)
    saved_map_cnn_in = (
        getattr(saved_map_cnn[0], "in_channels", None)
        if saved_map_cnn is not None and len(saved_map_cnn) > 0
        else None
    )
    saved_memory_cnn_in = (
        getattr(saved_memory_cnn[0], "in_channels", None)
        if saved_memory_cnn is not None and len(saved_memory_cnn) > 0
        else None
    )
    saved_map_head_in = (
        getattr(saved_map_head[0], "in_features", None)
        if saved_map_head is not None and len(saved_map_head) > 0
        else None
    )
    if (
        saved_extractor_name != CrafterMapStatsExtractor.__name__
        or saved_extractor.features_dim != expected_features_dim
        or saved_map_embedding is None
        or saved_map_embedding.num_embeddings != expected_map_embeddings
        or saved_map_embedding.embedding_dim != MAP_EMBEDDING_DIM
        or saved_map_cnn_in != expected_map_cnn_in
        or saved_memory_cnn_in != expected_memory_cnn_in
        or saved_map_head_in != expected_map_head_in
    ):
        raise SystemExit(
            "Resume feature-extractor conflict detected.\n"
            f"  checkpoint features_extractor={saved_extractor_name}\n"
            f"  checkpoint features_dim={saved_extractor.features_dim}\n"
            f"  checkpoint map_embeddings={getattr(saved_map_embedding, 'num_embeddings', None)}\n"
            f"  checkpoint map_embedding_dim={getattr(saved_map_embedding, 'embedding_dim', None)}\n"
            f"  checkpoint map_cnn_in_channels={saved_map_cnn_in}\n"
            f"  checkpoint memory_cnn_in_channels={saved_memory_cnn_in}\n"
            f"  checkpoint map_head_in_features={saved_map_head_in}\n"
            f"  current features_extractor={CrafterMapStatsExtractor.__name__}\n"
            f"  current features_dim={expected_features_dim}\n"
            f"  current map_embeddings={expected_map_embeddings}\n"
            f"  current map_embedding_dim={MAP_EMBEDDING_DIM}\n"
            f"  current map_cnn_in_channels={expected_map_cnn_in}\n"
            f"  current memory_cnn_in_channels={expected_memory_cnn_in}\n"
            f"  current map_head_in_features={expected_map_head_in}\n"
            "Start a fresh run for the memory-map observation encoder."
        )


def train(args):
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit(
            "Missing MaskablePPO training dependencies. In the crafter conda environment, run:\n"
            '  pip install -e ".[training]"\n'
            "or install the missing package directly:\n"
            "  pip install sb3-contrib\n"
            "Do not install these into the base Python/Conda environment."
        ) from exc

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reward_config = SurvivalRewardConfig()
    model = None
    if args.resume_from:
        resume_path = pathlib.Path(args.resume_from)
        if not resume_path.exists():
            raise SystemExit(f"Checkpoint not found: {resume_path}")
        print(f"[resume] loading checkpoint: {resume_path}")
        model = MaskablePPO.load(resume_path, verbose=1)
        validate_resume_hyperparams(model, args)

    archive_previous_run(output_dir, keep=args.keep_archives)

    env = build_training_env(args, reward_config, output_dir)
    render_callback = TrainingRenderCallback(
        render_every=args.render_every,
        report_every=args.report_every,
        output_dir=output_dir / "renders",
    )

    if model is not None:
        validate_resume_observation_space(model, env)
        validate_resume_policy_arch(model, env)
        validate_resume_fixed_hyperparams(model)
        model.set_env(env)
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            seed=args.seed,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=build_learning_rate(args.learning_rate, args.total_timesteps),
            gamma=PPO_GAMMA,
            n_epochs=PPO_N_EPOCHS,
            ent_coef=PPO_ENT_COEF,
            clip_range=PPO_CLIP_RANGE,
            policy_kwargs=dict(
                features_extractor_class=CrafterMapStatsExtractor,
                net_arch=POLICY_NET_ARCH,
            ),
            # tensorboard_log=str(output_dir / "tb"),
        )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=render_callback.callback,
        log_interval=args.log_interval,
    )
    model.save(output_dir / MODEL_BASENAME)
    env.close()


def smoke_test(use_new_map=False):
    env = BasicSurvivalRewardEnv(
        episode_steps=5,
        seed=0,
        area=(16, 16),
        print_recovery_probes=True,
        use_new_map=use_new_map,
    )
    obs, info = env.reset()
    print("obs", {key: (value.shape, value.dtype) for key, value in obs.items()})
    print("actions", env.action_space.n, env.action_names)
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            step,
            env.action_names[action],
            round(reward, 4),
            info["inventory"],
            info["survival_reward_components"],
        )
        if terminated or truncated:
            break
    env.close()


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=524288)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-steps", type=int, default=512)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=("subproc", "dummy"), default="subproc")
    parser.add_argument("--new-map", action="store_true")
    parser.add_argument("--batch-size", type=int, default=PPO_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=PPO_LEARNING_RATE)

    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--log-interval", type=int, default=5)

    parser.add_argument("--render-every", type=int, default=64)
    parser.add_argument("--report-every", type=int, default=64)
    parser.add_argument("--print-recovery-probes", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--keep-archives", type=int, default=3)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    args.provided_options = collect_provided_options(argv)
    return args


def main():
    args = parse_args()
    if args.smoke_test:
        smoke_test(use_new_map=args.new_map)
    else:
        train(args)


if __name__ == "__main__":
    main()
