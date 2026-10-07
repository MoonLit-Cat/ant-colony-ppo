from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Cell types (static layer)
EMPTY, FOOD, TREE, COLONY = 0, 1, 2, 3

# Action
UP, DOWN, LEFT, RIGHT, EAT = 0, 1, 2, 3, 4
MOVES = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

# Vision channels
CH_FOOD, CH_TREE, CH_COLONY, CH_ANT, CH_WALL = range(5)
N_CHANNELS = 5
N_SCALARS = 5   # energy, health, inventory, colony dx, colony dy

@dataclass
class Ant:
    id: int
    x: int
    y: int
    energy: float
    health: float
    inventory: int = 0
    alive: bool = True

    @property
    def pos(self):
        return (self.x, self.y)
    
class ColonyEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human", "rgb_array"], "render_fps": 5}

    def __init__(self,
                grid_size = 20,
                n_food = 50,
                n_trees = 20,
                n_ants = 10,
                vision_size = 5,
                start_energy = 100.0,
                start_health = 100.0,
                food_energy = 30.0,
                energy_per_step = 1.0,
                starvation_damage = 1.0,
                inventory_capacity = 5,
                food_respawn_prob = 0.05,
                max_steps = 1000,
                
                # reward design
                step_penalty = -0.01,
                collect_reward = 2.0,
                deliver_reward = 10.0,  # per food unit delivered
                invalid_penalty = -0.2,
                death_penalty = -10.0,
                energy_reward = 0.05, 
                health_reward = 0.1,   
                render_mode = None,
                ):
        super().__init__()

        self.grid_size = grid_size
        self.n_food = n_food
        self.n_trees = n_trees
        self.n_ants = n_ants
        
        self.vision_size = vision_size
        self.vision_radius = vision_size // 2
        self.start_energy = start_energy
        self.start_health = start_health
        self.food_energy = food_energy
        self.energy_per_step = energy_per_step
        self.starvation_damage = starvation_damage
        self.inventory_capacity = inventory_capacity
        self.food_respawn_prob = food_respawn_prob
        self.max_steps = max_steps
 
        self.step_penalty = step_penalty
        self.collect_reward = collect_reward
        self.deliver_reward = deliver_reward
        self.invalid_penalty = invalid_penalty
        self.death_penalty = death_penalty
        self.energy_reward = energy_reward
        self.health_reward = health_reward
        self.render_mode = render_mode
 
        self.action_space = spaces.Discrete(5)

        # Flat, normalised observation: works directly with SB3 MlpPolicy
        obs_dim = N_CHANNELS * vision_size  * vision_size + N_SCALARS
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.world: np.ndarray | None = None
        self.colony_pos: tuple[int, int] | None = None
        self.colony_food = 0
        self.ants: list[Ant] = []
        self.step_count = 0
        self.stats: dict[str, int] = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)    # seeds self.np_random
        self.step_count = 0
        self.colony_food = 0
        self.ants = []
        self.stats = {"collected": 0, "delivered": 0, "eaten": 0, "invalid": 0, "died": 0}

        self.world = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        self.colony_pos = self.random_empty_cell()
        cx, cy = self.colony_pos
        self.world[cy, cx] = COLONY

        for _ in range(self.n_food):
            x, y = self.random_empty_cell()
            self.world[y, x] = FOOD
        for _ in range(self.n_trees):
            x, y = self.random_empty_cell()
            self.world[y, x] = TREE

        for i in range(self.n_ants):
            x, y = self.random_empty_cell()
            self.ants.append(Ant(i, x, y, energy=self.start_energy, health=self.start_health))
        
        return self.observe_all(), self.info()

    def step(self, actions):
        """actions: array of n_ants ints (one per ant). Returns per-ant obs / reward / terminated."""
        self.step_count += 1
        rewards = np.full(self.n_ants, self.step_penalty, dtype=np.float32)
        terminated = np.zeros(self.n_ants, dtype=bool)

        # random order so no ant always moves first
        for i in self.np_random.permutation(self.n_ants):
            ant = self.ants[i]
            ev = self.act(ant, int(actions[i]))
            ev["delivered"] = self.deliver(ant)
            self.metabolize(ant)

            rewards[i] += self.collect_reward * ev["collected"]
            rewards[i] += self.deliver_reward * ev["delivered"]
            rewards[i] += self.invalid_penalty * ev["invalid"]
            rewards[i] += self.energy_reward * (ant.energy / self.start_energy)
            rewards[i] += self.health_reward * (ant.health / self.start_health)
            if not ant.alive:
                rewards[i] += self.death_penalty
                terminated[i] = True
                ev["died"] = 1
            for k, v in ev.items():
                self.stats[k] += v

        self.respawn_food()

        # a dead ant is replaced by a fresh one, so there are always n_ants agents
        for i in np.flatnonzero(terminated):
            x, y = self.random_empty_cell()
            self.ants[i] = Ant(int(i), x, y, energy=self.start_energy, health=self.start_health)

        truncated = self.step_count >= self.max_steps
        return self.observe_all(), rewards, terminated, truncated, self.info()

    def act(self, ant: Ant, action: int) -> dict:
        ev = {"collected": 0, "eaten": 0, "invalid": 0}

        if action in MOVES:
            dx, dy = MOVES[action]
            nx, ny = ant.x + dx, ant.y + dy
            blocked = (
                not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size)
                or self.world[ny, nx] == TREE
                or self.occupied(nx, ny, exclude=ant)
            )
            if blocked:
                ev["invalid"] = 1
            else:
                ant.x, ant.y = nx, ny
                if (
                    self.world[ny, nx] == FOOD
                    and ant.inventory < self.inventory_capacity
                ):
                    self.world[ny, nx] = EMPTY
                    ant.inventory += 1
                    ev["collected"] = 1

        elif action == EAT:
            if ant.inventory > 0:
                ant.inventory -= 1
                ant.energy = min(self.start_energy, ant.energy + self.food_energy)
                ev["eaten"] = 1
            elif ant.pos == self.colony_pos and self.colony_food > 0:
                self.colony_food -= 1
                ant.energy = min(self.start_energy, ant.energy + self.food_energy)
                ev["eaten"] = 1
            else:
                ev["invalid"] = 1
 
        return ev

    def deliver(self, ant: Ant) -> int:
        if ant.pos != self.colony_pos or ant.inventory == 0:
            return 0
        n = ant.inventory
        self.colony_food += n
        ant.inventory = 0
        return n

    def metabolize(self, ant: Ant) -> None:
        """Hunger applies EVERY step (idling is not free)."""
        ant.energy = max(0.0, ant.energy - self.energy_per_step)
        if ant.energy == 0:
            ant.health -= self.starvation_damage
        if ant.health <= 0:
            ant.health = 0
            ant.alive = False
 
    def respawn_food(self) -> None:
        if self.np_random.random() < self.food_respawn_prob:
            cells = self.empty_cells()
            if len(cells) > 0:
                y, x = cells[self.np_random.integers(len(cells))]
                self.world[y, x] = FOOD

    def occupied(self, x: int, y: int, exclude: Ant) -> bool:
        if (x, y) == self.colony_pos:  # many ants may stand on the colony
            return False
        return any(a.alive and a is not exclude and a.pos == (x, y) for a in self.ants)
 
    def empty_cells(self) -> np.ndarray:
        mask = self.world == EMPTY
        for a in self.ants:
            if a.alive:
                mask[a.y, a.x] = False
        return np.argwhere(mask)  # rows of (y, x)
 
    def random_empty_cell(self) -> tuple[int, int]:
        cells = self.empty_cells()
        if len(cells) == 0:
            raise RuntimeError("No empty cells available.")
        y, x = cells[self.np_random.integers(len(cells))]
        return int(x), int(y)

    def observe_all(self) -> np.ndarray:
        return np.stack([self.observe(a) for a in self.ants])   # (n_ants, obs_dim)

    def observe(self, ant: Ant) -> np.ndarray:
        vs, r = self.vision_size, self.vision_radius
        vision = np.zeros((N_CHANNELS, vs, vs), dtype=np.float32)
        others = {a.pos for a in self.ants if a.alive and a.id != ant.id}

        for ly in range(vs):
            for lx in range(vs):
                wx, wy = ant.x + lx - r, ant.y + ly - r
                if not (0 <= wx < self.grid_size and 0 <= wy < self.grid_size):
                    vision[CH_WALL, ly, lx] = 1.0
                    continue
                cell = self.world[wy, wx]
                if cell == FOOD:
                    vision[CH_FOOD, ly, lx] = 1.0
                elif cell == TREE:
                    vision[CH_TREE, ly, lx] = 1.0
                elif cell == COLONY:
                    vision[CH_COLONY, ly, lx] = 1.0
                if (wx, wy) in others:
                    vision[CH_ANT, ly, lx] = 1.0

        cx, cy = self.colony_pos
        norm = max(1, self.grid_size - 1)
        scalars = np.array([
            ant.energy / self.start_energy,
            ant.health / self.start_health,
            ant.inventory / self.inventory_capacity,
            (cx - ant.x) / norm,  # signed direction to colony, in [-1, 1]
            (cy - ant.y) / norm],

            dtype=np.float32,
            )
        
        return np.concatenate([vision.ravel(), scalars]).astype(np.float32)

    def info(self) -> dict:
        alive = self.ants
        return {
            "step": self.step_count,
            "colony_food": self.colony_food,
            "mean_energy": float(np.mean([a.energy for a in alive])),
            "mean_health": float(np.mean([a.health for a in alive])),
            **{f"total_{k}": v for k, v in self.stats.items()},
        }

    def render(self):
        if self.render_mode is None:
            return None

        if self.render_mode == "rgb_array":
            return self._render_rgb()

        if self.render_mode in ("ansi", "human"):
            text = self._render_ansi()
            if self.render_mode == "human":
                print(text)
                return None
            return text

        raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def _render_rgb(self) -> np.ndarray:
        cell = 40
        colors = {
            EMPTY:  (240, 240, 240),
            FOOD:   (220, 60, 60),
            TREE:   (30, 120, 40),
            COLONY: (150, 100, 50),
        }
        size = self.grid_size * cell
        frame = np.zeros((size, size, 3), dtype=np.uint8)

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                frame[y*cell:(y+1)*cell, x*cell:(x+1)*cell] = colors[int(self.world[y, x])]

        # grid lines
        frame[::cell, :] = 0
        frame[:, ::cell] = 0

        # ants: blue square, darker when carrying food
        margin = 8
        for a in self.ants:
            if not a.alive:
                continue
            color = (30, 60, 200) if a.inventory == 0 else (10, 20, 100)
            y0, x0 = a.y * cell, a.x * cell
            frame[y0+margin:y0+cell-margin, x0+margin:x0+cell-margin] = color

        return frame

    def _render_ansi(self) -> str:
        symbols = {EMPTY: "· ", FOOD: "🍎", TREE: "🌲", COLONY: "🏠"}
        ant_at = {a.pos for a in self.ants if a.alive}
        rows = []
        for y in range(self.grid_size):
            rows.append("".join(
                "🐜" if (x, y) in ant_at else symbols[int(self.world[y, x])]
                for x in range(self.grid_size)
            ))
        return "\n".join(rows)




if __name__ == "__main__":
    env = ColonyEnv(render_mode="ansi")
    obs, info = env.reset(seed=42)
    total, done = 0.0, False
    while not done:
        actions = env.np_random.integers(5, size=env.n_ants)
        obs, rewards, terminated, truncated, info = env.step(actions)
        total += rewards.mean()
        done = truncated
    print(env.render())
    print(f"steps={info['step']}  mean return per ant={total:.2f}  info={info}")