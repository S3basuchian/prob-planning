import time
from typing import Optional

import gymnasium as gym
import numpy as np

from env.dynamics import GardenerDynamics, _action_to_direction
from env.rendering import GardenerRenderer
from env.state import GardenerState


class GardenerEnv(gym.Env):

    def __init__(self, size: int = 15, grass_respawn: int = 50, lake_respawn: int = 20):
        # The size of the square grid (5x5 by default)

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._state = GardenerState()
        self._state.size = size
        self._state.grass_respawn = grass_respawn
        self._state.lake_respawn = lake_respawn
        num_frogs = max(1, int(size * size * 0.01))
        num_lakes = max(1, int(size * size * 0.02))
        num_grass = max(1, int(size * size * 0.04))
        num_walls = int(size * size * 0.30)
        self._state.walls = np.full((num_walls, 2), -1, dtype=int)

        self._state.agent = np.array([-1, -1], dtype=int)
        self._state.frogs = np.full((num_frogs, 2), -1, dtype=int)
        self._state.lakes = np.full((num_lakes, 2), -1, dtype=int)
        self._state.lakes_full = np.ones(num_lakes, dtype=bool)
        self._state.dead_frogs = np.zeros(num_frogs, dtype=bool)
        self._state.capt_frogs = np.zeros(num_frogs, dtype=bool)
        self._state.lake_timer = np.zeros(num_lakes, dtype=int)
        self._state.frog_timer = np.zeros(num_frogs, dtype=int)
        self._state.grass = np.full((num_grass, 2), -1, dtype=int)
        # Each grass patch starts active. A timer is used for reactivation.
        self._state.grass_active = np.ones(num_grass, dtype=bool)
        self._state.grass_timer = np.zeros(num_grass, dtype=int)
        self._state.score = 0

        # Define what actions are available (4 directions + 1 do nothing)
        self.action_space = gym.spaces.Discrete(5)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {"agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
             # [x, y] coordinates
             "frogs": gym.spaces.Box(0, size - 1, shape=(num_frogs, 2),
                                     dtype=int),  # array of [x, y] coordinates
             "lakes": gym.spaces.Box(0, size - 1, shape=(num_lakes, 2),
                                     dtype=int),  # array of [x, y] coordinates
             "size": gym.spaces.Discrete(size + 1),
             "grass_respawn": gym.spaces.Discrete(grass_respawn + 1),
             "lake_respawn": gym.spaces.Discrete(lake_respawn + 1),
             "lakes_full": gym.spaces.Box(0, 1, shape=(num_lakes,),
                                          dtype=bool),
             "dead_frogs": gym.spaces.Box(0, 1, shape=(num_frogs,),
                                          dtype=bool),
             "capt_frogs": gym.spaces.Box(0, 1, shape=(num_frogs,),
                                          dtype=bool),
             "lake_timer": gym.spaces.Box(0, lake_respawn, shape=(num_lakes,), dtype=int),
             "frog_timer": gym.spaces.Box(0, 5, shape=(num_frogs,), dtype=int),
             "grass": gym.spaces.Box(0, size - 1, shape=(num_grass, 2),
                                     dtype=int),
             "grass_active": gym.spaces.Box(0, 1, shape=(num_grass,), dtype=bool),
             "grass_timer": gym.spaces.Box(0, grass_respawn, shape=(num_grass,), dtype=int),
             "walls": gym.spaces.Box(0, size - 1, shape=(num_walls, 2),
                                     dtype=int),
             "pos_actions": gym.spaces.Box(0, 1, shape=(size, size, 5), dtype=int),
             "lake_dist": gym.spaces.Box(0, np.iinfo(np.int32).max, shape=(num_lakes, size, size), dtype=int),
             "grass_dist": gym.spaces.Box(0, np.iinfo(np.int32).max, shape=(num_grass, size, size), dtype=int),
             })


        self._renderer = GardenerRenderer()
        self._dynamics = GardenerDynamics(self._np_random_seed)

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent, target and frog positions
        """
        return {"agent": self._state.agent,
                "frogs": self._state.frogs,
                "size": self._state.size,
                "lake_respawn": self._state.lake_respawn,
                "grass_respawn": self._state.grass_respawn,
                "lakes": self._state.lakes,
                "lakes_full": self._state.lakes_full,
                "dead_frogs": self._state.dead_frogs,
                "capt_frogs": self._state.capt_frogs,
                "lake_timer": self._state.lake_timer,
                "frog_timer": self._state.frog_timer,
                "grass": self._state.grass,
                "grass_active": self._state.grass_active,
                "grass_timer": self._state.grass_timer,
                "walls": self._state.walls,
                "pos_actions": self._state.pos_actions,
                "lake_dist": np.array(self._state.lake_dist, dtype=int),
                "grass_dist": np.array(self._state.grass_dist, dtype=int)}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {"distance": np.linalg.norm(
            self._state.agent - self._state.grass[self._state.grass_active.argmax()],
            ord=1)}

    def reset(self, seed: Optional[int] = None,
              options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        self._dynamics = GardenerDynamics(seed=seed)

        self._state.score = 0

        # Randomly place the agent anywhere on the grid
        self._state.agent = self.np_random.integers(0, self._state.size,
                                                    size=2,
                                                    dtype=int)

        # Place frogs randomly on the grid, avoiding agent
        all_positions = {(x, y) for x in range(self._state.size) for y in
                         range(self._state.size)}
        all_positions.discard(tuple(self._state.agent))
        frog_positions = self.np_random.choice(list(all_positions),
                                               size=len(self._state.frogs),
                                               replace=False)
        for frog_pos in frog_positions:
            all_positions.discard(tuple(frog_pos))
        lake_positions = self.np_random.choice(list(all_positions),
                                               size=len(self._state.lakes),
                                               replace=False)
        #easy
        #lake_positions = []
        #while len(lake_positions) < len(self._state.lakes):
        #    lake_position = self.np_random.choice(list(all_positions),
        #                                          size=1,
        #                                          replace=False)
        #    for lake_pos in lake_position:
        #        lake_positions.append(tuple(lake_pos))
        #        all_positions.discard(tuple(lake_pos))
        #        lx, ly = lake_pos
        #        neighbors = [
        #            (lx + 1, ly), (lx - 1, ly), (lx + 1, ly + 1),
        #            (lx - 1, ly - 1),
        #            (lx, ly + 1), (lx, ly - 1), (lx - 1, ly + 1),
        #            (lx + 1, ly - 1)
        #        ]
        #        for nx, ny in neighbors:
        #            if 0 <= nx < self._state.size and 0 <= ny < self._state.size:
        #                all_positions.discard((nx, ny))

        for lake_pos in lake_positions:
            all_positions.discard(tuple(lake_pos))
            lx, ly = lake_pos
            neighbors = [
                (lx + 1, ly), (lx - 1, ly),
                (lx, ly + 1), (lx, ly - 1)
            ]
            for nx, ny in neighbors:
                if 0 <= nx < self._state.size and 0 <= ny < self._state.size:
                    all_positions.discard((nx, ny))
        grass_positions = self.np_random.choice(list(all_positions),
                                                size=len(self._state.grass),
                                                replace=False)


        for grass_pos in grass_positions:
            all_positions.discard(tuple(grass_pos))

        # Generate walls ensuring accessibility of all non-lake, non-wall cells
        def is_accessible(excluded):
            # BFS over free cells
            free = {(x, y) for x in range(self._state.size)
                    for y in range(self._state.size)}
            free -= set(map(tuple, lake_positions))
            free -= set(excluded)
            if not free:
                return True
            start = next(iter(free))
            stack = [start]
            visited = set([start])
            while stack:
                cx, cy = stack.pop()
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in free and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        stack.append((nx, ny))
            return visited == free

        wall_positions = []
        remaining_positions = list(all_positions)
        self.np_random.shuffle(remaining_positions)
        for pos in remaining_positions:
            if len(wall_positions) == len(self._state.walls):
                break
            trial = wall_positions + [tuple(pos)]
            if is_accessible(trial):
                wall_positions.append(tuple(pos))
        if len(wall_positions) < len(self._state.walls):
            return self.reset(seed=seed)
        self._state.walls = np.array(wall_positions, dtype=int)
        self._state.lakes = np.array(lake_positions, dtype=int)
        self._state.frogs = np.array(frog_positions, dtype=int)
        self._state.grass = np.array(grass_positions, dtype=int)

        #print("free candidates:", len(remaining_positions))
        #print("accepted:", len(wall_positions))

        # -------------------------------------------------------------
        # Precompute shortest-path distance and best-step fields
        # -------------------------------------------------------------
        size = self._state.size
        walls_set = {tuple(w) for w in self._state.walls}
        lakes_set = {tuple(w) for w in self._state.lakes}
        lake_dist = []
        lake_best_step = []

        self._state.pos_actions = np.zeros((size, size, 5), dtype=int)
        for c in range(size):
            for r in range(size):
                is_wall = np.any(np.all(self._state.walls == [c, r], axis=1))
                is_lake = np.any(np.all(self._state.lakes == [c, r], axis=1))
                if not is_wall and not is_lake:
                    mask = np.ones(len(_action_to_direction), dtype=np.int8)
                    x, y = (c,r)
                    # Prevent moves that leave the grid
                    if x == self._state.size - 1: mask[0] = 0
                    if y == self._state.size - 1: mask[1] = 0
                    if x == 0: mask[2] = 0
                    if y == 0: mask[3] = 0
                    # Prevent moves that would step onto a lake
                    for action, direction in _action_to_direction.items():
                        nx, ny = (c,r) + direction
                        if any((nx == lx and ny == ly) for lx, ly in
                               self._state.lakes):
                            mask[action] = 0
                        if any((nx == wx and ny == wy) for wx, wy in
                               self._state.walls):
                            mask[action] = 0
                    for i in range(len(mask)):
                        if mask[i] == 1:
                            self._state.pos_actions[c, r, i] = 1

        from collections import deque

        for (lx, ly) in lake_positions:
            dist = np.full((size, size), np.iinfo(np.int32).max, dtype=np.int32)
            best = np.zeros((size, size, 2), dtype=np.int8)

            q = deque()
            q.append((lx, ly))
            dist[lx, ly] = 0

            while q:
                x, y = q.popleft()
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        if (nx, ny) in walls_set:
                            continue
                        # Also avoid other lakes
                        if (nx, ny) in lakes_set and (nx, ny) != (lx, ly):
                            continue

                        if dist[nx, ny] > dist[x, y] + 1:
                            dist[nx, ny] = dist[x, y] + 1
                            if dist[nx, ny] == 1:
                                best[nx, ny] = np.array([0, 0], dtype=np.int8)
                            else:
                                best[nx, ny] = np.array([-dx, -dy], dtype=np.int8)
                            q.append((nx, ny))

            lake_dist.append(dist)
            lake_best_step.append(best)


        self._state.lake_dict = {}
        for c in range(size):
            for r in range(size):
                if (c, r) in walls_set or (c, r) in lakes_set:
                    continue
                lakes_info = []
                for i in range(len(self._state.lakes)):
                    dist = lake_dist[i][c, r]
                    step = lake_best_step[i][c, r]

                    action = 4
                    if step[0] == 1 and step[1] == 0:
                        action = 0
                    elif step[0] == 0 and step[1] == 1:
                        action = 1
                    elif step[0] == -1 and step[1] == 0:
                        action = 2
                    elif step[0] == 0 and step[1] == -1:
                        action = 3

                    lakes_info.append((i, dist, action))

                lakes_info.sort(key=lambda x: x[1])
                self._state.lake_dict[(c, r)] = lakes_info

        self._state.lake_dist = lake_dist
        self._state.lake_best_step = lake_best_step

        # Precompute grass distances
        grass_dist = []
        for (gx, gy) in grass_positions:
            dist = np.full((size, size), np.iinfo(np.int32).max, dtype=np.int32)
            q = deque()
            q.append((gx, gy))
            dist[gx, gy] = 0
            
            while q:
                x, y = q.popleft()
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        if (nx, ny) in walls_set:
                            continue
                        if (nx, ny) in lakes_set:
                            continue
                        if dist[nx, ny] > dist[x, y] + 1:
                            dist[nx, ny] = dist[x, y] + 1
                            q.append((nx, ny))
            grass_dist.append(dist)
        self._state.grass_dist = grass_dist


        # np_random.choice returns a 1D array if input is 1D, so convert to
        # 2D array of positions
        self._state.lake_timer = np.ones(len(self._state.lakes), dtype=int)
        self._state.frog_timer = np.zeros(len(self._state.frogs), dtype=int)
        self._state.grass_active[:] = True
        self._state.grass_timer[:] = 0

        observation = self._get_obs()
        info = self._get_info()

        # Save screenshot of initial configuration
        if options and options.get("save_screenshot"):
            self._renderer.draw(self._state)
            self._renderer.save_screenshot(options.get("screenshot_path", "initial_config.png"))

        return observation, info


    def simulate_samples(self, horizon, q_agent, actions):
        # simulate the environment for a certain horizon
        # return FALSE if the simulation violates the a norm
        executed_actions = []
        state = self._state.fast_clone()
        for h in range(horizon):
            if len(actions) > h:
                action = actions[h]
            else:
                action = q_agent.getAction(state)
            self._dynamics.move_agent(state, action)
            self._dynamics.move_frogs(state)
            self.update_env(state, 0 )
            executed_actions.append(action)
            #if np.any(np.all(state.agent == state.frogs, axis=1)):
            #    return False, executed_actions
        return True, executed_actions


    def sample(self, horizon, size):
        # create a copy of current random variable that does not influence og
        np_random = np.random.Generator(self.np_random.bit_generator.jumped())

        samples = []
        start_time = time.time()

        for i in range(size):
            # deep copy of full environment state
            state = self._state.fast_clone()
            world = [state.fast_clone()]
            for h in range(horizon):
                self._dynamics.move_frogs(state)
                world.append(state.fast_clone())
            samples.append(world)

        elapsed = time.time() - start_time
        print(f"sample() took {elapsed:.6f} seconds for size={size}, horizon={horizon}")

        return samples

    def check_violations(self, actions, samples, q_agent):
        frogs_killed_rl = 0
        frogs_killed_asp = 0
        state_rl = self._state.fast_clone()
        state_asp = self._state.fast_clone()

        positions_asp = [state_asp.agent]
        positions_rl = [state_rl.agent]
        for i in range(len(actions)):
            action_rl = q_agent.getAction(state_rl)
            action_asp = actions[i]
            self._dynamics.move_agent(state_rl, action_rl)
            self._dynamics.move_agent(state_asp, action_asp)
            positions_asp.append(state_asp.agent)
            positions_rl.append(state_rl.agent)

        for world in samples:
            killed_frog_rl = False
            killed_frog_asp = False
            for i, state in enumerate(world):
                if np.any(np.all(positions_rl[i] == state.frogs, axis=1)):
                    killed_frog_rl = True
                if np.any(np.all(positions_asp[i] == state.frogs, axis=1)):
                    killed_frog_asp = True
            if killed_frog_rl:
                frogs_killed_rl += 1
            if killed_frog_asp:
                frogs_killed_asp += 1

        print(f"Average number of frog kills (ASP): {frogs_killed_asp / len(samples):.2f}")
        print(f"Average number of frog kills (RL): {frogs_killed_rl / len(samples):.2f}")



    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        reward = 0

        grass_patch = self._dynamics.move_agent(self._state, action)

        self._dynamics.move_frogs(self._state)


        if np.any(np.all(self._state.agent == self._state.frogs, axis=1)):
            for i,(c,r) in enumerate(self._state.frogs):
                if self._state.dead_frogs[i]: continue
                if self._state.agent[0] == c and self._state.agent[1] == r:
                    self._state.dead_frogs[i] = True
                    msg = "FROG KILLED!"
                    #print(f"\033[31m{msg}\033[0m")


        reward = self.update_env(self._state, reward)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        self._state.score += reward

        observation = self._get_obs()
        info = self._get_info()

        # Terminate if point limit reached
        terminated = True if self._state.score >= 300 else False

        return observation, reward, terminated, truncated, info

    def update_env(self, state, reward: int) -> int:
        for f, (c, r) in enumerate(state.frogs):
            if state.frog_timer[f] > 0:
                state.frog_timer[f] = state.frog_timer[f] - 1
        # Update grass states
        for i, (gx, gy) in enumerate(state.grass):
            ax, ay = state.agent
            if ax == gx and ay == gy:
                if state.grass_active[i]:
                    state.grass_active[i] = False
                    reward += 10
                    state.grass_timer[i] = state.grass_respawn
            else:
                if not state.grass_active[i] and state.grass_timer[
                    i] > 0:
                    state.grass_timer[i] -= 1
                    if state.grass_timer[i] == 0:
                        state.grass_active[i] = True

        # Update lake states based on frog adjacency
        for i, (lx, ly) in enumerate(state.lakes):
            # Decrease timer if running
            if state.lake_timer[i] > 0:
                state.lake_timer[i] -= 1
                if state.lake_timer[i] == 0:
                    state.lakes_full[i] = True  # refill lake

            # Check adjacency to any frog (Manhattan distance 1)
            # for fx, fy in state.frogs:
            #    if abs(fx - lx) + abs(fy - ly) == 1 and state.lakes_full[i]:
            #        state.lakes_full[i] = False
            #        state.lake_timer[i] = 20
            #        break

            # Check adjacency to the agent (Manhattan distance 1)
            ax, ay = state.agent
            if state.lakes_full[i] and abs(ax - lx) + abs(ay - ly) == 1:
                # Additional reward for being adjacent (Manhattan distance 1) to any full lake
                reward += 5
                state.lakes_full[i] = False
                state.lake_timer[i] = state.lake_respawn
                for f,(c,r) in enumerate(state.frogs):
                    prox = False
                    if not state.dead_frogs[f]:
                        if abs(lx - c) + abs(
                                ly - r) == 1:
                            prox = True
                        elif abs(lx - c) + abs(
                                ly - r) == 2 and abs(
                            lx - c) == 1:
                            prox = True
                        elif abs(lx - c) + abs(
                                ly - r) == 2 and abs(
                            ly - r) == 1:
                            prox = True
                    if prox:
                        state.frog_timer[f] = 5
        return reward

    def render(self):
        self._renderer.draw(self._state)
