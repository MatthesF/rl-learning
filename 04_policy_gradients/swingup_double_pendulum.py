"""Swing-up double pendulum on a cart (MuJoCo).

Starts hanging, must swing up and balance. Importable so AsyncVectorEnv can pickle it.
"""

from __future__ import annotations

import numpy as np
from gymnasium.envs.mujoco.inverted_double_pendulum_v4 import InvertedDoublePendulumEnv

CART_LIMIT = 0.9


class SwingUpDoublePendulum(InvertedDoublePendulumEnv):
    """Start at bottom; exponential angle/height bonus + upright streak bonus."""

    UPRIGHT_Y = 1.0
    ANGLE_BASE = 1.08
    ANGLE_COEF = 2.0
    ANGLE_SCALE = 40.0
    STREAK_BASE = 1.05
    STREAK_COEF = 0.5
    STREAK_CAP = 120
    Y_HANG = -1.2
    Y_UP = 1.2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.upright_streak = 0

    def reset_model(self):
        self.upright_streak = 0
        qpos = self.init_qpos.copy()
        qpos[0] = self.np_random.uniform(-0.05, 0.05)
        qpos[1] = np.pi + self.np_random.uniform(-0.1, 0.1)
        qpos[2] = self.np_random.uniform(-0.1, 0.1)
        qvel = self.np_random.uniform(-0.05, 0.05, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.data.site_xpos[0]
        theta1 = float(self.data.qpos[1])
        theta2 = float(self.data.qpos[2])

        dist_penalty = 0.01 * x**2 + (y - 2.0) ** 2
        v1, v2 = self.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        reward = 10.0 - dist_penalty - vel_penalty

        angle_up = 0.5 * (np.cos(theta1) + 1.0)
        angle_up2 = 0.5 * (np.cos(theta1 + theta2) + 1.0)
        height_up = np.clip((y - self.Y_HANG) / (self.Y_UP - self.Y_HANG), 0.0, 1.0)
        closeness = 0.4 * angle_up + 0.3 * angle_up2 + 0.3 * height_up
        reward += self.ANGLE_COEF * (self.ANGLE_BASE ** (closeness * self.ANGLE_SCALE) - 1.0)

        if y > self.UPRIGHT_Y:
            self.upright_streak += 1
            capped = min(self.upright_streak, self.STREAK_CAP)
            reward += self.STREAK_COEF * (self.STREAK_BASE ** capped - 1.0)
        else:
            self.upright_streak = 0

        terminated = bool(abs(x) > CART_LIMIT)
        if self.render_mode == "human":
            self.render()
        return ob, float(reward), terminated, False, {}
