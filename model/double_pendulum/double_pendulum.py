# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example of using python simulate UI for a double pendulum"""

import mujoco
from mujoco.simulate import run_simulate_and_physics
import numpy as np
from scipy.linalg import solve_continuous_are

# Find a K matrix for a linearized double pendulum using LQR
def double_pendulum_lqr_K():
  # System linearization calculated using "derivative" sample
  A = np.array([[0, 0, 1, 0],
                [0, 0, 0, 1],
                [ 2.494531E+01, -1.565526E+01, -2.130544E-01,  5.164733E-01],
                [-3.252928E+01,  6.589209E+01,  5.164733E-01, -1.793546E+00],
               ])
  B = np.array([[0], [0], [2.130544E+00], [-5.164733E+00]])

  Q = 1 * np.array([[10,  0,  0,   0],
                    [0, 100,  0,   0],
                    [0,   0, 10,   0],
                    [0,   0,  0, 100]])
  R = 1 * np.eye(1)

  S = solve_continuous_are(A, B, Q, R)

  K = np.linalg.inv(R)@(B.T @ S)
  return K

def double_pendulum_control(m, d, K):
  x = np.concatenate((
        d.joint('shoulder').qpos,
        d.joint('elbow').qpos,
        d.joint('shoulder').qvel,
        d.joint('elbow').qvel))

  u = -K @ x
  d.actuator('shoulder').ctrl[0] = u

def preload_callback(m, d):
  # Have to clear this callback before loading a model
  mujoco.set_mjcb_control(None)

def load_callback(m, d, loadError):
  if m is not None:
    # Set some initial conditions
    d.joint('shoulder').qpos = 0.1
    d.joint('elbow').qpos = 0.1
    # Set the callback
    mujoco.set_mjcb_control(lambda m, d: double_pendulum_control(m, d, K))

if __name__ == '__main__':
  K = double_pendulum_lqr_K()
  run_simulate_and_physics('./double_pendulum.xml', preload_callback, load_callback)
