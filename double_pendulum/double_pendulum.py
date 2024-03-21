"""Example of using the MuJoCo Python viewer UI for a simple control task"""

import mujoco
import mujoco.viewer as viewer
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

def load_callback(m=None, d=None):
  # Clear the control callback before loading a new model
  # or a Python exception is raised
  mujoco.set_mjcb_control(None)

  m = mujoco.MjModel.from_xml_path('./double_pendulum.xml')
  d = mujoco.MjData(m)

  if m is not None:
    # Set some initial conditions
    d.joint('shoulder').qpos = 0.1
    d.joint('elbow').qpos = 0.1
    # Set the control callback
    K = double_pendulum_lqr_K()
    mujoco.set_mjcb_control(lambda m, d: double_pendulum_control(m, d, K))

  return m, d

if __name__ == '__main__':
  viewer.launch(loader=load_callback)
