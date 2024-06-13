"""Example of using the MuJoCo Python viewer with for offscreen cameras without the MuJoCo's Python Render wrapper"""

import mujoco
import mujoco.viewer as viewer
from mujoco.renderer import Renderer
import numpy as np
import cv2

def fixation_control(m, d, renderer):
  try:
    # Render the simulated camera
    # mujoco.mjv_updateScene(m, d, vopt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    # mujoco.mjr_render(viewport, scn, ctx)
    # image = np.empty((RES_Y, RES_X, 3), dtype=np.uint8)
    # mujoco.mjr_readPixels(image, None, viewport, ctx)
    # image = cv2.flip(image, 0) # OpenGL renders with inverted y axis
    renderer.update_scene(d, camera=0)
    image = renderer.render()

    # Show the simulated camera image
    cv2.imshow('fixation', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

    # Threshold image and use median of detection pixels as center of target
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    thresholded_image = (  (image_hsv[:, :, 0] > 115) & (image_hsv[:, :, 0] <= 130)
                         & (image_hsv[:, :, 1] > 150) & (image_hsv[:, :, 1] <= 255)
                         & (image_hsv[:, :, 2] > 100) & (image_hsv[:, :, 2] <= 200))
    target_detections = np.where(thresholded_image)
    x = np.median(target_detections[1])
    y = np.median(target_detections[0])

    # Distance of target center from center of image normalized
    dx = (x - RES_X / 2) / (RES_X / 2)
    dy = (y - RES_Y / 2) / (RES_Y / 2)

    # Set actuator velocities
    shoulder_v_gain = 5.0
    d.actuator('shoulderv').ctrl[0] = -shoulder_v_gain * np.arctan(dx)
    elbow_v_gain = 5.0
    d.actuator('elbowv').ctrl[0] = elbow_v_gain * -np.arctan(dy)

  except Exception as e:
    print(e)
    raise e

def load_callback(m=None, d=None):
  # Clear the control callback before loading a new model
  # or a Python exception is raised
  mujoco.set_mjcb_control(None)

  m = mujoco.MjModel.from_xml_path('./fixation.xml')
  d = mujoco.MjData(m)

  if m is not None:
    # Make the windmill spin
    d.joint('windmillrotor').qvel = 1

    # Make sure the following is in the xml:
    #   <visual>
    #     <global  offwidth="1280" offheight="720"/>
    #   </visual>
    renderer = Renderer(m, width=1280, height=720)

    # Set the callback and capture all variables needed for rendering
    mujoco.set_mjcb_control(
      lambda m, d: fixation_control(
        m, d, renderer))

  return m , d

if __name__ == '__main__':
  viewer.launch(loader=load_callback)
