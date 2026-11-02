import time
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

model = mujoco.MjModel.from_xml_path("humanoid.xml")
data = mujoco.MjData(model) # holds postions or qpos  and q vels (velocities) and xpos is the postion of bodies in the world frame
torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[:] = np.random.uniform(-1.0, 1.0, size=model.nu) # model.nu is tthe number of moters,
        # data.ctrl is moter controls, aka the numbers each moter needs to push or troque
        # this fills all the motor contorls forces with random numbers  each between -1 and 1

        
        
        mujoco.mj_step(model, data)
        height = data.xpos[torso_id][2]
        reward = height 
        
        

        viewer.sync()
        time.sleep(model.opt.timestep)