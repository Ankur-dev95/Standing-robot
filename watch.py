# ---------------------------------------------------------
# watch.py
# Opens the MuJoCo window and shows the robot.
# Run with:  mjpython watch.py
# Works while train.py is still running, using a checkpoint.
# ---------------------------------------------------------

import time                             # lets us slow things down
import mujoco.viewer                    # the window that shows the robot
from stable_baselines3 import PPO       # needed to load the saved brain
from env import HumanoidStandEnv        # the robot's game, from env.py

# Make the game, same as in training.
env = HumanoidStandEnv()

# Load a brain. Only ONE line should be active (no # in front).
# A checkpoint, to watch mid-training. Change the number to the
# newest file you see when you run:  ls checkpoints
model = PPO.load("stand_ppo")
# The final brain, only exists after train.py finishes:
# model = PPO.load("stand_ppo")

# Put the robot on the floor to start.
# "obs" is what the robot can see right now.
obs, _ = env.reset()

# Open the window. Keep going until you close it.
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():

        # 1. Ask the brain: "What should the motors do?"
        # deterministic=True means: use its best guess, no random tries.
        action, _ = model.predict(obs, deterministic=True)

        # 2. Do that action. Physics runs and we get the new state.
        # We only need the new "obs". The rest we ignore.
        obs, reward, terminated, truncated, _ = env.step(action)

        # 3. If the try is over, put the robot back on the floor.
        if truncated:
            obs, _ = env.reset()

        # 4. Redraw the window.
        viewer.sync()

        # 5. Wait a tiny bit so it does not run too fast to see.
        time.sleep(0.01)