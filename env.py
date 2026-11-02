# ---------------------------------------------------------
# env.py
# The robot's "game". PPO calls reset() and step() on it.
# Test it with:  mjpython env.py
# ---------------------------------------------------------

import numpy as np                      # numpy = tools for lists of numbers
import gymnasium as gym                 # gym = the rulebook PPO expects
from gymnasium import spaces            # spaces = describes our numbers
import mujoco                           # mujoco = the physics engine


class HumanoidStandEnv(gym.Env):

    # -----------------------------------------------------
    # __init__ runs ONE time, when the game is made.
    # It loads the robot and sets up the rules.
    # -----------------------------------------------------
    def __init__(self):
        # The model is the robot's blueprint. It never changes.
        self.model = mujoco.MjModel.from_xml_path("humanoid.xml")

        # The data is the robot right now. It changes every step.
        self.data = mujoco.MjData(self.model)

        # Find the torso by name so we can check its height later.
        self.torso = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )

        # Find the knee joints. We look for any joint with
        # "knee" in its name, then remember where its angle lives
        # in qpos. We use these to punish bent knees.
        self.knee_addrs = []
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name is not None and "knee" in name:
                self.knee_addrs.append(self.model.jnt_qposadr[j])
        print("knee joints found:", len(self.knee_addrs))  # should be 2

        # Find the feet bodies, so we can check if they touch
        # the ground. We look for bodies with "foot" in the name.
        self.foot_ids = []
        for b in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            if name is not None and "foot" in name:
                self.foot_ids.append(b)
        print("foot bodies found:", len(self.foot_ids))    # should be 2

        self.frame_skip = 5      # physics ticks per action
        self.max_steps = 500     # how long one try lasts
        self.steps = 0           # counts steps in this try

        # ACTIONS (the output): one number per motor, from -1 to 1.
        self.action_space = spaces.Box(-1.0, 1.0, (self.model.nu,), np.float32)

        # OBSERVATIONS (the input): what the brain can see.
        # We skip x and y, so the brain does not care where
        # on the floor the robot is.
        obs_dim = self.model.nq - 2 + self.model.nv
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)

    # -----------------------------------------------------
    # _obs packs the robot's state into one list of numbers.
    # -----------------------------------------------------
    def _obs(self):
        return np.concatenate([self.data.qpos[2:], self.data.qvel]).astype(np.float32)

    # -----------------------------------------------------
    # reset runs at the START of every try.
    # It puts the robot back on the floor.
    # -----------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)            # needed by gym, ignore it

        mujoco.mj_resetData(self.model, self.data)

        # Torso height. 0.25 means close to the ground.
        self.data.qpos[2] = 0.25

        # Torso rotation. These 4 numbers put it on its back.
        self.data.qpos[3:7] = [0.7071, 0.0, 0.7071, 0.0]

        # Tiny random wiggle on the joints.
        self.data.qpos[7:] += self.np_random.uniform(-0.05, 0.05, self.model.nq - 7)

        mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        return self._obs(), {}

    # -----------------------------------------------------
    # step runs OVER AND OVER. It is the main loop.
    # -----------------------------------------------------
    def step(self, action):
        # 1. Give the brain's commands to the motors.
        self.data.ctrl[:] = action

        # 2. Run physics a few times so each command lasts a bit.
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        # 3. SCORING.

        # How high is the torso? Biggest weight (x3).
        # Height is the main thing we want, so it pays the most.
        height = self.data.xpos[self.torso][2]
        height_reward = 3.0 * height

        # How straight up is the torso? 1 = straight, -1 = upside down.
        # Smaller weight (x0.5), so it matters less than height.
        upright = self.data.xmat[self.torso].reshape(3, 3)[2, 2]
        upright_reward = 0.5 * upright

        # Bent knees are punished.
        # A straight knee has an angle near 0, so its penalty is near 0.
        # abs() makes bending either way count as bad.
        knee_bend = 0.0
        for addr in self.knee_addrs:
            knee_bend += abs(self.data.qpos[addr])
        knee_penalty = 0.5 * knee_bend

        # Feet on the ground.
        # We look at every contact the physics found this step.
        # If a contact involves a foot body, that foot is touching something.
        feet_touching = 0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if body1 in self.foot_ids or body2 in self.foot_ids:
                feet_touching += 1
        feet_reward = 0.2 * min(feet_touching, 2)   # cap at 2 feet

        # Torso should sit directly above the feet.
        # Take the x,y position of the torso and the average x,y of the feet.
        # The farther apart they are (looking from above), the bigger the penalty.
        feet_xy = np.mean([self.data.xpos[b][:2] for b in self.foot_ids], axis=0)
        torso_xy = self.data.xpos[self.torso][:2]
        lean_dist = np.linalg.norm(torso_xy - feet_xy)
        lean_penalty = 2.0 * lean_dist

        # Small penalty for working the motors too hard.
        # (Squaring makes big commands cost much more than small ones.)
        effort_penalty = 0.005 * np.sum(np.square(action))

        # Add it all up.
        reward = (
            height_reward
            + upright_reward
            + feet_reward
            - knee_penalty
            - effort_penalty
            - lean_penalty
        )

        # The try ends when we hit the max number of steps.
        truncated = self.steps >= self.max_steps

        # Returns: new state, score, failed?, time up?, extra info
        return self._obs(), float(reward), False, truncated, {}


# ---------------------------------------------------------
# Quick test. Only runs when you type:  mjpython env.py
# ---------------------------------------------------------
if __name__ == "__main__":
    env = HumanoidStandEnv()
    obs, _ = env.reset()
    for _ in range(100):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
    print("env works, last reward:", r)