# ---------------------------------------------------------
# train.py
# Teaches the robot's brain. Run with:  mjpython train.py
# No window opens. It just trains and prints numbers.
# ---------------------------------------------------------

from stable_baselines3 import PPO                            # the learning method
from stable_baselines3.common.callbacks import CheckpointCallback  # saves copies while training
from env import HumanoidStandEnv                             # the robot's game, from env.py

# Only run the code below when I type "mjpython train.py".
if __name__ == "__main__":

    # Make the game. This loads the robot.
    env = HumanoidStandEnv()

    # Make the brain.
    # "MlpPolicy" = a plain neural network (takes in a list of numbers).
    # verbose=1 = print progress.
    # Watch "ep_rew_mean". If it goes UP, the robot is learning.
    model = PPO("MlpPolicy", env, verbose=1)

    # Every 100,000 steps, save a copy of the brain into the
    # "checkpoints" folder. This lets us watch it while it trains.
    checkpoint = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints",
        name_prefix="stand",
    )

    # Train! total_timesteps = how many steps of practice in total.
    # 2,000,000 takes about 10 minutes.
    model.learn(total_timesteps=2_000_000, callback=checkpoint)

    # Save the final brain, so watch.py can load it.
    model.save("stand_ppo")
    print("Done! Saved to stand_ppo.zip")