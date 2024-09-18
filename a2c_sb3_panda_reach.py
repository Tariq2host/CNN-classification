import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
import panda_gym
from wandb.integration.sb3 import WandbCallback
from huggingface_hub import notebook_login
from huggingface_sb3 import push_to_hub

# Log in to the notebook
notebook_login()

# Configuration dictionary for the RL agent
config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 500_000,
    "env_name": "PandaReachJointsDense-v3",
}

# Initialize a new wandb run
run = wandb.init(
    project="a2c_panda_reach2",
    config=config,
    sync_tensorboard=True,  # Auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # Auto-upload the videos of agents playing the game
    save_code=True,  # Save the code (optional)
)

def make_env():
    """Create and return a wrapped gym environment."""
    env = gym.make(config["env_name"])
    env = Monitor(env)  # Record stats such as returns
    return env

# Create the environment and model, then start training
env = DummyVecEnv([make_env])
model = A2C(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.save("a2c_PandaReachJointsDense-v3")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
)

# Finish the run
run.finish()

# Push the model to the Hugging Face Hub
push_to_hub(
    repo_id="Tariq2host/second_model",
    filename="./a2c_PandaReachJointsDense-v3.zip",
    commit_message="Added a2c_PandaReachJointsDense-v3 model trained with A2C",
)
