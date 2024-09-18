import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import numpy as np

def train_agent(num_episodes=500):
    """
    Train an A2C model on the CartPole-v1 environment.

    Args:
        num_episodes (int): The number of episodes to train the agent.

    Returns:
        model: The trained A2C model.
        episode_rewards (list): Total rewards per episode.
    """
    # Initialize parallel environments for more efficient training
    vec_env = make_vec_env("CartPole-v1", n_envs=4)

    # Create the A2C model with a multi-layer perceptron policy
    model = A2C("MlpPolicy", vec_env, verbose=1)

    # Lists to store episode rewards and statistics
    episode_rewards = []

    # Train the model for a specified number of episodes
    for episode in range(num_episodes):
        # Reset the environment and accumulators for episode rewards
        obs = vec_env.reset()
        episode_reward_sum = 0

        while True:
            # Predict and take action
            action, _states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
            episode_reward_sum += np.sum(rewards)

            # Break loop when any environment is done
            if np.any(dones):
                break

        # Store the total reward for the episode
        episode_rewards.append(episode_reward_sum)

        # Log progress
        print(f"Episode: {episode + 1}, Reward: {episode_reward_sum}")

    # Save the trained model for future use
    model.save("a2c_cartpole_model")

    # Close the environments
    vec_env.close()

    return model, episode_rewards

def plot_results(episode_rewards):
    """
    Plot the training progress using episode rewards.

    Args:
        episode_rewards (list): List of total rewards per episode.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()


# Main function to train the agent and plot results
if __name__ == "__main__":
    trained_model, rewards = train_agent()
    plot_results(rewards)
