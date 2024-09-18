import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Policy(nn.Module):
    """Policy network for REINFORCE algorithm.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        dropout (nn.Dropout): Dropout layer for regularization.
        fc2 (nn.Linear): Second fully connected layer.
    """

    def __init__(self):
        """Initializes the policy network with two layers."""
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        """Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input state.

        Returns:
            torch.Tensor: The output action probabilities.
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def train_reinforce(env, policy_net, optimizer, num_episodes=500):
    """Trains the policy network using the REINFORCE algorithm.

    Args:
        env (gym.Env): The environment to train on.
        policy_net (Policy): The policy network.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        num_episodes (int): Number of episodes to train.

    Returns:
        list: List of total rewards obtained in each episode during training.
    """
    rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards_episode = []

        while True:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action_probs = policy_net(state_tensor)
            m = torch.distributions.Categorical(action_probs)
            action = m.sample()
            log_probs.append(m.log_prob(action))
            state, reward, done, _ = env.step(action.item())
            rewards_episode.append(reward)
            if done:
                break

        # Compute returns with discounting
        returns = torch.tensor([sum(rewards_episode[i:] * (0.99 ** np.arange(len(rewards_episode) - i)))
                                 for i in range(len(rewards_episode))])
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Compute policy loss and entropy loss
        policy_loss = -torch.stack(log_probs).mul(returns).sum()
        entropy_loss = -0.01 * (action_probs * torch.log(action_probs)).sum(dim=1).mean()

        total_loss = policy_loss + entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_reward = sum(rewards_episode)
        rewards.append(total_reward)

    return rewards


if __name__ == "__main__": 
    # Setup and training
    env = gym.make('CartPole-v1')
    policy_net = Policy()
    optimizer = optim.Adam(policy_net.parameters(), lr=5e-3)

    rewards = train_reinforce(env, policy_net, optimizer)

    # Plotting the training progress
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.title('Training Progress')
    plt.show()
