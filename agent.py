import flappy_bird_gymnasium
import gymnasium

import torch
import torch.nn.functional as F
import numpy as np

from dqn import DQN
from experience_replay import ReplayMemory
import itertools

import yaml
import random

import matplotlib
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import os
import argparse

DATE_FORMAT = "%m-%d %H:%M:%S"

# directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# use matplotlib agg
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

class Agent:

    def __init__(self, hyperparameters):
        """
        Initialises the agent with the specified hyperparameters.
        """
        with open('hyperparameters.yaml', 'r') as file:
            all_hyperparameters = yaml.safe_load(file)
            instance_hyperparameters = all_hyperparameters[hyperparameters]

        self.hyperparameters = hyperparameters
        self.env_id = instance_hyperparameters['env_id']
        self.checkpoint = instance_hyperparameters['checkpoint']
        self.prior_data = instance_hyperparameters['prior_data']
        self.replay_memory_size = instance_hyperparameters['replay_memory_size']
        self.batch_size = instance_hyperparameters['batch_size']
        self.epsilon_start = instance_hyperparameters['epsilon_start']
        self.epsilon_decay = instance_hyperparameters['epsilon_decay']
        self.epsilon_end = instance_hyperparameters['epsilon_end']
        self.network_sync_rate = instance_hyperparameters['network_sync_rate']
        self.discount_factor = instance_hyperparameters['discount_factor']
        self.lr = instance_hyperparameters['lr']
        self.hidden_dims = instance_hyperparameters['hidden_dims']
        self.maximum_reward_stop = instance_hyperparameters['maximum_reward_stop']
        self.enable_double_DQN = instance_hyperparameters['enable_double_DQN']
        self.enable_dueling_DQN = instance_hyperparameters['enable_dueling_DQN']

        # define loss and optimiser
        self.loss_fn = torch.nn.MSELoss()
        self.optimiser = None
        
        # path to run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameters}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameters}.pth")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameters}.png")
        self.DATA_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameters}.pkl")

    def run(self, is_training=True, render=False):
        """
        Runs the agent in either training or evaluation mode.
        """
        if is_training:
            start_time = datetime.now()
            last_graph_update = start_time

            log_message = f"Training {start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, "w") as file:
                file.write(log_message)

        # set up the environment
        env = gymnasium.make(self.env_id, render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewards_per_episode = []
        
        # Initialize the policy DQN with the specified dimensions and move it to the appropriate device
        policy_dqn = DQN(state_dim=num_states, action_dim=num_actions, hidden_dim=self.hidden_dims, enable_dueling_DQN=self.enable_dueling_DQN).to(device)

        # Set up the optimizer with the policy DQN parameters and learning rate
        self.optimiser = torch.optim.Adam(params=policy_dqn.parameters(), lr=self.lr)

        if is_training:

            # Print confirmation conficguration
            # if provided checkpoint, load from it
            if self.checkpoint != "":
                print("Loading from checkpoint...")
                policy_dqn.load_state_dict(torch.load(self.checkpoint))
                print(f"Checkpoint loaded from {self.checkpoint}")
            else:
                print("Training from scratch...")

            if self.enable_double_DQN:
                print("Double DQN enabled")
            if self.enable_dueling_DQN:
                print("Dueling DQN enabled")

                
            # set up the memory, target network, and step count
            memory = ReplayMemory(self.replay_memory_size)

            # load if prior data is provided
            if self.prior_data != "":
                print("Loading prior data...")
                memory.load_memory_from_file(self.prior_data)
                print(f"Prior data loaded from {self.prior_data}")


            epsilon = self.epsilon_start

            target_dqn = DQN(state_dim=num_states, action_dim=num_actions, hidden_dim=self.hidden_dims, enable_dueling_DQN=self.enable_dueling_DQN).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            epsilon_history = []
            
            step_count = 0

            best_reward = -9999999
            

        else:
            # test the model to evaluation mode
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()


        # run the episodes
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)

            terminated = False
            episode_reward = 0.0

            while (not terminated and episode_reward < self.maximum_reward_stop):
                # Next action using epsilon-greedy
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64).to(device)

                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).squeeze(0).argmax()

                # apply action
                new_state, reward, terminated, _, _ = env.step(action.item())

                episode_reward += reward

                # convert new_state and reward to tensor
                new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
                reward = torch.tensor(reward, dtype=torch.float32).to(device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_count += 1

                state = new_state

            rewards_per_episode.append(episode_reward)

            # save model when best rewards is obtained to log
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"Training {datetime.now().strftime(DATE_FORMAT)}: New best reward: {best_reward} Time taken: {datetime.now() - start_time} Episode: {episode}\n"
                    print(log_message)
                    with open(self.LOG_FILE, "a") as file:
                        file.write(log_message)

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                    # write into a file the memory contents
                    memory.save_memory_to_file(self.DATA_FILE)


                #update graph every 10 second
                current_time = datetime.now()
                if current_time - last_graph_update > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update = current_time


                # linear decay for epsilon
                epsilon = max(self.epsilon_end, self.epsilon_decay * epsilon)
                epsilon_history.append(epsilon)

                if len(memory) > self.batch_size:
                    batch = memory.sample(self.batch_size)

                    self.optimise(policy_dqn, target_dqn, batch)

                    # sync the target network with respect to the policy network
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

    def optimise(self, policy_dqn, target_dqn, batch):
        """
        Optimises the policy DQN using the provided batch.
        """
        # optimise the network
        states, actions, new_states, rewards, terminated = zip(*batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminated = torch.tensor(terminated, dtype=torch.float32).to(device)

        with torch.no_grad():
            if self.enable_double_DQN:
                policy_best_action = policy_dqn(new_states).argmax(dim=1)
                target_q_values = rewards + self.discount_factor * target_dqn(new_states).gather(1, policy_best_action.unsqueeze(1)).squeeze(1)
            else:
                target_q_values = target_dqn(new_states).max(dim=1)[0]
                target_q_values = rewards + self.discount_factor * target_q_values * (1 - terminated)

        current_q_values = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(current_q_values, target_q_values)

        # Reset the gradients of all optimised variables
        self.optimiser.zero_grad()

        # Backpropagate the gradients of the loss with respect to the model parameters
        loss.backward()

        # Update the model parameters using the gradients and the chosen optimiser algorithm
        self.optimiser.step()

    def save_graph(self, rewards_per_episode, epsilon_history):
        
        fig = plt.figure(1)

        # plot rewards mean

        plt.subplot(1,2,1)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(rewards_per_episode)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-100):x+1])

        plt.plot(mean_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward per episode")

        # plot epsilon
        plt.subplot(1,2,2)
        plt.plot(epsilon_history)
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Epsilon per episode")

        fig.savefig(self.GRAPH_FILE)

        plt.close(fig)

if __name__ == "__main__":
    # parse command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("gym_environment", help="Name of the gym environment")
    parser.add_argument("--train", help="Train the agent", action="store_true")
    parser.add_argument("--render", help="Render the environment", action="store_true")
    args = parser.parse_args()
    
    dql = Agent(args.gym_environment)
    if args.train:
        dql.run(is_training=True, render=args.render)
    else:
        dql.run(is_training=False, render=args.render)  

