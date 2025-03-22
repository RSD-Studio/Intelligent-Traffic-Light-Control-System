import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.sumo_env import SumoSingleIntersectionEnv, create_sumo_files
from models.dqn import DQNAgent


def train_dqn(env, agent, episodes=200, max_steps=1000, target_update_freq=10,
             log_freq=10, save_freq=50, render=False, save_dir='saved_models'):
    """Train the DQN agent on the SUMO environment."""
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create log directory
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('logs', 'dqn_' + current_time)
    os.makedirs(log_dir, exist_ok=True)
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    episode_queue_lengths = []
    best_avg_reward = float('-inf')
    
    # Training loop
    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        episode_queue_length = []
        
        # Episode loop
        for step in range(max_steps):
            # Select action
            action = agent.get_action(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience in replay buffer
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            episode_queue_length.append(info['queue_length_we'] + info['queue_length_ns'])
            
            # Train the agent
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                episode_loss += loss
            
            # Update target network
            if step % target_update_freq == 0:
                agent.update_target_network()
            
            if done:
                break
        
        # Append episode metrics
        episode_rewards.append(episode_reward)
        episode_losses.append(episode_loss / (step + 1) if step > 0 else 0)
        episode_queue_lengths.append(np.mean(episode_queue_length))
        
        # Log to file
        with open(os.path.join(log_dir, 'training_log.csv'), 'a') as f:
            f.write(f"{episode},{episode_reward},{episode_losses[-1]},{episode_queue_lengths[-1]},{agent.epsilon}\n")
        
        # Print episode results
        if episode % log_freq == 0:
            avg_reward = np.mean(episode_rewards[-log_freq:])
            avg_loss = np.mean(episode_losses[-log_freq:])
            avg_queue = np.mean(episode_queue_lengths[-log_freq:])
            
            print(f"Episode {episode}/{episodes} - "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Avg Loss: {avg_loss:.4f}, "
                  f"Avg Queue: {avg_queue:.2f}, "
                  f"Epsilon: {agent.epsilon:.2f}")
            
            # Save if improved
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_model(os.path.join(save_dir, 'dqn_best.pt'))
        
        # Save model periodically
        if episode % save_freq == 0:
            agent.save_model(os.path.join(save_dir, f'dqn_episode_{episode}.pt'))
    
    # Save final model
    agent.save_model(os.path.join(save_dir, 'dqn_final.pt'))
    
    # Plot training results
    plot_training_results(episode_rewards, episode_losses, episode_queue_lengths, save_dir)
    
    return episode_rewards, episode_losses, episode_queue_lengths


def plot_training_results(rewards, losses, queue_lengths, save_dir):
    """Plot and save training metrics."""
    # Create plots directory
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(os.path.join(plots_dir, 'rewards.png'))
    plt.close()
    
    # Plot smoothed rewards
    plt.figure(figsize=(10, 5))
    window_size = 10
    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(smoothed_rewards)
    plt.title(f'Smoothed Episode Rewards (Window Size: {window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.savefig(os.path.join(plots_dir, 'smoothed_rewards.png'))
    plt.close()
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Episode Losses')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.savefig(os.path.join(plots_dir, 'losses.png'))
    plt.close()
    
    # Plot queue lengths
    plt.figure(figsize=(10, 5))
    plt.plot(queue_lengths)
    plt.title('Average Queue Length per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Queue Length')
    plt.savefig(os.path.join(plots_dir, 'queue_lengths.png'))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN for traffic light control')
    parser.add_argument('--episodes', type=int, default=200, help='Number of episodes for training')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--arrival_rate', type=float, default=0.25, help='Vehicle arrival rate (Bernoulli parameter)')
    parser.add_argument('--hidden_size', type=int, default=400, help='Hidden layer size for DQN')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--memory_size', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save models')
    parser.add_argument('--device', type=str, default=None, help='Device to run the model on (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Create SUMO files if they don't exist
    create_sumo_files()
    
    # Create environment
    env = SumoSingleIntersectionEnv(use_gui=args.render, 
                                    max_steps=args.max_steps,
                                    arrival_rate=args.arrival_rate)
    
    # Create agent
    agent = DQNAgent(state_size=3,  # (X1, X2, L)
                    action_size=2,  # 0: Continue, 1: Switch
                    hidden_size=args.hidden_size,
                    learning_rate=args.learning_rate,
                    gamma=args.gamma,
                    memory_size=args.memory_size,
                    batch_size=args.batch_size,
                    device=args.device)
    
    # Train agent
    start_time = time.time()
    train_dqn(env, agent, episodes=args.episodes, max_steps=args.max_steps,
             render=args.render, save_dir=args.save_dir)
    end_time = time.time()
    
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    env.close()