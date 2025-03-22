import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import seaborn as sns
import pickle

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.sumo_env import SumoSingleIntersectionEnv, create_sumo_files
from models.dqn import DQNAgent, FixedCycleAgent, OptimalThresholdAgent


def test_policy(env, agent, episodes=10, max_steps=1000, render=False):
    """Test the agent policy on the SUMO environment."""
    # Testing metrics
    episode_rewards = []
    episode_queue_lengths = []
    episode_waiting_times = []
    episode_actions = []
    episode_states = []
    
    # Testing loop
    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_queue = []
        waiting_times = []
        actions = []
        states = [state]
        
        # Episode loop
        for step in range(max_steps):
            # Select action (no exploration)
            action = agent.get_action(state, training=False)
            actions.append(action)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Update state and metrics
            state = next_state
            states.append(state)
            episode_reward += reward
            episode_queue.append(info['queue_length_we'] + info['queue_length_ns'])
            
            # Track waiting time (sum of queue lengths)
            waiting_times.append(info['queue_length_we'] + info['queue_length_ns'])
            
            if done:
                break
        
        # Append episode metrics
        episode_rewards.append(episode_reward)
        episode_queue_lengths.append(np.mean(episode_queue))
        episode_waiting_times.append(np.sum(waiting_times))
        episode_actions.append(actions)
        episode_states.append(states)
        
        print(f"Episode {episode}/{episodes} - "
              f"Reward: {episode_reward:.2f}, "
              f"Avg Queue: {episode_queue_lengths[-1]:.2f}")
    
    # Calculate average metrics
    avg_reward = np.mean(episode_rewards)
    avg_queue = np.mean(episode_queue_lengths)
    avg_waiting = np.mean(episode_waiting_times)
    
    print("\nTest Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Queue Length: {avg_queue:.2f}")
    print(f"Average Total Waiting Time: {avg_waiting:.2f}")
    
    return {
        'rewards': episode_rewards,
        'queue_lengths': episode_queue_lengths,
        'waiting_times': episode_waiting_times,
        'actions': episode_actions,
        'states': episode_states,
        'avg_reward': avg_reward,
        'avg_queue': avg_queue,
        'avg_waiting_time': avg_waiting
    }


def compare_policies(env_config, agents, episodes=10, max_steps=1000, render=False, save_dir='results'):
    """Compare multiple policies on the same SUMO environment."""
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Results container
    results = {}
    
    # Test each agent
    for agent_name, agent in agents.items():
        print(f"\nTesting {agent_name}...")
        
        # Create environment
        env = SumoSingleIntersectionEnv(**env_config)
        
        # Test agent
        agent_results = test_policy(env, agent, episodes=episodes, max_steps=max_steps, render=render)
        results[agent_name] = agent_results
        
        # Close environment
        env.close()
    
    # Plot comparison results
    plot_comparison_results(results, save_dir)
    
    return results


def plot_comparison_results(results, save_dir):
    """Plot and save comparison metrics."""
    # Create plots directory
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract data for plots
    agent_names = list(results.keys())
    avg_rewards = [results[agent]['avg_reward'] for agent in agent_names]
    avg_queues = [results[agent]['avg_queue'] for agent in agent_names]
    avg_waiting_times = [results[agent]['avg_waiting_time'] for agent in agent_names]
    
    # Bar plot for average reward
    plt.figure(figsize=(12, 6))
    plt.bar(agent_names, avg_rewards)
    plt.title('Average Reward per Policy')
    plt.ylabel('Average Reward')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'avg_rewards.png'))
    plt.close()
    
    # Bar plot for average queue length
    plt.figure(figsize=(12, 6))
    plt.bar(agent_names, avg_queues)
    plt.title('Average Queue Length per Policy')
    plt.ylabel('Average Queue Length')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'avg_queues.png'))
    plt.close()
    
    # Bar plot for average waiting time
    plt.figure(figsize=(12, 6))
    plt.bar(agent_names, avg_waiting_times)
    plt.title('Average Waiting Time per Policy')
    plt.ylabel('Average Waiting Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'avg_waiting_times.png'))
    plt.close()
    
    # Box plots for rewards, queue lengths, and waiting times across episodes
    # Rewards
    data = []
    for agent in agent_names:
        for reward in results[agent]['rewards']:
            data.append({'Policy': agent, 'Reward': reward})
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Policy', y='Reward', data=df)
    plt.title('Distribution of Rewards per Policy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'reward_boxplot.png'))
    plt.close()
    
    # Queue lengths
    data = []
    for agent in agent_names:
        for queue in results[agent]['queue_lengths']:
            data.append({'Policy': agent, 'Queue Length': queue})
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Policy', y='Queue Length', data=df)
    plt.title('Distribution of Queue Lengths per Policy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'queue_boxplot.png'))
    plt.close()
    
    # Waiting times
    data = []
    for agent in agent_names:
        for waiting in results[agent]['waiting_times']:
            data.append({'Policy': agent, 'Waiting Time': waiting})
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Policy', y='Waiting Time', data=df)
    plt.title('Distribution of Waiting Times per Policy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'waiting_boxplot.png'))
    plt.close()
    
    # Create policy visualization for one episode
    # (to see how the actions change based on the state)
    for agent_name in agent_names:
        # Get first episode actions and states
        actions = results[agent_name]['actions'][0]
        states = results[agent_name]['states'][0]
        
        # Extract queue lengths and traffic light phases
        x1 = [state[0] for state in states]
        x2 = [state[1] for state in states]
        phases = [state[2] for state in states]
        
        # Plot queue lengths and actions
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Queue lengths
        ax1.plot(x1, label='West-East Queue')
        ax1.plot(x2, label='North-South Queue')
        ax1.set_ylabel('Queue Length')
        ax1.set_title(f'{agent_name} - Queue Lengths Over Time')
        ax1.legend()
        ax1.grid(True)
        
        # Traffic light phases and actions
        ax2.plot(phases, label='Traffic Light Phase')
        ax2.plot(actions, label='Action Taken', linestyle='--')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Phase / Action')
        ax2.set_title(f'{agent_name} - Traffic Light Phases and Actions')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{agent_name}_behavior.png'))
        plt.close()
    
    # Save results to CSV
    summary_data = {
        'Policy': agent_names,
        'Average Reward': avg_rewards,
        'Average Queue Length': avg_queues,
        'Average Waiting Time': avg_waiting_times
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(save_dir, 'policy_comparison_summary.csv'), index=False)
    
    # Save results to pickle for further visualization
    with open(os.path.join(save_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test and compare traffic light control policies')
    parser.add_argument('--dqn_model', type=str, default='saved_models/dqn_best.pt', help='Path to saved DQN model')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes for testing')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--arrival_rate', type=float, default=0.25, help='Vehicle arrival rate (Bernoulli parameter)')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--device', type=str, default=None, help='Device to run the model on (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Create SUMO files if they don't exist
    create_sumo_files()
    
    # Environment configuration
    env_config = {
        'use_gui': args.render,
        'max_steps': args.max_steps,
        'arrival_rate': args.arrival_rate
    }
    
    # Create agents
    agents = {}
    
    # DQN agent (if model file exists)
    if os.path.exists(args.dqn_model):
        dqn_agent = DQNAgent(state_size=3, action_size=2, device=args.device)
        dqn_agent.load_model(args.dqn_model)
        agents['DQN'] = dqn_agent
    else:
        print(f"Warning: DQN model file '{args.dqn_model}' not found. Skipping DQN evaluation.")
    
    # Fixed-cycle agent
    agents['FixedCycle'] = FixedCycleAgent(cycle_length=20)
    
    # Optimal threshold agent
    agents['OptimalThreshold'] = OptimalThresholdAgent(threshold=5)
    
    # Compare policies
    results = compare_policies(
        env_config, 
        agents, 
        episodes=args.episodes, 
        max_steps=args.max_steps, 
        render=args.render, 
        save_dir=args.save_dir
    )