import os
import sys
import argparse
import pickle
import time
from datetime import datetime

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.sumo_env import SumoSingleIntersectionEnv, create_sumo_files
from models.dqn import DQNAgent, FixedCycleAgent, OptimalThresholdAgent
from scripts.train import train_dqn
from scripts.test import test_policy, compare_policies
from utils.visualization import IntersectionVisualizer


def setup_environment():
    """Setup the environment by creating necessary directories and SUMO files."""
    # Create directories
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Create SUMO files
    create_sumo_files()
    
    print("Environment setup complete.")


def train_model(args):
    """Train a DQN model for traffic light control."""
    print("Starting DQN training...")
    
    # Create environment
    env = SumoSingleIntersectionEnv(
        use_gui=args.render,
        max_steps=args.max_steps,
        arrival_rate=args.arrival_rate
    )
    
    # Create agent
    agent = DQNAgent(
        state_size=3,  # (X1, X2, L)
        action_size=2,  # 0: Continue, 1: Switch
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Train agent
    start_time = time.time()
    train_dqn(
        env, 
        agent, 
        episodes=args.episodes, 
        max_steps=args.max_steps,
        render=args.render, 
        save_dir=args.save_dir
    )
    end_time = time.time()
    
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    env.close()


def test_models(args):
    """Test and compare different traffic light control policies."""
    print("Starting policy comparison...")
    
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
        episodes=args.test_episodes, 
        max_steps=args.max_steps, 
        render=args.render, 
        save_dir=args.save_dir
    )
    
    # Create visualizations
    visualizer = IntersectionVisualizer(save_dir=os.path.join(args.save_dir, 'visualizations'))
    visualizer.create_policy_visualization(results)


def main():
    """Main function for running the traffic light control project."""
    parser = argparse.ArgumentParser(description='Traffic Light Control with DQN')
    
    # Common arguments
    parser.add_argument('--mode', type=str, choices=['setup', 'train', 'test', 'all'], 
                        default='all', help='Mode of operation')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to run the model on (cpu or cuda)')
    
    # Training arguments
    parser.add_argument('--episodes', type=int, default=200, help='Number of episodes for training')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--arrival_rate', type=float, default=0.25, help='Vehicle arrival rate (Bernoulli parameter)')
    parser.add_argument('--hidden_size', type=int, default=400, help='Hidden layer size for DQN')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--memory_size', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    
    # Testing arguments
    parser.add_argument('--test_episodes', type=int, default=10, help='Number of episodes for testing')
    parser.add_argument('--dqn_model', type=str, default='saved_models/dqn_best.pt', help='Path to saved DQN model')
    
    args = parser.parse_args()
    
    # Create results directory with timestamp
    if args.mode != 'setup':
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.save_dir = os.path.join(args.save_dir, timestamp)
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Run selected mode
    if args.mode == 'setup' or args.mode == 'all':
        setup_environment()
    
    if args.mode == 'train' or args.mode == 'all':
        train_model(args)
    
    if args.mode == 'test' or args.mode == 'all':
        test_models(args)
    
    print("Done!")


if __name__ == "__main__":
    main()