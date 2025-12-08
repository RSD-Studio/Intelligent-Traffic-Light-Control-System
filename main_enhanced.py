"""
Enhanced Main Entry Point for Intelligent Traffic Light Control System
Supports all advanced features and improvements
"""

import os
import sys
import argparse
import numpy as np
import torch

from env.sumo_env import create_sumo_files
from env.enhanced_sumo_env import EnhancedSumoEnv
from models.advanced_dqn import ImprovedDQNAgent
from scripts.enhanced_train import enhanced_train, CurriculumTrainer


def setup():
    """Setup environment and create necessary files"""
    print("Setting up environment...")
    create_sumo_files()

    # Create directories
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('saved_models_enhanced', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('optimization_results', exist_ok=True)

    print("Setup complete!")


def train_mode(args):
    """Run training with enhanced features"""
    print("\n" + "=" * 70)
    print("ENHANCED TRAINING MODE")
    print("=" * 70)

    # Create environment
    env = EnhancedSumoEnv(
        use_gui=args.render,
        max_steps=args.max_steps,
        arrival_rate=args.arrival_rate,
        time_varying_traffic=args.time_varying
    )

    # Create agent with enhanced features
    agent = ImprovedDQNAgent(
        state_size=15,  # Enhanced state
        action_size=2,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        device=args.device,
        use_double_dqn=not args.no_double_dqn,
        use_dueling=not args.no_dueling,
        use_noisy=not args.no_noisy,
        use_per=not args.no_per,
        use_lstm=args.use_lstm
    )

    # Setup curriculum learning if requested
    curriculum_trainer = None
    if args.curriculum:
        curriculum_trainer = CurriculumTrainer(base_arrival_rate=args.arrival_rate)
        total_episodes = sum(stage['episodes'] for stage in curriculum_trainer.stages)
        args.episodes = total_episodes
        print(f"Curriculum Learning Enabled - Total Episodes: {total_episodes}")

    # Train
    history = enhanced_train(
        env, agent,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        save_dir=args.save_dir,
        use_tensorboard=args.tensorboard,
        use_curriculum=args.curriculum,
        curriculum_trainer=curriculum_trainer
    )

    env.close()
    print("\nTraining completed successfully!")


def test_mode(args):
    """Run testing and evaluation"""
    print("\n" + "=" * 70)
    print("ENHANCED TESTING MODE")
    print("=" * 70)

    # Load the enhanced model
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return

    print(f"Loading model from: {args.model_path}")

    # Create environment
    env = EnhancedSumoEnv(
        use_gui=args.render,
        max_steps=args.max_steps,
        arrival_rate=args.arrival_rate,
        time_varying_traffic=args.time_varying
    )

    # Create agent and load model
    agent = ImprovedDQNAgent(
        state_size=15,
        action_size=2,
        hidden_size=args.hidden_size,
        device=args.device,
        use_double_dqn=True,
        use_dueling=True,
        use_noisy=False,  # No exploration during testing
        use_per=True,
        use_lstm=args.use_lstm
    )

    agent.load_model(args.model_path)

    # Run test episodes
    print(f"\nRunning {args.test_episodes} test episodes...")

    all_rewards = []
    all_queues = []
    all_waiting_times = []
    all_throughputs = []

    for episode in range(args.test_episodes):
        state = env.reset()
        episode_reward = 0
        episode_queue = []
        episode_wait = []

        for step in range(args.max_steps):
            action = agent.get_action(state, training=False)
            next_state, reward, done, info = env.step(action)

            state = next_state
            episode_reward += reward
            episode_queue.append(info['total_queue'])
            episode_wait.append(info['avg_waiting_time'])

            if done:
                break

        all_rewards.append(episode_reward)
        all_queues.append(np.mean(episode_queue))
        all_waiting_times.append(np.mean(episode_wait))
        all_throughputs.append(env.vehicles_passed)

        print(f"Episode {episode+1}/{args.test_episodes} - "
              f"Reward: {episode_reward:.1f}, "
              f"Avg Queue: {np.mean(episode_queue):.2f}, "
              f"Avg Wait: {np.mean(episode_wait):.2f}s, "
              f"Throughput: {env.vehicles_passed}")

    env.close()

    # Print summary statistics
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"Average Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Average Queue Length: {np.mean(all_queues):.2f} ± {np.std(all_queues):.2f}")
    print(f"Average Waiting Time: {np.mean(all_waiting_times):.2f}s ± {np.std(all_waiting_times):.2f}s")
    print(f"Average Throughput: {np.mean(all_throughputs):.1f} ± {np.std(all_throughputs):.1f}")
    print("=" * 70)


def optimize_mode(args):
    """Run hyperparameter optimization"""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER OPTIMIZATION MODE")
    print("=" * 70)

    from scripts.hyperparameter_tuning import run_hyperparameter_optimization

    base_config = {
        'train_episodes': args.optim_train_episodes,
        'max_steps': args.max_steps,
        'arrival_rate': args.arrival_rate,
        'device': args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    }

    study = run_hyperparameter_optimization(
        n_trials=args.optim_trials,
        n_jobs=args.optim_jobs,
        base_config=base_config
    )

    print("\nOptimization completed successfully!")


def compress_mode(args):
    """Compress a trained model"""
    print("\n" + "=" * 70)
    print("MODEL COMPRESSION MODE")
    print("=" * 70)

    from utils.model_compression import compress_dqn_model

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return

    output_path = args.compress_output or args.model_path.replace('.pt', '_compressed.pt')

    compressed_model, comparison = compress_dqn_model(
        args.model_path,
        output_path,
        method=args.compress_method,
        sample_state_size=15
    )

    print("\nCompression completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Intelligent Traffic Light Control System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                       choices=['setup', 'train', 'test', 'optimize', 'compress', 'all'],
                       help='Operation mode')

    # Environment parameters
    parser.add_argument('--episodes', type=int, default=200, help='Training episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--arrival_rate', type=float, default=0.25, help='Base arrival rate')
    parser.add_argument('--time_varying', action='store_true', help='Use time-varying traffic')
    parser.add_argument('--render', action='store_true', help='Render SUMO GUI')

    # Agent parameters
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden layer size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient')
    parser.add_argument('--memory_size', type=int, default=50000, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    # Enhanced features toggles
    parser.add_argument('--no_double_dqn', action='store_true', help='Disable Double DQN')
    parser.add_argument('--no_dueling', action='store_true', help='Disable Dueling')
    parser.add_argument('--no_noisy', action='store_true', help='Disable Noisy Networks')
    parser.add_argument('--no_per', action='store_true', help='Disable PER')
    parser.add_argument('--use_lstm', action='store_true', help='Use LSTM architecture')

    # Training features
    parser.add_argument('--curriculum', action='store_true', help='Use curriculum learning')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='Use TensorBoard')

    # Testing
    parser.add_argument('--test_episodes', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--model_path', type=str, default='saved_models_enhanced/enhanced_dqn_best.pt',
                       help='Path to model for testing/compression')

    # Optimization
    parser.add_argument('--optim_trials', type=int, default=50, help='Optimization trials')
    parser.add_argument('--optim_jobs', type=int, default=1, help='Parallel optimization jobs')
    parser.add_argument('--optim_train_episodes', type=int, default=30,
                       help='Training episodes per optimization trial')

    # Compression
    parser.add_argument('--compress_method', type=str, default='dynamic_quantization',
                       choices=['dynamic_quantization', 'pruning', 'both'],
                       help='Compression method')
    parser.add_argument('--compress_output', type=str, help='Output path for compressed model')

    # Misc
    parser.add_argument('--save_dir', type=str, default='saved_models_enhanced',
                       help='Directory to save models')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda)')

    args = parser.parse_args()

    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Execute based on mode
    if args.mode == 'setup':
        setup()

    elif args.mode == 'train':
        setup()
        train_mode(args)

    elif args.mode == 'test':
        test_mode(args)

    elif args.mode == 'optimize':
        setup()
        optimize_mode(args)

    elif args.mode == 'compress':
        compress_mode(args)

    elif args.mode == 'all':
        setup()
        train_mode(args)
        test_mode(args)

    print("\n✓ All operations completed successfully!")


if __name__ == "__main__":
    main()
