"""
Enhanced Training Script with All Improvements
- TensorBoard logging
- Curriculum learning
- Parallel training support
- Advanced metrics tracking
"""

import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pickle

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.sumo_env import create_sumo_files
from env.enhanced_sumo_env import EnhancedSumoEnv
from models.advanced_dqn import ImprovedDQNAgent


class CurriculumTrainer:
    """Curriculum learning for traffic control"""

    def __init__(self, base_arrival_rate=0.25):
        self.base_arrival_rate = base_arrival_rate
        self.stages = [
            {
                'name': 'Easy',
                'arrival_rate': base_arrival_rate * 0.4,
                'episodes': 30,
                'time_varying': False,
                'max_steps': 500
            },
            {
                'name': 'Medium',
                'arrival_rate': base_arrival_rate,
                'episodes': 50,
                'time_varying': False,
                'max_steps': 1000
            },
            {
                'name': 'Hard',
                'arrival_rate': base_arrival_rate * 2.0,
                'episodes': 50,
                'time_varying': False,
                'max_steps': 1000
            },
            {
                'name': 'Variable',
                'arrival_rate': base_arrival_rate,
                'episodes': 70,
                'time_varying': True,
                'max_steps': 1000
            }
        ]
        self.current_stage = 0

    def get_current_stage(self):
        """Get current curriculum stage"""
        if self.current_stage >= len(self.stages):
            return None
        return self.stages[self.current_stage]

    def advance_stage(self):
        """Move to next curriculum stage"""
        self.current_stage += 1

    def reset(self):
        """Reset to first stage"""
        self.current_stage = 0


def collect_experience_parallel(env_config, agent_config, num_steps):
    """Collect experience from parallel environment (for multiprocessing)"""
    # This would be used in a more advanced parallel training setup
    # For now, we'll implement basic parallel data collection
    env = EnhancedSumoEnv(**env_config)
    state = env.reset()

    experiences = []

    for _ in range(num_steps):
        # Simple random policy for demonstration
        action = np.random.choice(2)
        next_state, reward, done, info = env.step(action)
        experiences.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            state = env.reset()

    env.close()
    return experiences


def enhanced_train(env, agent, episodes=200, max_steps=1000,
                   log_freq=5, save_freq=25, render=False,
                   save_dir='saved_models', use_tensorboard=True,
                   use_curriculum=True, curriculum_trainer=None):
    """
    Enhanced training with all improvements

    Args:
        env: Training environment
        agent: DQN agent
        episodes: Number of training episodes
        max_steps: Max steps per episode
        log_freq: Logging frequency
        save_freq: Model save frequency
        render: Whether to render
        save_dir: Directory to save models
        use_tensorboard: Use TensorBoard logging
        use_curriculum: Use curriculum learning
        curriculum_trainer: CurriculumTrainer instance
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)

    # Create log directory
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('logs', 'enhanced_dqn_' + current_time)
    os.makedirs(log_dir, exist_ok=True)

    # TensorBoard writer
    if use_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
    else:
        writer = None

    # Training metrics
    episode_rewards = []
    episode_losses = []
    episode_queue_lengths = []
    episode_waiting_times = []
    episode_throughputs = []
    best_avg_reward = float('-inf')

    # Global step counter
    global_step = 0

    print("=" * 70)
    print("ENHANCED TRAINING STARTED")
    print("=" * 70)
    print(f"Device: {agent.device}")
    print(f"Double DQN: {agent.use_double_dqn}")
    print(f"Dueling: {agent.use_dueling}")
    print(f"Noisy Networks: {agent.use_noisy}")
    print(f"PER: {agent.use_per}")
    print(f"LSTM: {agent.use_lstm}")
    print(f"TensorBoard: {use_tensorboard}")
    print(f"Curriculum Learning: {use_curriculum}")
    print("=" * 70)

    # Curriculum learning
    if use_curriculum and curriculum_trainer:
        current_stage = curriculum_trainer.get_current_stage()
        if current_stage:
            print(f"\n[Curriculum] Stage: {current_stage['name']}")
            print(f"  Arrival Rate: {current_stage['arrival_rate']:.2f}")
            print(f"  Episodes: {current_stage['episodes']}")
            print(f"  Time Varying: {current_stage['time_varying']}")

    # Training loop
    episode_in_stage = 0

    for episode in range(1, episodes + 1):
        # Check curriculum stage progression
        if use_curriculum and curriculum_trainer:
            current_stage = curriculum_trainer.get_current_stage()
            if current_stage and episode_in_stage >= current_stage['episodes']:
                curriculum_trainer.advance_stage()
                episode_in_stage = 0
                next_stage = curriculum_trainer.get_current_stage()
                if next_stage:
                    print(f"\n[Curriculum] Advanced to Stage: {next_stage['name']}")
                    print(f"  Arrival Rate: {next_stage['arrival_rate']:.2f}")
                    print(f"  Episodes: {next_stage['episodes']}")
                    print(f"  Time Varying: {next_stage['time_varying']}")

                    # Update environment parameters
                    env.base_arrival_rate = next_stage['arrival_rate']
                    env.time_varying_traffic = next_stage['time_varying']
                    env.max_steps = next_stage['max_steps']

        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        loss_count = 0
        episode_queue = []
        episode_wait = []

        # Episode loop
        for step in range(max_steps):
            # Select action
            action = agent.get_action(state, training=True)

            # Take step
            next_state, reward, done, info = env.step(action)

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Train agent
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                episode_loss += loss
                loss_count += 1

                # Log to TensorBoard
                if writer and global_step % 10 == 0:
                    writer.add_scalar('Training/Loss', loss, global_step)

            # Update metrics
            state = next_state
            episode_reward += reward
            episode_queue.append(info['total_queue'])
            episode_wait.append(info['avg_waiting_time'])
            global_step += 1

            if done:
                break

        # Episode metrics
        avg_queue = np.mean(episode_queue)
        avg_wait = np.mean(episode_wait)
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        throughput = env.vehicles_passed

        episode_rewards.append(episode_reward)
        episode_losses.append(avg_loss)
        episode_queue_lengths.append(avg_queue)
        episode_waiting_times.append(avg_wait)
        episode_throughputs.append(throughput)

        episode_in_stage += 1

        # TensorBoard logging
        if writer:
            writer.add_scalar('Episode/Reward', episode_reward, episode)
            writer.add_scalar('Episode/AvgLoss', avg_loss, episode)
            writer.add_scalar('Episode/AvgQueue', avg_queue, episode)
            writer.add_scalar('Episode/AvgWaitingTime', avg_wait, episode)
            writer.add_scalar('Episode/Throughput', throughput, episode)

            # Log reward components
            if 'reward_components' in info:
                for comp_name, comp_value in info['reward_components'].items():
                    writer.add_scalar(f'Reward/{comp_name}', comp_value, episode)

        # Console logging
        if episode % log_freq == 0:
            avg_reward = np.mean(episode_rewards[-log_freq:])
            avg_loss_recent = np.mean(episode_losses[-log_freq:])
            avg_queue_recent = np.mean(episode_queue_lengths[-log_freq:])
            avg_wait_recent = np.mean(episode_waiting_times[-log_freq:])
            avg_throughput = np.mean(episode_throughputs[-log_freq:])

            print(f"Episode {episode:4d}/{episodes} | "
                  f"Reward: {avg_reward:8.1f} | "
                  f"Loss: {avg_loss_recent:6.2f} | "
                  f"Queue: {avg_queue_recent:5.2f} | "
                  f"Wait: {avg_wait_recent:5.2f}s | "
                  f"Throughput: {avg_throughput:4.0f}")

            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_model(os.path.join(save_dir, 'enhanced_dqn_best.pt'))
                print(f"  → New best model saved! (Reward: {avg_reward:.1f})")

        # Periodic model save
        if episode % save_freq == 0:
            agent.save_model(os.path.join(save_dir, f'enhanced_dqn_episode_{episode}.pt'))
            print(f"  → Checkpoint saved at episode {episode}")

    # Save final model
    agent.save_model(os.path.join(save_dir, 'enhanced_dqn_final.pt'))
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    # Save training history
    history = {
        'rewards': episode_rewards,
        'losses': episode_losses,
        'queue_lengths': episode_queue_lengths,
        'waiting_times': episode_waiting_times,
        'throughputs': episode_throughputs
    }

    with open(os.path.join(log_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    # Plot results
    plot_enhanced_training_results(history, log_dir)

    # Close TensorBoard writer
    if writer:
        writer.close()

    return history


def plot_enhanced_training_results(history, save_dir):
    """Plot comprehensive training results"""
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Enhanced DQN Training Results', fontsize=16)

    # 1. Rewards
    ax = axes[0, 0]
    ax.plot(history['rewards'], alpha=0.3, label='Raw')
    window = min(20, len(history['rewards']) // 10)
    if window > 1:
        smoothed = np.convolve(history['rewards'], np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(history['rewards'])), smoothed, label=f'Smoothed ({window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Losses
    ax = axes[0, 1]
    ax.plot(history['losses'])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)

    # 3. Queue Lengths
    ax = axes[0, 2]
    ax.plot(history['queue_lengths'])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Queue Length')
    ax.set_title('Average Queue Length per Episode')
    ax.grid(True, alpha=0.3)

    # 4. Waiting Times
    ax = axes[1, 0]
    ax.plot(history['waiting_times'])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Waiting Time (s)')
    ax.set_title('Average Waiting Time per Episode')
    ax.grid(True, alpha=0.3)

    # 5. Throughput
    ax = axes[1, 1]
    ax.plot(history['throughputs'])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Vehicles Passed')
    ax.set_title('Throughput per Episode')
    ax.grid(True, alpha=0.3)

    # 6. Performance Summary
    ax = axes[1, 2]
    window = 50
    if len(history['rewards']) >= window:
        early_reward = np.mean(history['rewards'][:window])
        late_reward = np.mean(history['rewards'][-window:])
        early_queue = np.mean(history['queue_lengths'][:window])
        late_queue = np.mean(history['queue_lengths'][-window:])
        early_wait = np.mean(history['waiting_times'][:window])
        late_wait = np.mean(history['waiting_times'][-window:])

        metrics = ['Reward\n(×100)', 'Queue\nLength', 'Waiting\nTime']
        early_vals = [early_reward/100, early_queue, early_wait]
        late_vals = [late_reward/100, late_queue, late_wait]

        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(x - width/2, early_vals, width, label=f'First {window} episodes', alpha=0.8)
        ax.bar(x + width/2, late_vals, width, label=f'Last {window} episodes', alpha=0.8)

        ax.set_ylabel('Value')
        ax.set_title('Training Progress Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_summary.png'), dpi=150)
    plt.close()

    print(f"Training plots saved to {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced DQN Training for Traffic Control')

    # Environment parameters
    parser.add_argument('--episodes', type=int, default=200, help='Number of training episodes')
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
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')

    # Enhanced features
    parser.add_argument('--no_double_dqn', action='store_true', help='Disable Double DQN')
    parser.add_argument('--no_dueling', action='store_true', help='Disable Dueling architecture')
    parser.add_argument('--no_noisy', action='store_true', help='Disable Noisy Networks')
    parser.add_argument('--no_per', action='store_true', help='Disable PER')
    parser.add_argument('--use_lstm', action='store_true', help='Use LSTM architecture')

    # Training features
    parser.add_argument('--curriculum', action='store_true', help='Use curriculum learning')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='Use TensorBoard')

    # Misc
    parser.add_argument('--save_dir', type=str, default='saved_models_enhanced', help='Model save directory')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda)')

    args = parser.parse_args()

    # Create SUMO files
    create_sumo_files()

    # Create environment
    env = EnhancedSumoEnv(
        use_gui=args.render,
        max_steps=args.max_steps,
        arrival_rate=args.arrival_rate,
        time_varying_traffic=args.time_varying
    )

    # Create agent
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

    # Curriculum trainer
    curriculum_trainer = None
    if args.curriculum:
        curriculum_trainer = CurriculumTrainer(base_arrival_rate=args.arrival_rate)
        total_episodes = sum(stage['episodes'] for stage in curriculum_trainer.stages)
        args.episodes = total_episodes

    # Train
    start_time = time.time()

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

    end_time = time.time()

    print(f"\nTotal training time: {end_time - start_time:.2f} seconds")
    print(f"Average time per episode: {(end_time - start_time) / args.episodes:.2f} seconds")

    env.close()
