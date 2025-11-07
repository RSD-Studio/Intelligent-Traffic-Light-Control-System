"""
Hyperparameter Optimization using Optuna
Automatically finds optimal hyperparameters for the DQN agent
"""

import os
import sys
import argparse
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.sumo_env import create_sumo_files
from env.enhanced_sumo_env import EnhancedSumoEnv
from models.advanced_dqn import ImprovedDQNAgent


def objective(trial, base_config):
    """
    Objective function for Optuna optimization

    Args:
        trial: Optuna trial object
        base_config: Base configuration dictionary

    Returns:
        Average reward over test episodes
    """
    # Suggest hyperparameters
    hidden_size = trial.suggest_categorical('hidden_size', [256, 512, 768, 1024])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    gamma = trial.suggest_uniform('gamma', 0.95, 0.999)
    tau = trial.suggest_loguniform('tau', 0.001, 0.01)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    memory_size = trial.suggest_categorical('memory_size', [20000, 50000, 100000])

    # PER parameters
    alpha = trial.suggest_uniform('alpha', 0.4, 0.8)
    beta = trial.suggest_uniform('beta', 0.3, 0.6)

    # Architecture choices
    use_dueling = trial.suggest_categorical('use_dueling', [True, False])
    use_noisy = trial.suggest_categorical('use_noisy', [True, False])

    print(f"\nTrial {trial.number}:")
    print(f"  Hidden Size: {hidden_size}")
    print(f"  Learning Rate: {learning_rate:.6f}")
    print(f"  Gamma: {gamma:.4f}")
    print(f"  Tau: {tau:.6f}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Memory Size: {memory_size}")
    print(f"  PER Alpha: {alpha:.3f}")
    print(f"  PER Beta: {beta:.3f}")
    print(f"  Dueling: {use_dueling}")
    print(f"  Noisy: {use_noisy}")

    # Create environment
    env = EnhancedSumoEnv(
        use_gui=False,
        max_steps=base_config.get('max_steps', 500),
        arrival_rate=base_config.get('arrival_rate', 0.25),
        time_varying_traffic=False  # Use simpler traffic for faster tuning
    )

    # Create agent with suggested hyperparameters
    agent = ImprovedDQNAgent(
        state_size=15,
        action_size=2,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        memory_size=memory_size,
        batch_size=batch_size,
        device=base_config.get('device', 'cpu'),
        use_double_dqn=True,  # Always use Double DQN
        use_dueling=use_dueling,
        use_noisy=use_noisy,
        use_per=True,  # Always use PER
        use_lstm=False,  # Don't use LSTM for faster tuning
        alpha=alpha,
        beta=beta
    )

    # Train for a limited number of episodes
    train_episodes = base_config.get('train_episodes', 30)
    max_steps = base_config.get('max_steps', 500)

    episode_rewards = []

    try:
        for episode in range(train_episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                action = agent.get_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)

                if len(agent.memory) > agent.batch_size:
                    agent.replay()

                state = next_state
                episode_reward += reward

                if done:
                    break

            episode_rewards.append(episode_reward)

            # Report intermediate value for pruning
            trial.report(np.mean(episode_rewards[-5:]), episode)

            # Prune if performance is poor
            if trial.should_prune():
                env.close()
                raise optuna.TrialPruned()

        env.close()

        # Return average reward over last 10 episodes
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"  Final Average Reward: {avg_reward:.2f}")

        return avg_reward

    except Exception as e:
        print(f"  Trial failed with error: {e}")
        env.close()
        return float('-inf')


def run_hyperparameter_optimization(n_trials=50, n_jobs=1, base_config=None):
    """
    Run hyperparameter optimization

    Args:
        n_trials: Number of trials to run
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        base_config: Base configuration for training

    Returns:
        study: Optuna study object with results
    """
    if base_config is None:
        base_config = {
            'train_episodes': 30,
            'max_steps': 500,
            'arrival_rate': 0.25,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, base_config),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )

    # Print results
    print("\n" + "=" * 70)
    print("HYPERPARAMETER OPTIMIZATION COMPLETED")
    print("=" * 70)
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Avg Reward): {trial.value:.2f}")
    print("\n  Best Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save results
    results_dir = 'optimization_results'
    os.makedirs(results_dir, exist_ok=True)

    # Save study
    import joblib
    joblib.dump(study, os.path.join(results_dir, 'optuna_study.pkl'))

    # Save best parameters
    import json
    with open(os.path.join(results_dir, 'best_params.json'), 'w') as f:
        json.dump(trial.params, f, indent=2)

    print(f"\nResults saved to {results_dir}/")

    # Plot optimization history
    try:
        import matplotlib.pyplot as plt

        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(os.path.join(results_dir, 'optimization_history.png'), dpi=150)
        plt.close()

        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(os.path.join(results_dir, 'param_importances.png'), dpi=150)
        plt.close()

        print("Plots saved to optimization_results/")
    except Exception as e:
        print(f"Could not create plots: {e}")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization for Traffic Control')

    parser.add_argument('--n_trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--train_episodes', type=int, default=30, help='Episodes per trial')
    parser.add_argument('--max_steps', type=int, default=500, help='Steps per episode')
    parser.add_argument('--arrival_rate', type=float, default=0.25, help='Base arrival rate')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda)')

    args = parser.parse_args()

    # Create SUMO files
    create_sumo_files()

    # Base configuration
    base_config = {
        'train_episodes': args.train_episodes,
        'max_steps': args.max_steps,
        'arrival_rate': args.arrival_rate,
        'device': args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    }

    print("=" * 70)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"Number of trials: {args.n_trials}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Episodes per trial: {args.train_episodes}")
    print(f"Device: {base_config['device']}")
    print("=" * 70)

    # Run optimization
    study = run_hyperparameter_optimization(
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        base_config=base_config
    )
