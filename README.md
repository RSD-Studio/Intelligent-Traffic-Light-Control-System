# Traffic Light Control System with Deep Reinforcement Learning

This project implements a Deep Q-Network (DQN) algorithm for intelligent traffic light control at a single intersection, as described in the paper "Deep Reinforcement Learning for Traffic Light Control in Intelligent Transportation Systems" by Liu et al. The system demonstrates how reinforcement learning can optimize traffic flow and reduce congestion compared to traditional fixed-cycle approaches.

## Project Overview

Traffic congestion represents a significant challenge in urban environments, leading to increased travel times, fuel consumption, and emissions. This project addresses this issue by implementing an intelligent traffic light control system that adapts to real-time traffic conditions. By leveraging deep reinforcement learning, the system learns optimal traffic light timing strategies to minimize queue lengths and waiting times at intersections.

The implementation includes:
- A SUMO-based simulation environment for realistic traffic modeling
- A PyTorch implementation of Deep Q-Network (DQN) for adaptive control
- Comparative analysis with fixed-cycle and optimal threshold policies
- Comprehensive visualization tools for performance evaluation

## Project Structure

```
traffic_light_dqn/
├── env/
│   ├── __init__.py
│   ├── sumo_env.py          # SUMO environment wrapper
│   └── sumo_files/          # SUMO configuration files
├── models/
│   ├── __init__.py
│   └── dqn.py               # DQN implementation with PyTorch
├── utils/
│   ├── __init__.py
│   └── visualization.py     # Visualization utilities
├── scripts/
│   ├── train.py             # Training script
│   └── test.py              # Testing script
├── saved_models/            # Directory for trained models
├── results/                 # Results and visualizations
├── main.py                  # Main entry point
└── README.md                # This file
```

## Requirements

### Software Dependencies
- Python 3.8+
- SUMO 1.10.0+
- PyTorch 1.8.0+
- NumPy
- Matplotlib
- Pandas
- Seaborn
- FFmpeg (for animations)

### Hardware Requirements
- Minimum 4GB RAM
- 2+ CPU cores recommended
- GPU optional but beneficial for faster training

## Installation

### 1. Set Up Python Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install torch numpy matplotlib pandas seaborn gym
```

### 2. Install SUMO Traffic Simulator

#### For Ubuntu:
```bash
sudo apt-get install sumo sumo-tools sumo-doc
```

#### For macOS:
```bash
brew install sumo
```

#### For Windows:
- Download from [SUMO website](https://sumo.dlr.de/docs/Downloads.php)
- Run the installer and follow the prompts
- Set the environment variable:
  ```
  setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"
  ```
  (Adjust path according to your installation location)

### 3. Install FFmpeg (for animations)

#### For Ubuntu:
```bash
sudo apt-get install ffmpeg
```

#### For macOS:
```bash
brew install ffmpeg
```

#### For Windows:
- Install using [Chocolatey](https://chocolatey.org/):
  ```
  choco install ffmpeg
  ```
- Or download from [FFmpeg website](https://ffmpeg.org/download.html)

### 4. Clone the Repository

```bash
git clone <repository-url>
cd traffic_light_control_system
```

## Usage

### Setting Up the Environment

```bash
python main.py --mode=setup
```

This command creates necessary SUMO configuration files for the simulation.

### Training the DQN Model

```bash
python main.py --mode=train --episodes=200 --arrival_rate=0.25
```

Parameters:
- `--episodes`: Number of training episodes (default: 200)
- `--arrival_rate`: Probability of vehicle arrival per time step (default: 0.25)
- `--hidden_size`: Neural network hidden layer size (default: 400)
- `--learning_rate`: Learning rate for optimization (default: 0.001)
- `--render`: Add this flag to visualize the training in SUMO GUI
- `--device`: Specify "cuda" or "cpu" for computation

The training process will save models periodically in the `saved_models` directory and generate learning curves in `saved_models/plots/`.

### Testing and Comparing Policies

```bash
python main.py --mode=test --test_episodes=10 --arrival_rate=0.25 --dqn_model="saved_models/dqn_best.pt"
```

Parameters:
- `--test_episodes`: Number of evaluation episodes (default: 10)
- `--dqn_model`: Path to the saved DQN model
- `--render`: Add this flag to visualize the testing in SUMO GUI

This command evaluates the DQN policy against fixed-cycle and optimal threshold policies, generating performance metrics and visualizations in the `results` directory.

### Running the Complete Pipeline

```bash
python main.py --mode=all --episodes=200 --test_episodes=10 --arrival_rate=0.25
```

This command runs the entire pipeline: setup, training, and testing.

## Implementation Details

### Environment

The environment simulates a single intersection with two traffic flows (West-East and North-South). The state is represented by:
- Queue lengths in each direction (X1, X2)
- Traffic light configuration (0-3, representing different phases)

Actions are binary (0: continue current phase, 1: switch to next phase), and the reward is the negative sum of squared queue lengths.

### DQN Implementation

The DQN implementation features:
- A two-layer neural network with tanh activation
- Experience replay buffer for stable learning
- Target network for reducing overestimation bias
- Epsilon-greedy exploration strategy

### Comparative Policies

1. **Fixed-Cycle**: Traditional approach with fixed timing for each phase
2. **Optimal Threshold**: Switches based on difference in queue lengths
3. **DQN**: Learns optimal policy through interaction with the environment

## Results and Visualization

The system generates several visualizations to evaluate performance:

- **Learning Curves**: Show training progress of the DQN agent
- **Queue Evolution Plots**: Track queue lengths over time for each policy
- **Performance Comparisons**: Compare average queue length, waiting time, and rewards
- **Policy Heatmaps**: Visualize action probabilities in different states
- **Traffic Animations**: Animate the intersection dynamics (requires FFmpeg)

Example results demonstrate that both DQN and Optimal Threshold policies significantly outperform the Fixed-Cycle policy in terms of average queue length and waiting time.

## Troubleshooting

### Common Issues

1. **SUMO_HOME Not Found**:
   - Ensure SUMO is properly installed
   - Set the SUMO_HOME environment variable correctly
   - Start a new terminal session after setting the variable

2. **FFmpeg Not Available**:
   - Install FFmpeg as described in the installation section
   - Or modify `utils/visualization.py` to use GIF format instead

3. **DQN Model Not Found**:
   - Check that the model path is correct
   - Ensure training has completed successfully
   - Look in timestamped result directories for model files

4. **CUDA Not Available**:
   - Verify PyTorch is installed with CUDA support
   - Fall back to CPU with `--device=cpu`

## References

Liu, X.-Y., Zhu, M., Borst, S., & Walid, A. (2023). Deep Reinforcement Learning for Traffic Light Control in Intelligent Transportation Systems. IEEE Transactions on Network Science and Engineering.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
