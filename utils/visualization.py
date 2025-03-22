import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Polygon
import pickle

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class IntersectionVisualizer:
    """Visualizer for traffic light control at single intersection."""
    
    def __init__(self, save_dir='visualizations'):
        """Initialize the visualizer."""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Intersection coordinates
        self.intersection_center = (0, 0)
        self.road_width = 0.5
        self.road_length = 5
        
        # Colors
        self.road_color = 'gray'
        self.background_color = 'white'
        self.car_color = 'blue'
        self.light_colors = {
            0: 'green',   # Green for west-east
            1: 'yellow',  # Yellow for west-east
            2: 'red',     # Red for west-east (green for north-south)
            3: 'orange'   # Orange for north-south (red for west-east)
        }
    
    def create_intersection_plot(self):
        """Create a plot for the intersection."""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-self.road_length - 1, self.road_length + 1)
        ax.set_ylim(-self.road_length - 1, self.road_length + 1)
        ax.set_aspect('equal')
        ax.set_facecolor(self.background_color)
        
        # Draw west-east road
        we_road = Rectangle(
            (-self.road_length, -self.road_width/2),
            2 * self.road_length,
            self.road_width,
            color=self.road_color
        )
        ax.add_patch(we_road)
        
        # Draw north-south road
        ns_road = Rectangle(
            (-self.road_width/2, -self.road_length),
            self.road_width,
            2 * self.road_length,
            color=self.road_color
        )
        ax.add_patch(ns_road)
        
        # Add traffic light marker (will be updated)
        traffic_light = Rectangle(
            (-0.2, -0.2),
            0.4,
            0.4,
            color='green'
        )
        ax.add_patch(traffic_light)
        
        # Create legend for traffic light states
        ax.text(
            -self.road_length + 0.5, 
            self.road_length - 0.5,
            "Traffic Light States:\n"
            "0: Green for West-East\n"
            "1: Yellow for West-East\n"
            "2: Green for North-South\n"
            "3: Yellow for North-South",
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.5)
        )
        
        return fig, ax, traffic_light
    
    def animate_episode(self, states, actions, filename='traffic_animation.mp4'):
        """Create an animation for a complete episode."""
        fig, ax, traffic_light = self.create_intersection_plot()
        
        # Cars container
        west_east_cars = []
        north_south_cars = []
        
        # Animation function
        def update(frame):
            # Clear previous cars
            for car in west_east_cars + north_south_cars:
                car.remove()
            west_east_cars.clear()
            north_south_cars.clear()
            
            # Get state
            x1, x2, light_config = states[frame]
            
            # Update traffic light color
            traffic_light.set_color(self.light_colors[light_config])
            
            # Add cars for west-east direction
            for i in range(min(x1, 20)):  # Limit to 20 cars for visualization
                car = Rectangle(
                    (-self.road_length + 0.6 * i, -0.15),
                    0.4,
                    0.3,
                    color=self.car_color
                )
                ax.add_patch(car)
                west_east_cars.append(car)
            
            # Add cars for north-south direction
            for i in range(min(x2, 20)):  # Limit to 20 cars for visualization
                car = Rectangle(
                    (-0.15, -self.road_length + 0.6 * i),
                    0.3,
                    0.4,
                    color='red'
                )
                ax.add_patch(car)
                north_south_cars.append(car)
            
            # Update title with state and action
            action = "Continue" if actions[frame] == 0 else "Switch"
            ax.set_title(f"Step {frame}: X1={x1}, X2={x2}, Light={light_config}, Action={action}")
            
            return west_east_cars + north_south_cars + [traffic_light]
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, 
            update, 
            frames=len(states) - 1,  # -1 because we don't have an action for the last state
            interval=200,
            blit=True
        )
        
        # Save animation
        try:
            ani.save(os.path.join(self.save_dir, filename), writer='ffmpeg', fps=5)
        except ValueError:
            # Fall back to GIF if ffmpeg is unavailable
            gif_filename = filename.replace('.mp4', '.gif')
            ani.save(os.path.join(self.save_dir, gif_filename), writer='pillow', fps=5)
            print(f"Animation saved as GIF to {os.path.join(self.save_dir, gif_filename)}")
        plt.close(fig)
        
        print(f"Animation saved to {os.path.join(self.save_dir, filename)}")
    
    def plot_queue_evolution(self, states, actions, filename='queue_evolution.png'):
        """Plot the evolution of queue lengths and traffic light states."""
        # Extract data
        x1 = [state[0] for state in states]
        x2 = [state[1] for state in states]
        light_configs = [state[2] for state in states]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Plot queue lengths
        ax1.plot(x1, label='West-East Queue (X1)')
        ax1.plot(x2, label='North-South Queue (X2)')
        ax1.set_ylabel('Queue Length')
        ax1.set_title('Queue Lengths Over Time')
        ax1.legend()
        ax1.grid(True)
        
        # Plot traffic light states and actions
        ax2.plot(light_configs, label='Traffic Light State', color='purple')
        ax2.plot(actions, label='Action Taken', linestyle='--', color='orange')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('State / Action')
        ax2.set_title('Traffic Light States and Actions')
        ax2.legend()
        ax2.grid(True)
        
        # Add color bands for traffic light states
        for i in range(len(light_configs)-1):
            if light_configs[i] == 0:  # Green for west-east
                ax2.axvspan(i, i+1, alpha=0.1, color='green')
            elif light_configs[i] == 1:  # Yellow for west-east
                ax2.axvspan(i, i+1, alpha=0.1, color='yellow')
            elif light_configs[i] == 2:  # Green for north-south
                ax2.axvspan(i, i+1, alpha=0.1, color='red')
            elif light_configs[i] == 3:  # Yellow for north-south
                ax2.axvspan(i, i+1, alpha=0.1, color='orange')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close(fig)
        
        print(f"Queue evolution plot saved to {os.path.join(self.save_dir, filename)}")
    
    def plot_policy_heatmap(self, results, policy_name, filename='policy_heatmap.png'):
        """Create a heatmap showing the policy's actions for different states."""
        # Extract states and actions
        all_states = []
        all_actions = []
        
        for episode_states, episode_actions in zip(results[policy_name]['states'], results[policy_name]['actions']):
            all_states.extend(episode_states[:-1])  # Exclude last state which has no action
            all_actions.extend(episode_actions)
        
        # Create state-action matrix for each light configuration
        max_queue = 20  # Limit for visualization
        
        for light_config in range(4):
            # Filter states for this light configuration
            light_states = [(s[0], s[1]) for s, a in zip(all_states, all_actions) if s[2] == light_config]
            light_actions = [a for s, a in zip(all_states, all_actions) if s[2] == light_config]
            
            if not light_states:
                continue
            
            # Create action matrix
            action_matrix = np.zeros((max_queue + 1, max_queue + 1))
            count_matrix = np.zeros((max_queue + 1, max_queue + 1))
            
            for (x1, x2), action in zip(light_states, light_actions):
                if x1 <= max_queue and x2 <= max_queue:
                    action_matrix[x1, x2] += action
                    count_matrix[x1, x2] += 1
            
            # Normalize by count
            with np.errstate(divide='ignore', invalid='ignore'):
                probability_matrix = np.divide(action_matrix, count_matrix)
                probability_matrix = np.nan_to_num(probability_matrix)
            
            # Plot heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            cax = ax.matshow(probability_matrix, cmap='viridis', vmin=0, vmax=1)
            fig.colorbar(cax, label='Probability of Switching')
            
            # Add tick labels
            ax.set_xticks(np.arange(max_queue + 1, step=5))
            ax.set_yticks(np.arange(max_queue + 1, step=5))
            ax.set_xticklabels(np.arange(0, max_queue + 1, step=5))
            ax.set_yticklabels(np.arange(0, max_queue + 1, step=5))
            
            # Add axis labels and title
            ax.set_xlabel('North-South Queue (X2)')
            ax.set_ylabel('West-East Queue (X1)')
            
            light_states_map = {
                0: "Green for West-East",
                1: "Yellow for West-East",
                2: "Green for North-South",
                3: "Yellow for North-South"
            }
            
            ax.set_title(f'{policy_name} Policy: {light_states_map[light_config]}\nProbability of Switching Action')
            
            # Draw thresholds or patterns if visible
            if light_config == 0 or light_config == 2:
                # Draw a line where switching probability changes
                for i in range(max_queue + 1):
                    for j in range(max_queue + 1):
                        if probability_matrix[i, j] > 0.5:
                            ax.plot(j, i, 'r.', markersize=5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'{policy_name}_heatmap_light{light_config}.png'))
            plt.close(fig)
            
            print(f"Policy heatmap for light {light_config} saved to {os.path.join(self.save_dir, f'{policy_name}_heatmap_light{light_config}.png')}")
    
    def create_policy_visualization(self, results, save_dir='policy_viz'):
        """Create visualizations for all policies in the results."""
        os.makedirs(os.path.join(self.save_dir, save_dir), exist_ok=True)
        
        for policy_name in results.keys():
            print(f"Creating visualizations for {policy_name}...")
            
            # Get first episode states and actions
            states = results[policy_name]['states'][0]
            actions = results[policy_name]['actions'][0]
            
            # Plot queue evolution
            self.plot_queue_evolution(
                states, 
                actions, 
                filename=os.path.join(save_dir, f'{policy_name}_queue_evolution.png')
            )
            
            # Create animation
            self.animate_episode(
                states, 
                actions, 
                filename=os.path.join(save_dir, f'{policy_name}_animation.mp4')
            )
            
            # Create policy heatmap
            self.plot_policy_heatmap(
                results, 
                policy_name, 
                filename=os.path.join(save_dir, f'{policy_name}_heatmap.png')
            )


# Usage example
if __name__ == "__main__":
    # Load test results if available
    if os.path.exists('results/test_results.pkl'):
        with open('results/test_results.pkl', 'rb') as f:
            results = pickle.load(f)
        
        # Create visualizer
        vis = IntersectionVisualizer(save_dir='results/visualizations')
        
        # Create visualizations
        vis.create_policy_visualization(results)
    else:
        print("No test results found. Run test.py first to generate results.")