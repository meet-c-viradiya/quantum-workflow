# Resource Visualization Module

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def visualize_resource_usage(G):
    """
    Enhanced 3D visualization of resource usage across tasks with improved clarity.
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get resource data
    x = [G.nodes[node]['CPU'] for node in G.nodes()]
    y = [G.nodes[node]['RAM'] for node in G.nodes()]
    z = [G.nodes[node]['disk'] for node in G.nodes()]
    task_ids = list(G.nodes())
    
    # Create scatter plot with better visibility
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=100, alpha=0.6)
    
    # Add task labels for important points
    for i, task_id in enumerate(task_ids):
        if x[i] > np.median(x) or y[i] > np.median(y) or z[i] > np.median(z):
            ax.text(x[i], y[i], z[i], task_id, fontsize=9)
    
    # Improve axes labels and ticks
    ax.set_xlabel('CPU Usage (cores)', fontsize=10, labelpad=10)
    ax.set_ylabel('RAM Usage (GB)', fontsize=10, labelpad=10)
    ax.set_zlabel('Disk Usage (GB)', fontsize=10, labelpad=10)
    
    # Add colorbar with better formatting
    cbar = plt.colorbar(scatter, label='Disk Usage (GB)', pad=0.1)
    cbar.ax.tick_params(labelsize=8)
    
    # Add statistics as text
    stats_text = f'Resource Statistics:\n'
    stats_text += f'CPU: mean={np.mean(x):.1f}, max={max(x):.1f}\n'
    stats_text += f'RAM: mean={np.mean(y):.1f}, max={max(y):.1f}\n'
    stats_text += f'Disk: mean={np.mean(z):.1f}, max={max(z):.1f}'
    
    ax.text2D(0.02, 0.95, stats_text, transform=ax.transAxes,
              bbox=dict(facecolor='white', alpha=0.8),
              fontsize=8)
    
    plt.title('3D Resource Usage Distribution', pad=20, fontsize=12)
    plt.tight_layout()
    plt.show()