# Quantum Workflow Visualization Module

import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(G):
    """
    Visualizes the workflow graph using NetworkX and Matplotlib.

    Parameters:
    G (networkx.DiGraph): The directed graph representing the workflow.

    Returns:
    None
    """
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_color='black', font_weight='bold', arrows=True)
    plt.title("Workflow Graph Visualization")
    plt.show()

def visualize_schedule_timeline(assignments, G):
    """
    Create a timeline visualization of task schedules.

    Parameters:
    assignments (dict): A dictionary mapping tasks to processors.
    G (networkx.DiGraph): The directed graph representing the workflow.

    Returns:
    None
    """
    processors = defaultdict(list)
    for task, proc in assignments.items():
        processors[proc].append({
            'task': task,
            'runtime': G.nodes[task]['Runtime_C1'],
            'cpu': G.nodes[task]['CPU'],
            'ram': G.nodes[task]['RAM']
        })
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    colors = plt.cm.Set3(np.linspace(0, 1, len(G.nodes())))
    
    # Timeline
    for i, (proc, tasks) in enumerate(processors.items()):
        y = i
        current_time = 0
        for task in tasks:
            ax1.barh(y, task['runtime'], left=current_time, 
                    color=colors[list(G.nodes()).index(task['task'])])
            current_time += task['runtime']
    
    ax1.set_yticks(range(len(processors)))
    ax1.set_yticklabels(processors.keys())
    ax1.set_xlabel('Time Units')
    ax1.set_title('Schedule Timeline')
    
    # Resource Usage
    for i, (proc, tasks) in enumerate(processors.items()):
        cpu_usage = sum(task['cpu'] for task in tasks)
        ram_usage = sum(task['ram'] for task in tasks)
        ax2.bar(i-0.2, cpu_usage, width=0.4, label='CPU' if i==0 else "")
        ax2.bar(i+0.2, ram_usage, width=0.4, label='RAM' if i==0 else "")
    
    ax2.set_xticks(range(len(processors)))
    ax2.set_xticklabels(processors.keys())
    ax2.set_title('Resource Usage per Processor')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()