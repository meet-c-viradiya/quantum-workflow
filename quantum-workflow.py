#!/usr/bin/env python3

"""
Workflow DAG Analyzer and QAOA Optimizer

This script:
1. Loads workflow data from a CSV file.
2. Builds a directed acyclic graph (DAG) from the workflow.
3. Visualizes the graph with category-based coloring.
4. Prepares for QUBO and quantum optimization (QAOA via Qiskit v1.4.0).
"""

import sys
import ast
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from networkx.drawing.nx_agraph import graphviz_layout
# Updated imports for latest Qiskit
from qiskit_algorithms import QAOA
from qiskit.primitives import Sampler  # Changed from SamplerV1
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_aer import AerSimulator
from scipy.optimize import minimize  # Add this line to existing imports

from collections import defaultdict, deque
from mpl_toolkits.mplot3d import Axes3D


##############################
# Step 1: Load Workflow CSV
##############################

def load_data(filepath):
    """Load workflow data from CSV into a pandas DataFrame."""
    try:
        df = pd.read_csv(filepath, delimiter=',')
        print("CSV data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"File '{filepath}' not found. Please check the path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error while loading CSV: {e}")
        sys.exit(1)

##############################
# Step 2: Build DAG from DataFrame
##############################

def build_graph(df):
    """Build a directed acyclic graph (DAG) from workflow DataFrame."""
    G = nx.DiGraph()
    task_type_col = 'K' if 'K' in df.columns else 'task_type'

    for idx, row in df.iterrows():
        task_id_raw = row.get('taskID')
        if pd.isna(task_id_raw):
            continue

        task_id = str(task_id_raw).strip()
        category_raw = row.get(task_type_col, "Unknown")
        task_category = "Unknown" if pd.isna(category_raw) or category_raw == '' else str(category_raw).strip()

        G.add_node(task_id,
                   jobID=row.get('jobID'),
                   CPU=row.get('CPU'),
                   RAM=row.get('RAM'),
                   disk=row.get('disk'),
                   Runtime_C1=row.get('Runtime_C1', 0),
                   Runtime_C2=row.get('Runtime_C2', 0),
                   Runtime_C3=row.get('Runtime_C3', 0),
                   deadline=row.get('deadline', 0),
                   task_type=row.get('task_type'),
                   task_category=task_category)

        parent_field = row.get('parent_task')
        parents = []

        if isinstance(parent_field, str) and parent_field.startswith('['):
            try:
                parents = ast.literal_eval(parent_field)
                if not isinstance(parents, list):
                    parents = [parents]
            except (ValueError, SyntaxError):
                print(f"Error parsing parent_task at row {idx}: {parent_field}")
        elif parent_field not in [0, '0', '', None, float('nan')]:
            parents = [parent_field]

        for parent in parents:
            parent_id = str(parent).strip()
            if parent_id not in G.nodes:
                G.add_node(parent_id, task_category="Unknown")
            G.add_edge(parent_id, task_id)

    print("The graph is a DAG!" if nx.is_directed_acyclic_graph(G) else "Warning: The graph is not a DAG.")
    return G

##############################
# Step 3: Visualize Workflow Graph
##############################

def visualize_graph(G, df):
    """
    Visualize the DAG in layers: parent tasks on top, child tasks below.
    Nodes are color-coded by task category.
    """
    import matplotlib.pyplot as plt
    from collections import deque, defaultdict

    # Topological sorting for layer assignment
    in_degrees = dict(G.in_degree())
    zero_in_deg = deque([n for n, deg in in_degrees.items() if deg == 0])
    layers = defaultdict(list)
    node_layers = {}
    
    while zero_in_deg:
        node = zero_in_deg.popleft()
        layer = max([node_layers.get(pred, -1) for pred in G.predecessors(node)], default=-1) + 1
        node_layers[node] = layer
        layers[layer].append(node)
        for succ in G.successors(node):
            in_degrees[succ] -= 1
            if in_degrees[succ] == 0:
                zero_in_deg.append(succ)

    # Assign x, y positions by layer
    pos = {}
    max_width = max(len(nodes) for nodes in layers.values())
    for layer, nodes in layers.items():
        x_spacing = 1.5
        y = -layer
        offset = (max_width - len(nodes)) * x_spacing / 2
        for i, node in enumerate(nodes):
            pos[node] = (i * x_spacing + offset, y)

    # Color settings
    task_type_col = 'K' if 'K' in df.columns else 'task_type'
    unique_types = df[task_type_col].dropna().unique()
    base_colors = ['violet', 'indigo', 'blue', 'green', 'yellow', 'orange', 'red', 'lime', 'magenta', 'cyan']
    color_map = {t: base_colors[i % len(base_colors)] for i, t in enumerate(unique_types)}

    node_colors = [color_map.get(G.nodes[n].get('task_category'), 'gray') for n in G.nodes]

    # Draw
    plt.figure(figsize=(18, 12))
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='black')

    # Legend
    legend = [mpatches.Patch(color=color_map[t], label=f"Task Type: {t}") for t in unique_types]
    if 'gray' in node_colors:
        legend.append(mpatches.Patch(color='gray', label="Task Type: Unknown"))
    plt.legend(handles=legend, loc='upper left', bbox_to_anchor=(1, 1))

    plt.title("Workflow DAG (Layered View)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_resource_usage(G):
    """
    Enhanced 3D visualization of resource usage across tasks with improved clarity
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
            ax.text(x[i], y[i], z[i], task_id, fontsize=8)
    
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

def create_qubo_matrix(G, num_processors=3):
    """
    Create QUBO matrix for workflow scheduling as QAP
    Parameters:
        G: NetworkX DiGraph containing workflow
        num_processors: Number of available processors (default=3 from Runtime_C1/C2/C3)
    Returns:
        QUBO matrix for the scheduling problem
    """

    # Define processor capacity (example values, adjust as needed)
    processor_capacity = {
        'CPU': 8,   # Number of CPU cores per processor
        'RAM': 32,  # RAM in GB per processor
        'disk': 500 # Disk in GB per processor
    }
    
    # Get task IDs and create a mapping to consecutive integers
    task_ids = list(G.nodes())
    task_to_idx = {task_id: idx for idx, task_id in enumerate(task_ids)}
    
    num_tasks = len(task_ids)
    matrix_size = num_tasks * num_processors
    Q = np.zeros((matrix_size, matrix_size))
    
    # 1. Assignment constraints (each task must be assigned to exactly one processor)
    A = 1000  # Penalty weight
    for task in range(num_tasks):
        for p1 in range(num_processors):
            idx1 = task * num_processors + p1
            for p2 in range(num_processors):
                if p1 != p2:
                    idx2 = task * num_processors + p2
                    Q[idx1][idx2] += A
    
    # 2. Task execution costs
    for task_id in task_ids:
        task = G.nodes[task_id]
        task_idx = task_to_idx[task_id]
        for p in range(num_processors):
            idx = task_idx * num_processors + p
            runtime = task[f'Runtime_C{p+1}']
            Q[idx][idx] += runtime
    
    # 3. Precedence constraints from DAG
    B = 1000  # Dependency penalty weight
    for edge in G.edges():
        task1_idx = task_to_idx[edge[0]]
        task2_idx = task_to_idx[edge[1]]
        for p1 in range(num_processors):
            for p2 in range(num_processors):
                idx1 = task1_idx * num_processors + p1
                idx2 = task2_idx * num_processors + p2
                Q[idx1][idx2] += B
    
    # Add resource constraints
    C = 1000  # Resource violation penalty
    for task_id in task_ids:
        task = G.nodes[task_id]
        task_idx = task_to_idx[task_id]
        
        # CPU constraints
        cpu_required = task['CPU']
        for p in range(num_processors):
            idx = task_idx * num_processors + p
            Q[idx][idx] += C * max(0, cpu_required - processor_capacity['CPU'])
            
        # RAM constraints
        ram_required = task['RAM']
        for p in range(num_processors):
            idx = task_idx * num_processors + p
            Q[idx][idx] += C * max(0, ram_required - processor_capacity['RAM'])
            
        # Disk constraints
        disk_required = task['disk']
        for p in range(num_processors):
            idx = task_idx * num_processors + p
            Q[idx][idx] += C * max(0, disk_required - processor_capacity['disk'])
    
    return Q

def create_resource_hamiltonian(G, processor_capacities):
    """Create resource constraint Hamiltonian"""
    num_tasks = len(G.nodes())
    num_processors = 3
    
    # Initialize Hamiltonian terms
    cpu_terms = []
    ram_terms = []
    disk_terms = []
    
    for task_id in G.nodes():
        task = G.nodes[task_id]
        for p in range(num_processors):
            # CPU constraint
            if task['CPU'] > processor_capacities['CPU']:
                cpu_terms.append((f"Z_{task_id}_{p}", 1000))
            
            # RAM constraint    
            if task['RAM'] > processor_capacities['RAM']:
                ram_terms.append((f"Z_{task_id}_{p}", 1000))
                
            # Disk constraint
            if task['disk'] > processor_capacities['disk']:
                disk_terms.append((f"Z_{task_id}_{p}", 1000))
    
    return cpu_terms + ram_terms + disk_terms

def create_qaoa_circuit(Q, p=1):
    """
    Create QAOA solver and QuadraticProgram from QUBO matrix.
    """
    num_vars = Q.shape[0]
    qp = QuadraticProgram()
    
    # Create binary variables
    for i in range(num_vars):
        qp.binary_var(name=f'x_{i}')
    
    # Set up the objective function
    linear = {f'x_{i}': Q[i, i] for i in range(num_vars)}
    quadratic = {}
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            if Q[i, j] != 0:
                quadratic[(f'x_{i}', f'x_{j}')] = Q[i, j] + Q[j, i]
    
    qp.minimize(linear=linear, quadratic=quadratic)
    
    # Updated QAOA setup for latest Qiskit
    backend = AerSimulator()
    sampler = Sampler()  # Changed from SamplerV1
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=p)
    solver = MinimumEigenOptimizer(qaoa)
    
    return solver, qp

def solve_workflow_scheduling(G, Q):
    """
    Solve workflow scheduling using QAOA with fallback to classical solver
    """
    num_tasks = len(G.nodes())
    num_processors = 3
    matrix_size = num_tasks * num_processors
    
    # Check problem size
    MAX_QUBITS = 20  # Maximum practical size for QAOA
    
    if matrix_size > MAX_QUBITS:
        print(f"Problem size ({matrix_size} qubits) exceeds maximum allowed dimension ({MAX_QUBITS})")
        print("Falling back to classical solver...")
        return solve_classical(G, Q)
    
    try:
        solver, qp = create_qaoa_circuit(Q, p=1)
        result = solver.solve(qp)
        
        assignments = {}
        x = result.x
        for task_idx in range(num_tasks):
            for p in range(num_processors):
                idx = task_idx * num_processors + p
                if x[idx] > 0.5:
                    task_id = list(G.nodes())[task_idx]
                    assignments[task_id] = f"Processor {p+1}"
        
        return assignments, result.fval
    except Exception as e:
        print(f"Error in quantum optimization: {e}")
        return solve_classical(G, Q)

def solve_classical(G, Q):
    """
    Classical fallback solver using simulated annealing
    """
    from scipy.optimize import minimize
    
    num_tasks = len(G.nodes())
    num_processors = 3
    
    # Initial solution: round-robin assignment
    x0 = np.zeros(Q.shape[0])
    for i in range(num_tasks):
        x0[i * num_processors + (i % num_processors)] = 1
    
    # Objective function
    def objective(x):
        return x.T @ Q @ x
    
    # Solve using simulated annealing
    result = minimize(objective, x0, method='Nelder-Mead', 
                     options={'maxiter': 1000})
    
    # Convert solution to assignments
    assignments = {}
    x = np.round(result.x)  # Round to binary
    
    for task_idx in range(num_tasks):
        for p in range(num_processors):
            idx = task_idx * num_processors + p
            if x[idx] > 0.5:
                task_id = list(G.nodes())[task_idx]
                assignments[task_id] = f"Processor {p+1}"
    
    return assignments, result.fun

def analyze_schedule(assignments, G):
    """Enhanced schedule analysis with detailed metrics and visualizations"""
    # Calculate processor-specific metrics
    processor_metrics = defaultdict(lambda: {
        'load': 0,
        'tasks': 0,
        'cpu_usage': 0,
        'ram_usage': 0,
        'disk_usage': 0,
        'task_list': []
    })

    for task, processor in assignments.items():
        node = G.nodes[task]
        metrics = processor_metrics[processor]
        metrics['load'] += node['Runtime_C1']
        metrics['tasks'] += 1
        metrics['cpu_usage'] += node['CPU']
        metrics['ram_usage'] += node['RAM']
        metrics['disk_usage'] += node['disk']
        metrics['task_list'].append(task)

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Processor Loads Comparison
    procs = list(processor_metrics.keys())
    loads = [metrics['load'] for metrics in processor_metrics.values()]
    ax1.bar(procs, loads, color='skyblue')
    ax1.set_title('Processor Loads')
    ax1.set_ylabel('Total Runtime')
    for i, v in enumerate(loads):
        ax1.text(i, v, f'{v:.1f}', ha='center', va='bottom')

    # 2. Resource Usage per Processor
    width = 0.25
    x = np.arange(len(procs))
    
    cpu_usage = [metrics['cpu_usage'] for metrics in processor_metrics.values()]
    ram_usage = [metrics['ram_usage'] for metrics in processor_metrics.values()]
    disk_usage = [metrics['disk_usage'] for metrics in processor_metrics.values()]
    
    ax2.bar(x - width, cpu_usage, width, label='CPU', color='lightcoral')
    ax2.bar(x, ram_usage, width, label='RAM', color='lightgreen')
    ax2.bar(x + width, disk_usage, width, label='Disk', color='lightskyblue')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(procs)
    ax2.set_title('Resource Usage per Processor')
    ax2.legend()

    # 3. Task Distribution
    tasks_per_proc = [metrics['tasks'] for metrics in processor_metrics.values()]
    ax3.pie(tasks_per_proc, labels=procs, autopct='%1.1f%%',
            colors=plt.cm.Pastel1(np.linspace(0, 1, len(procs))))
    ax3.set_title('Task Distribution')

    # 4. Load Balance Metrics
    metrics_text = "Load Balance Metrics:\n\n"
    
    total_load = sum(loads)
    avg_load = total_load / len(loads)
    load_imbalance = max(loads) / min(loads)
    
    metrics_text += f"Total Load: {total_load:.1f}\n"
    metrics_text += f"Average Load: {avg_load:.1f}\n"
    metrics_text += f"Load Imbalance Ratio: {load_imbalance:.2f}\n"
    metrics_text += f"Coefficient of Variation: {np.std(loads)/np.mean(loads):.2f}\n\n"
    
    for proc, metrics in processor_metrics.items():
        metrics_text += f"\n{proc}:\n"
        metrics_text += f"  Tasks: {metrics['tasks']}\n"
        metrics_text += f"  Load: {metrics['load']:.1f}\n"
        
    ax4.text(0.05, 0.95, metrics_text,
             transform=ax4.transAxes,
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    ax4.axis('off')

    plt.tight_layout()
    plt.show()

    # Print additional analysis
    print("\nDetailed Schedule Analysis:")
    print(f"Load Balance Score: {min(loads)/max(loads):.2f}")
    print(f"Standard Deviation of Loads: {np.std(loads):.2f}")
    print("\nProcessor Assignments:")
    for proc, metrics in processor_metrics.items():
        print(f"\n{proc}:")
        print(f"  Number of tasks: {metrics['tasks']}")
        print(f"  Total load: {metrics['load']:.2f}")
        print(f"  Tasks: {', '.join(metrics['task_list'])}")

def visualize_schedule_timeline(assignments, G):
    """Create enhanced timeline visualization of task schedule with better clarity and more information"""
    processors = defaultdict(list)
    for task, proc in assignments.items():
        processors[proc].append({
            'task': task,
            'runtime': G.nodes[task]['Runtime_C1'],
            'cpu': G.nodes[task]['CPU'],
            'ram': G.nodes[task]['RAM'],
            'disk': G.nodes[task]['disk']
        })

    # Calculate total time span
    max_time = max(sum(task['runtime'] for task in tasks) 
                  for tasks in processors.values())

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])

    # 1. Enhanced Timeline
    ax1 = fig.add_subplot(gs[0])
    colors = plt.cm.Set3(np.linspace(0, 1, len(G.nodes())))
    
    for i, (proc, tasks) in enumerate(processors.items()):
        y = i
        current_time = 0
        for task in tasks:
            # Create task block
            ax1.barh(y, task['runtime'], left=current_time, 
                    color=colors[list(G.nodes()).index(task['task'])],
                    alpha=0.7,
                    edgecolor='black')
            
            # Add task labels
            ax1.text(current_time + task['runtime']/2, y,
                    f"{task['task']}\n({task['runtime']:.1f})",
                    va='center', ha='center',
                    fontsize=8)
            
            current_time += task['runtime']

    ax1.set_yticks(range(len(processors)))
    ax1.set_yticklabels(processors.keys())
    ax1.set_xlabel('Time Units')
    ax1.set_title('Enhanced Schedule Timeline', pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_time * 1.1)

    # 2. Resource Usage Over Time
    ax2 = fig.add_subplot(gs[1])
    
    for proc, tasks in processors.items():
        times = [0]
        cpu_usage = [0]
        ram_usage = [0]
        
        current_time = 0
        current_cpu = 0
        current_ram = 0
        
        for task in tasks:
            current_cpu += task['cpu']
            current_ram += task['ram']
            
            times.extend([current_time, current_time])
            cpu_usage.extend([current_cpu, current_cpu])
            ram_usage.extend([current_ram, current_ram])
            
            current_time += task['runtime']
            
            times.append(current_time)
            cpu_usage.append(current_cpu)
            ram_usage.append(current_ram)
            
            current_cpu -= task['cpu']
            current_ram -= task['ram']

        ax2.plot(times, cpu_usage, '-', label=f'{proc} CPU', alpha=0.7)
        ax2.plot(times, ram_usage, '--', label=f'{proc} RAM', alpha=0.7)

    ax2.set_xlabel('Time Units')
    ax2.set_ylabel('Resource Units')
    ax2.set_title('Resource Usage Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_xlim(0, max_time * 1.1)

    # 3. Cumulative Work Distribution
    ax3 = fig.add_subplot(gs[2])
    
    for proc, tasks in processors.items():
        times = [0]
        cumulative_work = [0]
        current_time = 0
        total_work = 0
        
        for task in tasks:
            times.append(current_time)
            cumulative_work.append(total_work)
            
            current_time += task['runtime']
            total_work += task['runtime']
            
            times.append(current_time)
            cumulative_work.append(total_work)
            
        ax3.plot(times, cumulative_work, '-o', label=proc, alpha=0.7)

    ax3.set_xlabel('Time Units')
    ax3.set_ylabel('Cumulative Work')
    ax3.set_title('Cumulative Work Distribution')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.set_xlim(0, max_time * 1.1)

    plt.tight_layout()
    plt.show()

def print_workflow_statistics(G):
    """Print detailed workflow statistics"""
    print("\n=== Workflow Statistics ===")
    
    # Basic graph statistics
    print("\nGraph Structure:")
    print(f"Number of tasks: {G.number_of_nodes()}")
    print(f"Number of dependencies: {G.number_of_edges()}")
    print(f"Graph density: {nx.density(G):.3f}")
    
    # Resource statistics
    print("\nResource Statistics:")
    for resource in ['CPU', 'RAM', 'disk']:
        values = list(nx.get_node_attributes(G, resource).values())
        print(f"\n{resource}:")
        print(f"  Mean: {np.mean(values):.2f}")
        print(f"  Median: {np.median(values):.2f}")
        print(f"  Std Dev: {np.std(values):.2f}")
        print(f"  Min: {min(values):.2f}")
        print(f"  Max: {max(values):.2f}")
    
    # Runtime statistics
    print("\nRuntime Statistics:")
    for proc in ['Runtime_C1', 'Runtime_C2', 'Runtime_C3']:
        values = list(nx.get_node_attributes(G, proc).values())
        print(f"\n{proc}:")
        print(f"  Total: {sum(values):.2f}")
        print(f"  Mean: {np.mean(values):.2f}")
        print(f"  Median: {np.median(values):.2f}")
        print(f"  Std Dev: {np.std(values):.2f}")
    
    # Dependency analysis
    print("\nDependency Analysis:")
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    print(f"Max in-degree: {max(in_degrees.values())}")
    print(f"Max out-degree: {max(out_degrees.values())}")
    print(f"Isolated tasks: {len([n for n in G.nodes() if G.degree(n) == 0])}")
    
    # Critical path analysis
    critical_path = nx.dag_longest_path(G)
    cp_length = sum(G.nodes[node]['Runtime_C1'] for node in critical_path)
    print(f"\nCritical Path Length: {cp_length:.2f}")
    print(f"Critical Path Tasks: {' → '.join(critical_path)}")

def visualize_workflow_optimization(G, assignments):
    """Create a before/after visualization of workflow optimization"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Calculate layered layout once and reuse
    # Topological sorting for layer assignment
    in_degrees = dict(G.in_degree())
    zero_in_deg = deque([n for n, deg in in_degrees.items() if deg == 0])
    layers = defaultdict(list)
    node_layers = {}
    
    while zero_in_deg:
        node = zero_in_deg.popleft()
        layer = max([node_layers.get(pred, -1) for pred in G.predecessors(node)], default=-1) + 1
        node_layers[node] = layer
        layers[layer].append(node)
        for succ in G.successors(node):
            in_degrees[succ] -= 1
            if in_degrees[succ] == 0:
                zero_in_deg.append(succ)

    # Assign x, y positions by layer
    pos = {}
    max_width = max(len(nodes) for nodes in layers.values())
    for layer, nodes in layers.items():
        x_spacing = 1.5
        y = -layer
        offset = (max_width - len(nodes)) * x_spacing / 2
        for i, node in enumerate(nodes):
            pos[node] = (i * x_spacing + offset, y)
    
    # Original workflow
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='gray', arrows=True)
    nx.draw_networkx_nodes(G, pos, ax=ax1, 
                          node_color='lightblue',
                          node_size=500)
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8)
    ax1.set_title('Original Workflow', pad=20)
    
    # Optimized workflow with same layout
    # Group nodes by processor assignment
    processor_groups = defaultdict(list)
    for task, proc in assignments.items():
        processor_groups[proc].append(task)
    
    # Draw nodes with different colors for each processor
    colors = ['#FF9999', '#99FF99', '#9999FF']  # Red, Green, Blue tints
    for i, (proc, tasks) in enumerate(processor_groups.items()):
        nx.draw_networkx_nodes(G, pos, ax=ax2,
                             nodelist=tasks,
                             node_color=colors[i],
                             node_size=500,
                             label=proc)
    
    nx.draw_networkx_edges(G, pos, ax=ax2, edge_color='gray', arrows=True)
    nx.draw_networkx_labels(G, pos, ax=ax2, font_size=8)
    
    # Add processor assignment legend
    ax2.legend(fontsize=10)
    ax2.set_title('Optimized Workflow (Tasks Colored by Processor Assignment)', pad=20)
    
    # Add optimization metrics
    processor_loads = defaultdict(float)
    for task, proc in assignments.items():
        processor_loads[proc] += G.nodes[task]['Runtime_C1']
    
    metrics = f"Optimization Metrics:\n"
    metrics += f"Load Balance Ratio: {min(processor_loads.values())/max(processor_loads.values()):.2f}\n"
    metrics += f"Max Load: {max(processor_loads.values()):.2f}\n"
    metrics += f"Min Load: {min(processor_loads.values()):.2f}"
    
    fig.text(0.02, 0.02, metrics, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Set the same axis limits for both plots
    ax1.set_axis_off()
    ax2.set_axis_off()
    plt.tight_layout()
    plt.show()
def main():
    filepath = "workflow.csv"
    df = load_data(filepath)
    G = build_graph(df)
    

    print_workflow_statistics(G)
    visualize_graph(G, df)
    
  
   
    visualize_resource_usage(G)
    
    Q = create_qubo_matrix(G)
    assignments, objective = solve_workflow_scheduling(G, Q)
    
    if assignments:
        print(f"\nOptimal Schedule (objective value: {objective:.2f}):")
        for task, processor in sorted(assignments.items()):
            print(f"Task {task} -> {processor}")
        visualize_schedule_timeline(assignments, G)
        analyze_schedule(assignments, G)
        visualize_workflow_optimization(G, assignments)
    else:
        print("Failed to find optimal schedule")

if __name__ == "__main__":
    main()
