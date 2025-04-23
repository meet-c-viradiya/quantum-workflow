from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
import numpy as np
import networkx as nx

def create_qubo_matrix(G, num_processors=3):
    # Define processor capacity (example values, adjust as needed)
    processor_capacity = {
        'CPU': 8,   # Number of CPU cores per processor
        'RAM': 32,  # RAM in GB per processor
        'disk': 500 # Disk in GB per processor
    }
    
    task_ids = list(G.nodes())
    task_to_idx = {task_id: idx for idx, task_id in enumerate(task_ids)}
    
    num_tasks = len(task_ids)
    matrix_size = num_tasks * num_processors
    Q = np.zeros((matrix_size, matrix_size))
    
    # Assignment constraints
    A = 1000
    for task in range(num_tasks):
        for p1 in range(num_processors):
            idx1 = task * num_processors + p1
            for p2 in range(num_processors):
                if p1 != p2:
                    Q[idx1][task * num_processors + p2] = A
    
    # Task execution costs
    for task_id in task_ids:
        task = G.nodes[task_id]
        task_idx = task_to_idx[task_id]
        for p in range(num_processors):
            idx = task_idx * num_processors + p
            runtime = task[f'Runtime_C{p+1}']
            Q[idx][idx] += runtime
    
    # Precedence constraints from DAG
    B = 1000
    for edge in G.edges():
        task1_idx = task_to_idx[edge[0]]
        task2_idx = task_to_idx[edge[1]]
        for p1 in range(num_processors):
            for p2 in range(num_processors):
                Q[task1_idx * num_processors + p1][task2_idx * num_processors + p2] += B
    
    # Resource constraints
    C = 1000
    for task_id in task_ids:
        task = G.nodes[task_id]
        task_idx = task_to_idx[task_id]
        
        cpu_required = task['CPU']
        for p in range(num_processors):
            idx = task_idx * num_processors + p
            Q[idx][idx] += C * max(0, cpu_required - processor_capacity['CPU'])
        
        ram_required = task['RAM']
        for p in range(num_processors):
            idx = task_idx * num_processors + p
            Q[idx][idx] += C * max(0, ram_required - processor_capacity['RAM'])
        
        disk_required = task['disk']
        for p in range(num_processors):
            idx = task_idx * num_processors + p
            Q[idx][idx] += C * max(0, disk_required - processor_capacity['disk'])
    
    return Q

def solve_workflow_scheduling(G, Q):
    num_tasks = len(G.nodes())
    num_processors = 3
    matrix_size = num_tasks * num_processors
    
    MAX_QUBITS = 20
    
    if matrix_size > MAX_QUBITS:
        print(f"Problem size ({matrix_size} qubits) exceeds maximum allowed dimension ({MAX_QUBITS})")
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
    num_tasks = len(G.nodes())
    num_processors = 3
    
    x0 = np.zeros(Q.shape[0])
    for i in range(num_tasks):
        x0[i * num_processors + (i % num_processors)] = 1
    
    def objective(x):
        return x.T @ Q @ x
    
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000})
    
    assignments = {}
    x = np.round(result.x)
    
    for task_idx in range(num_tasks):
        for p in range(num_processors):
            idx = task_idx * num_processors + p
            if x[idx] > 0.5:
                task_id = list(G.nodes())[task_idx]
                assignments[task_id] = f"Processor {p+1}"
    
    return assignments, result.fun