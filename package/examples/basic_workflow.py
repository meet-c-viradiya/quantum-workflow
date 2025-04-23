from src.core.loader import load_data
from src.core.graph import build_graph
from src.optimization.qaoa import solve_workflow_scheduling
from src.visualization.graph_viz import visualize_graph
from src.visualization.schedule_viz import visualize_schedule_timeline

# Load workflow data
data = load_data('data/workflow.csv')

# Build the workflow graph
G = build_graph(data)

# Solve the scheduling problem
assignments, cost = solve_workflow_scheduling(G)

# Visualize the workflow graph
visualize_graph(G, data)

# Visualize the schedule timeline
visualize_schedule_timeline(assignments, G)

# Print the assignments and cost
print("Assignments:", assignments)
print("Total Cost:", cost)