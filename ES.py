import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# =========================
# LOAD DATA (example)
# =========================
# df must contain:
# node_id | x | y | demand | node_type | vehicle_capacity

# Example dummy data (REMOVE if you already load df)
"""
df = pd.DataFrame({
    'node_id': [0,1,2,3,4,5],
    'x': [50,20,60,30,70,10],
    'y': [50,40,20,60,10,80],
    'demand': [0,10,15,20,10,5],
    'node_type': ['depot','customer','customer','customer','customer','customer'],
    'vehicle_capacity': [0,30,30,30,30,30]
})
"""

# =========================
# PREPROCESSING
# =========================
depot = df[df['node_type'] == 'depot'].iloc[0]
customers = df[df['node_type'] == 'customer'].copy()
capacity = customers['vehicle_capacity'].iloc[0]

coords = df[['x', 'y']].values
node_ids = df['node_id'].values

# Distance Matrix
dist_matrix = np.sqrt(
    np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
)

# =========================
# FITNESS FUNCTION
# =========================
def calculate_fitness(permutation):
    total_distance = 0
    current_load = 0
    current_node = depot['node_id']

    for cust_id in permutation:
        demand = customers.loc[
            customers['node_id'] == cust_id, 'demand'
        ].values[0]

        if current_load + demand > capacity:
            total_distance += dist_matrix[current_node, depot['node_id']]
            current_node = depot['node_id']
            current_load = 0

        total_distance += dist_matrix[current_node, cust_id]
        current_node = cust_id
        current_load += demand

    total_distance += dist_matrix[current_node, depot['node_id']]
    return total_distance

# =========================
# ROUTE CONSTRUCTION
# =========================
def get_routes(permutation):
    routes = []
    current_route = []
    current_load = 0

    for cust_id in permutation:
        demand = customers.loc[
            customers['node_id'] == cust_id, 'demand'
        ].values[0]

        if current_load + demand > capacity:
            routes.append(current_route)
            current_route = []
            current_load = 0

        current_route.append(cust_id)
        current_load += demand

    if current_route:
        routes.append(current_route)

    return routes

# =========================
# EVOLUTION STRATEGY PARAMS
# =========================
mu = 20
lam = 200
generations = 200

customer_ids = customers['node_id'].values
num_customers = len(customer_ids)

# =========================
# INITIALIZATION
# =========================
population = [np.random.permutation(customer_ids) for _ in range(mu)]
fitnesses = [calculate_fitness(ind) for ind in population]

history = []

# =========================
# START TIMER
# =========================
start_time = time.time()

# =========================
# EVOLUTION LOOP
# =========================
for gen in range(generations):
    offspring = []

    for _ in range(lam):
        parent = population[np.random.randint(mu)]
        child = parent.copy()

        # Inversion Mutation
        i, j = sorted(np.random.choice(num_customers, 2, replace=False))
        child[i:j] = child[i:j][::-1]

        offspring.append(child)

    offspring_fitness = [calculate_fitness(ind) for ind in offspring]

    # (mu + lambda) selection
    combined_pop = population + offspring
    combined_fit = fitnesses + offspring_fitness

    best_idx = np.argsort(combined_fit)[:mu]
    population = [combined_pop[i] for i in best_idx]
    fitnesses = [combined_fit[i] for i in best_idx]

    history.append(fitnesses[0])

# =========================
# END TIMER
# =========================
runtime = time.time() - start_time

# =========================
# RESULTS
# =========================
best_individual = population[0]
best_fitness = fitnesses[0]
best_routes = get_routes(best_individual)

# =========================
# VISUALIZATION 1: CONVERGENCE
# =========================
plt.figure(figsize=(10, 5))
plt.plot(history)
plt.xlabel("Generation")
plt.ylabel("Best Distance")
plt.title("Evolution Strategy Convergence")
plt.grid()
plt.savefig("convergence.png")
plt.show()

# =========================
# VISUALIZATION 2: BEST ROUTE
# =========================
plt.figure(figsize=(10, 10))
plt.scatter(customers['x'], customers['y'], label='Customers')
plt.scatter(depot['x'], depot['y'], marker='s', s=150, label='Depot')

colors = plt.cm.get_cmap('tab20', len(best_routes))
for i, route in enumerate(best_routes):
    full_route = [depot['node_id']] + route + [depot['node_id']]
    route_coords = coords[full_route]
    plt.plot(route_coords[:, 0], route_coords[:, 1], marker='o',
             label=f'Route {i+1}', color=colors(i))

plt.title(f"Best VRP Solution (Distance = {best_fitness:.2f})")
plt.legend()
plt.grid()
plt.savefig("vrp_solution.png")
plt.show()

# =========================
# PRINT SUMMARY
# =========================
print("\n========== VRP EVOLUTION STRATEGY RESULTS ==========")
print(f"Best Distance        : {best_fitness:.4f}")
print(f"Number of Routes Used: {len(best_routes)}")
print(f"Runtime (seconds)    : {runtime:.2f}s\n")

print("Routes Detail:")
for i, route in enumerate(best_routes):
    print(f"Route {i+1}: Depot -> {' -> '.join(map(str, route))} -> Depot")
