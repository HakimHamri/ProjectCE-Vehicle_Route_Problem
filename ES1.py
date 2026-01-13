import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Forces Matplotlib to use a non-interactive backend
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="VRP with Evolution Strategy", layout="wide")

st.title("Vehicle Routing Problem - Evolution Strategy")

# =========================
# DATA INPUT
# =========================
st.subheader("Upload CSV Data")
uploaded_file = st.file_uploader("Upload CSV with columns: node_id, x, y, demand, node_type, vehicle_capacity", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    depot = df[df['node_type'] == 'depot'].iloc[0]
    customers = df[df['node_type'] == 'customer'].copy()
    capacity = customers['vehicle_capacity'].iloc[0]

    coords = df[['x', 'y']].values
    node_ids = df['node_id'].values

    dist_matrix = np.sqrt(
        np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    )

    # =========================
    # FITNESS & ROUTE FUNCTIONS
    # =========================
    def calculate_fitness(permutation):
        total_distance = 0
        current_load = 0
        current_node = depot['node_id']

        for cust_id in permutation:
            demand = customers.loc[customers['node_id'] == cust_id, 'demand'].values[0]
            if current_load + demand > capacity:
                total_distance += dist_matrix[current_node, depot['node_id']]
                current_node = depot['node_id']
                current_load = 0

            total_distance += dist_matrix[current_node, cust_id]
            current_node = cust_id
            current_load += demand

        total_distance += dist_matrix[current_node, depot['node_id']]
        return total_distance

    def get_routes(permutation):
        routes = []
        current_route = []
        current_load = 0

        for cust_id in permutation:
            demand = customers.loc[customers['node_id'] == cust_id, 'demand'].values[0]
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
    # STREAMLIT PARAMETERS
    # =========================
    st.sidebar.subheader("ES Parameters")
    mu = st.sidebar.number_input("Parent Population (mu)", value=20, min_value=1)
    lam = st.sidebar.number_input("Offspring Population (lambda)", value=200, min_value=1)
    generations = st.sidebar.number_input("Generations", value=200, min_value=1)

    # =========================
    # RUN BUTTON
    # =========================
    if st.button("Run Evolution Strategy"):

        customer_ids = customers['node_id'].values
        num_customers = len(customer_ids)

        population = [np.random.permutation(customer_ids) for _ in range(mu)]
        fitnesses = [calculate_fitness(ind) for ind in population]
        history = []

        start_time = time.time()

        for gen in range(generations):
            offspring = []
            for _ in range(lam):
                parent = population[np.random.randint(mu)]
                child = parent.copy()
                i, j = sorted(np.random.choice(num_customers, 2, replace=False))
                child[i:j] = child[i:j][::-1]
                offspring.append(child)

            offspring_fitness = [calculate_fitness(ind) for ind in offspring]

            combined_pop = population + offspring
            combined_fit = fitnesses + offspring_fitness

            best_idx = np.argsort(combined_fit)[:mu]
            population = [combined_pop[i] for i in best_idx]
            fitnesses = [combined_fit[i] for i in best_idx]
            history.append(fitnesses[0])

        runtime = time.time() - start_time

        best_individual = population[0]
        best_fitness = fitnesses[0]
        best_routes = get_routes(best_individual)

        # =========================
        # RESULTS
        # =========================
        st.subheader("Results")
        st.write(f"**Best Distance:** {best_fitness:.2f}")
        st.write(f"**Number of Routes:** {len(best_routes)}")
        st.write(f"**Runtime:** {runtime:.2f}s")

        for i, route in enumerate(best_routes):
            st.write(f"Route {i+1}: Depot -> {' -> '.join(map(str, route))} -> Depot")

        # =========================
        # CONVERGENCE PLOT
        # =========================
        st.subheader("Convergence Plot")
        fig, ax = plt.subplots()
        ax.plot(history)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Distance")
        ax.set_title("Evolution Strategy Convergence")
        st.pyplot(fig)

        # =========================
        # ROUTE PLOT
        # =========================
        st.subheader("Best Routes")
        fig2, ax2 = plt.subplots()
        ax2.scatter(customers['x'], customers['y'], label='Customers')
        ax2.scatt
