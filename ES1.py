import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="VRP with Evolution Strategy", layout="wide")
st.title("Vehicle Routing Problem - Evolution Strategy")

# ------------------------
# Upload CSV
# ------------------------
uploaded_file = st.file_uploader(
    "Upload CSV with columns: node_id, x, y, demand, node_type, vehicle_capacity",
    type="csv"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    depot = df[df['node_type'] == 'depot'].iloc[0]
    customers = df[df['node_type'] == 'customer'].copy()
    capacity = customers['vehicle_capacity'].iloc[0]

    coords = df[['x','y']].values
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

    # ------------------------
    # Fitness function
    # ------------------------
    def fitness(perm):
        total, load, node = 0, 0, depot['node_id']
        for c in perm:
            demand = customers.loc[customers['node_id']==c, 'demand'].values[0]
            if load + demand > capacity:
                total += dist_matrix[node, depot['node_id']]
                node, load = depot['node_id'], 0
            total += dist_matrix[node, c]
            node, load = c, load + demand
        return total + dist_matrix[node, depot['node_id']]

    # ------------------------
    # Convert permutation to routes
    # ------------------------
    def get_routes(perm):
        routes, route, load = [], [], 0
        for c in perm:
            demand = customers.loc[customers['node_id']==c, 'demand'].values[0]
            if load + demand > capacity:
                routes.append(route)
                route, load = [], 0
            route.append(c)
            load += demand
        if route: routes.append(route)
        return routes

    # ------------------------
    # ES parameters
    # ------------------------
    st.sidebar.subheader("Evolution Strategy Parameters")
    mu = st.sidebar.number_input("Parent population (mu)", value=20, min_value=1)
    lam = st.sidebar.number_input("Offspring population (lambda)", value=200, min_value=1)
    generations = st.sidebar.number_input("Generations", value=200, min_value=1)

    # ------------------------
    # Run Evolution Strategy
    # ------------------------
    if st.button("Run Evolution Strategy"):
        ids = customers['node_id'].values
        pop = [np.random.permutation(ids) for _ in range(mu)]
        fitnesses = [fitness(ind) for ind in pop]
        history = []

        start_time = time.time()
        for _ in range(generations):
            offspring = []
            for _ in range(lam):
                p = pop[np.random.randint(mu)].copy()
                i, j = sorted(np.random.choice(len(ids), 2, replace=False))
                p[i:j] = p[i:j][::-1]  # 2-opt swap
                offspring.append(p)
            off_fit = [fitness(c) for c in offspring]
            combined = pop + offspring
            combined_fit = fitnesses + off_fit
            best_idx = np.argsort(combined_fit)[:mu]
            pop = [combined[i] for i in best_idx]
            fitnesses = [combined_fit[i] for i in best_idx]
            history.append(fitnesses[0])

        runtime = time.time() - start_time
        best_ind = pop[0]
        best_fit = fitnesses[0]
        best_routes = get_routes(best_ind)

        # ------------------------
        # Results
        # ------------------------
        st.subheader("Results")
        st.write(f"**Best Distance:** {best_fit:.2f}")
        st.write(f"**Number of Routes:** {len(best_routes)}")
        st.write(f"**Runtime:** {runtime:.2f}s")

        for i, r in enumerate(best_routes):
            st.write(f"Route {i+1}: Depot -> {' -> '.join(map(str,r))} -> Depot")

        # ------------------------
        # Convergence plot
        # ------------------------
        st.subheader("Convergence Plot")
        fig, ax = plt.subplots()
        ax.plot(history)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Distance")
        ax.set_title("Evolution Strategy Convergence")
        st.pyplot(fig)

        # ------------------------
        # Route plot
        # ------------------------
        st.subheader("Best Routes")
        fig2, ax2 = plt.subplots()
        ax2.scatter(customers['x'], customers['y'], label='Customers')
        ax2.scatter(depot['x'], depot['y'], color='red', label='Depot')
        colors = plt.cm.tab20.colors

        for idx, route in enumerate(best_routes):
            x = [depot['x']] + [customers.loc[customers['node_id']==c,'x'].values[0] for c in route] + [depot['x']]
            y = [depot['y']] + [customers.loc[customers['node_id']==c,'y'].values[0] for c in route] + [depot['y']]
            ax2.plot(x, y, color=colors[idx % len(colors)], label=f'Route {idx+1}')

        ax2.legend()
        st.pyplot(fig2)
