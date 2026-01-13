import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="VRP Evolution Strategy", layout="wide")

st.title("🚚 Vehicle Routing Problem (VRP)")
st.subheader("Evolution Strategy with Local Search (Best Performance)")

# =========================
# Upload Dataset
# =========================
uploaded_file = st.file_uploader("Upload VRP CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    depot = df[df['node_type'] == 'depot'].iloc[0]
    customers = df[df['node_type'] == 'customer'].copy()
    capacity = df['vehicle_capacity'].iloc[0]

    nodes = df[['x', 'y']].values
    dist_matrix = np.sqrt(
        np.sum((nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]) ** 2, axis=2)
    )
    demands = df.set_index('node_id')['demand'].to_dict()

    # =========================
    # Functions
    # =========================
    def calculate_route_distance(route):
        if not route:
            return 0
        d = dist_matrix[0, route[0]]
        for i in range(len(route) - 1):
            d += dist_matrix[route[i], route[i + 1]]
        d += dist_matrix[route[-1], 0]
        return d

    def decode_and_eval(permutation):
        routes = []
        current_route = []
        current_load = 0
        total_dist = 0

        for node_id in permutation:
            demand = demands[node_id]
            if current_load + demand <= capacity:
                current_route.append(node_id)
                current_load += demand
            else:
                total_dist += calculate_route_distance(current_route)
                routes.append(current_route)
                current_route = [node_id]
                current_load = demand

        if current_route:
            total_dist += calculate_route_distance(current_route)
            routes.append(current_route)

        return total_dist, routes

    def two_opt_permutation(perm):
        best_perm = list(perm)
        best_dist, _ = decode_and_eval(best_perm)
        improved = True

        while improved:
            improved = False
            for i in range(len(best_perm) - 1):
                for j in range(i + 1, len(best_perm)):
                    new_perm = best_perm[:i] + best_perm[i:j+1][::-1] + best_perm[j+1:]
                    new_dist, _ = decode_and_eval(new_perm)
                    if new_dist < best_dist:
                        best_dist = new_dist
                        best_perm = new_perm
                        improved = True
        return best_perm

    def best_performance_es(mu=20, lambda_=100, generations=1000):
        cust_ids = customers['node_id'].tolist()
        pop = [np.random.permutation(cust_ids).tolist() for _ in range(mu)]
        pop_scores = [decode_and_eval(p)[0] for p in pop]

        history = []

        for gen in range(generations):
            offspring = []
            for _ in range(lambda_):
                parent = pop[np.random.randint(mu)]
                child = list(parent)

                r = np.random.rand()
                if r < 0.4:
                    i, j = np.random.choice(len(child), 2, replace=False)
                    child[i], child[j] = child[j], child[i]
                elif r < 0.8:
                    a, b = sorted(np.random.choice(len(child), 2, replace=False))
                    child[a:b] = child[a:b][::-1]
                else:
                    a, b = sorted(np.random.choice(len(child), 2, replace=False))
                    segment = child[a:b]
                    del child[a:b]
                    pos = np.random.randint(0, len(child) + 1)
                    for s in reversed(segment):
                        child.insert(pos, s)

                offspring.append(child)

            offspring_scores = [decode_and_eval(p)[0] for p in offspring]

            combined = pop + offspring
            combined_scores = pop_scores + offspring_scores

            idx = np.argsort(combined_scores)[:mu]
            pop = [combined[i] for i in idx]
            pop_scores = [combined_scores[i] for i in idx]

            if gen % 100 == 0:
                pop[0] = two_opt_permutation(pop[0])
                pop_scores[0], _ = decode_and_eval(pop[0])

            history.append(pop_scores[0])

        best_score, best_routes = decode_and_eval(pop[0])
        return best_score, best_routes, history

    # =========================
    # Sidebar Controls
    # =========================
    st.sidebar.header("⚙️ Parameters")
    mu = st.sidebar.slider("μ (Population)", 10, 50, 20)
    lambda_ = st.sidebar.slider("λ (Offspring)", 50, 200, 100)
    generations = st.sidebar.slider("Generations", 100, 2000, 1000)

    # =========================
    # Run Button
    # =========================
    if st.button("🚀 Run Evolution Strategy"):
        with st.spinner("Optimizing routes..."):
            start = time.time()
            best_score, best_routes, history = best_performance_es(
                mu, lambda_, generations
            )
            exec_time = time.time() - start

        st.success("Optimization Completed!")

        # =========================
        # Results Summary
        # =========================
        st.metric("Best Total Distance", f"{best_score:.4f}")
        st.metric("Vehicles Used", len(best_routes))
        st.metric("Execution Time (s)", f"{exec_time:.2f}")

        # =========================
        # Plot Routes
        # =========================
        st.subheader("📍 Best Route Visualization")
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(depot['x'], depot['y'], c='red', marker='s', s=150, label='Depot')
        ax.scatter(customers['x'], customers['y'], c='blue', s=50, label='Customers')

        colors = plt.cm.tab10(np.linspace(0, 1, len(best_routes)))
        for i, route in enumerate(best_routes):
            coords = [(depot['x'], depot['y'])]
            for node_id in route:
                node = df[df['node_id'] == node_id].iloc[0]
                coords.append((node['x'], node['y']))
            coords.append((depot['x'], depot['y']))

            xs, ys = zip(*coords)
            ax.plot(xs, ys, color=colors[i], linewidth=2, label=f'Vehicle {i+1}')

        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # =========================
        # Convergence Plot
        # =========================
        st.subheader("📉 Convergence Curve")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(history)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Total Distance")
        ax2.grid(True)
        st.pyplot(fig2)

else:
    st.info("👆 Please upload a VRP CSV dataset to start.")
