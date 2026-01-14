import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# =========================
# Streamlit Page Config
# =========================
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
    # FUNCTIONS
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
            if current_load + demands[node_id] <= capacity:
                current_route.append(node_id)
                current_load += demands[node_id]
            else:
                total_dist += calculate_route_distance(current_route)
                routes.append(current_route)
                current_route = [node_id]
                current_load = demands[node_id]

        if current_route:
            total_dist += calculate_route_distance(current_route)
            routes.append(current_route)

        return total_dist, routes

    def two_opt_permutation(perm):
        best = list(perm)
        best_dist, _ = decode_and_eval(best)

        improved = True
        while improved:
            improved = False
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    new = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    new_dist, _ = decode_and_eval(new)
                    if new_dist < best_dist:
                        best_dist = new_dist
                        best = new
                        improved = True
        return best

    def best_performance_es(mu, lambda_, generations):
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
    # SIDEBAR PARAMETERS
    # =========================
    st.sidebar.header("⚙️ Parameters")
    mu = st.sidebar.slider("μ Population", 10, 50, 20)
    lambda_ = st.sidebar.slider("λ Offspring", 50, 500 200)
    generations = st.sidebar.slider("Generations", 100, 1000, 200)

    # =========================
    # RUN BUTTON
    # =========================
    if st.button("🚀 Run Evolution Strategy"):
        with st.spinner("Optimizing routes..."):
            start = time.time()
            best_score, best_routes, history = best_performance_es(
                mu, lambda_, generations
            )
            exec_time = time.time() - start

        # =========================
        # METRICS
        # =========================
        st.success("Optimization Completed!")
        st.metric("Best Total Distance", f"{best_score:.4f}")
        st.metric("Vehicles Used", len(best_routes))
        st.metric("Execution Time (seconds)", f"{exec_time:.2f}")

        # =========================
        # ROUTE SUMMARY TABLE
        # =========================
        route_table = []
        customer_table = []

        for i, route in enumerate(best_routes, start=1):
            load = sum(demands[n] for n in route)
            dist = calculate_route_distance(route)

            route_table.append({
                "Vehicle": f"Vehicle {i}",
                "Route": "Depot → " + " → ".join(map(str, route)) + " → Depot",
                "Total Demand": load,
                "Route Distance": round(dist, 4),
                "Capacity Usage (%)": round((load / capacity) * 100, 2)
            })

            for n in route:
                customer_table.append({
                    "Customer ID": n,
                    "Vehicle": f"Vehicle {i}",
                    "Demand": demands[n]
                })

        route_df = pd.DataFrame(route_table)
        customer_df = pd.DataFrame(customer_table)

        st.subheader("📋 Vehicle Route Summary")
        st.dataframe(route_df, use_container_width=True, hide_index=True)

        st.subheader("👥 Customer Assignment Table")
        st.dataframe(customer_df, use_container_width=True, hide_index=True)

        # =========================
        # DOWNLOAD BUTTONS
        # =========================
        st.download_button(
            "⬇️ Download Route Summary (CSV)",
            route_df.to_csv(index=False).encode("utf-8"),
            "vrp_route_summary.csv",
            "text/csv"
        )

        st.download_button(
            "⬇️ Download Customer Assignment (CSV)",
            customer_df.to_csv(index=False).encode("utf-8"),
            "vrp_customer_assignment.csv",
            "text/csv"
        )

        # =========================
        # ROUTE VISUALIZATION
        # =========================
        st.subheader("📍 Best Route Visualization")
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(depot['x'], depot['y'], c='red', s=150, marker='s', label='Depot')
        ax.scatter(customers['x'], customers['y'], c='blue', s=50, label='Customers')

        colors = plt.cm.tab10(np.linspace(0, 1, len(best_routes)))

        for i, route in enumerate(best_routes):
            coords = [(depot['x'], depot['y'])]
            for n in route:
                node = df[df['node_id'] == n].iloc[0]
                coords.append((node['x'], node['y']))
            coords.append((depot['x'], depot['y']))

            xs, ys = zip(*coords)
            ax.plot(xs, ys, color=colors[i], linewidth=2, label=f'Vehicle {i+1}')

        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # =========================
        # CONVERGENCE CURVE
        # =========================
        st.subheader("📉 Convergence Curve")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(history)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Best Distance")
        ax2.grid(True)
        st.pyplot(fig2)

else:
    st.info("👆 Please upload a VRP CSV dataset to start.")
