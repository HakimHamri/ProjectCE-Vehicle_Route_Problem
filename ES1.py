import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# =====================
# Streamlit Page Setup
# =====================
st.set_page_config(page_title="VRP with Evolution Strategy", layout="wide")
st.title("Vehicle Routing Problem - Evolution Strategy")

# =====================
# Upload CSV Data
# =====================
st.subheader("Upload CSV File")
uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    coords = df[['x', 'y']].values
    n_points = len(coords)

    # =====================
    # Distance Functions
    # =====================
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def total_distance(route, coords):
        dist = 0
        for i in range(len(route) - 1):
            dist += euclidean_distance(coords[route[i]], coords[route[i+1]])
        dist += euclidean_distance(coords[route[-1]], coords[route[0]])  # Return to depot
        return dist

    # =====================
    # Mutations
    # =====================
    def swap_mutation(route):
        a, b = random.sample(range(len(route)), 2)
        route[a], route[b] = route[b], route[a]
        return route

    def reverse_mutation(route):
        a, b = sorted(random.sample(range(len(route)), 2))
        route[a:b+1] = reversed(route[a:b+1])
        return route

    def insert_mutation(route):
        a, b = random.sample(range(len(route)), 2)
        city = route.pop(a)
        route.insert(b, city)
        return route

    def mutate(route):
        mutation_type = random.choice([swap_mutation, reverse_mutation, insert_mutation])
        return mutation_type(route.copy())

    # =====================
    # Evolution Strategy
    # =====================
    def evolution_strategy(coords, mu=30, lam=200, generations=500):
        population = [random.sample(range(n_points), n_points) for _ in range(mu)]
        best_solution = None
        best_distance = float('inf')
        
        for gen in range(generations):
            offspring = [mutate(random.choice(population)) for _ in range(lam)]
            combined = population + offspring
            combined.sort(key=lambda r: total_distance(r, coords))
            population = combined[:mu]
            
            current_best = population[0]
            current_distance = total_distance(current_best, coords)
            if current_distance < best_distance:
                best_distance = current_distance
                best_solution = current_best
            
            if gen % max(1, generations // 10) == 0:
                st.write(f"Generation {gen} | Best Distance = {best_distance:.2f}")
        
        return best_solution, best_distance

    # =====================
    # Streamlit Parameters
    # =====================
    st.subheader("Evolution Strategy Parameters")
    mu = st.number_input("Population size (mu)", min_value=5, max_value=100, value=30)
    lam = st.number_input("Offspring size (lambda)", min_value=10, max_value=500, value=200)
    generations = st.number_input("Generations", min_value=10, max_value=1000, value=200)

    if st.button("Run Evolution Strategy"):
        with st.spinner("Running Evolution Strategy..."):
            best_route, best_dist = evolution_strategy(coords, mu=int(mu), lam=int(lam), generations=int(generations))
        
        st.success(f"Best Distance: {best_dist:.2f}")
        
        # =====================
        # Plot Best Route
        # =====================
        st.subheader("Best Route Plot")
        route_coords = coords[best_route + [best_route[0]]]
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(route_coords[:,0], route_coords[:,1], 'o-', color='blue')
        ax.scatter(coords[0,0], coords[0,1], color='red', s=100, label='Depot')
        ax.set_title(f'Best VRP Route | Distance = {best_dist:.2f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        st.pyplot(fig)
