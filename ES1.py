import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# =====================
# Page Configuration
# =====================
st.set_page_config(page_title="VRP Evolution Strategy", layout="wide")
st.title("🚚 Vehicle Routing Problem (VRP) Solver")
st.markdown("This app uses an **Evolutionary Strategy** (ES) to find the shortest route for a set of coordinates.")

# =====================
# Functions (Logic remains same)
# =====================
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def total_distance(route, coords):
    dist = 0
    for i in range(len(route) - 1):
        dist += euclidean_distance(coords[route[i]], coords[route[i+1]])
    dist += euclidean_distance(coords[route[-1]], coords[route[0]])
    return dist

def swap_mutation(route):
    a, b = random.sample(range(len(route)), 2)
    route[a], route[b] = route[b], route[a]
    return route

def reverse_mutation(route):
    a, b = sorted(random.sample(range(len(route)), 2))
    route[a:b+1] = list(reversed(route[a:b+1]))
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
# Sidebar Parameters
# =====================
st.sidebar.header("Evolution Parameters")
uploaded_file = st.sidebar.file_uploader("Upload VRP CSV (must have x, y columns)", type="csv")

mu = st.sidebar.slider("Population Size (mu)", 10, 100, 30)
lam = st.sidebar.slider("Offspring Size (lambda)", 50, 500, 200)
gens = st.sidebar.number_input("Generations", min_value=10, max_value=2000, value=500)

# =====================
# Main Execution
# =====================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    coords = df[['x', 'y']].values
    n_points = len(coords)

    if st.button("🚀 Start Evolution"):
        # Progress Bar and Status
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()
        
        # Initialization
        population = [random.sample(range(n_points), n_points) for _ in range(mu)]
        best_solution = None
        best_distance = float('inf')
        history = []

        # Evolution Loop
        for gen in range(gens):
            offspring = [mutate(random.choice(population)) for _ in range(lam)]
            combined = population + offspring
            combined.sort(key=lambda r: total_distance(r, coords))
            population = combined[:mu]
            
            current_best = population[0]
            current_distance = total_distance(current_best, coords)
            
            if current_distance < best_distance:
                best_distance = current_distance
                best_solution = current_best
            
            history.append(best_distance)
            
            # Update UI every 10 generations
            if gen % 10 == 0 or gen == gens - 1:
                progress_bar.progress((gen + 1) / gens)
                status_text.text(f"Generation {gen} | Best Distance: {best_distance:.2f}")

        # =====================
        # Results Display
        # =====================
        st.success(f"Evolution Complete! Final Distance: {best_distance:.2f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Optimized Route")
            fig, ax = plt.subplots()
            route_idx = best_solution + [best_solution[0]]
            route_coords = coords[route_idx]
            ax.plot(route_coords[:,0], route_coords[:,1], 'o-', color='blue', markersize=4)
            ax.scatter(coords[0,0], coords[0,1], color='red', s=100, label='Depot', zorder=5)
            ax.set_title(f"Distance: {best_distance:.2f}")
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("Convergence Curve")
            st.line_chart(history)

else:
    st.info("Please upload a CSV file in the sidebar to begin.")
    st.write("The CSV should look like this:")
    st.write(pd.DataFrame({'x': [10, 20, 30], 'y': [15, 25, 35]}))
