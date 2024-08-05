import pandas as pd
import numpy as np

# Function to load centroids from the file
def load_centroids(file_path):
    with open(file_path, 'r') as file:
        num_centroids = int(file.readline().strip())
        centroids = np.array([list(map(float, line.strip().split('\t'))) for line in file])
    return centroids

# Function to load data points from the file
def load_data(file_path):
    with open(file_path, 'r') as file:
        m, n = map(int, file.readline().strip().split('\t'))
        data = np.array([list(map(float, line.strip().split('\t'))) for line in file])
    return data

centroids = load_centroids('P5_Centroids.txt')
data = load_data('P5_Data.txt')

def k_means(data, initial_centroids, max_iters=100):
    centroids = np.array(initial_centroids)
    for iteration in range(max_iters):
        # Compute distances and assign clusters
        expanded_centroids = np.expand_dims(centroids, axis=1)
        expanded_data = np.expand_dims(data, axis=0)
        distances = np.sqrt(np.sum((expanded_data - expanded_centroids) ** 2, axis=2))
        closest_centroids = np.argmin(distances, axis=0)

        # Initialize a list to store the new centroids
        new_centroids = []

        # Compute the new centroids and log additional information
        for k in range(len(centroids)):
            cluster_data = data[closest_centroids == k]
            if cluster_data.size:
                new_centroids.append(cluster_data.mean(axis=0))
                print(f"Cluster {k}, Size: {len(cluster_data)}, New Centroid: {new_centroids[-1]}")
            else:
                # Reuse the old centroid if no points are closest to it
                new_centroids.append(centroids[k])
                print(f"Cluster {k} is empty. Keeping old Centroid.")

        new_centroids = np.array(new_centroids)

        # Print centroids before updating for the next iteration
        print(f"Iteration {iteration + 1}: Centroids\n{new_centroids}")

        # Check for convergence
        if np.allclose(centroids, new_centroids, atol=1e-7):
            break

        # Update centroids for the next iteration
        centroids = new_centroids

    return closest_centroids, centroids



import matplotlib.pyplot as plt

def plot_k_means(data, centroids, labels):
    plt.figure(figsize=(8, 6))
    colors = ['red', 'green']
    
    for i in range(len(centroids)):
        points = data[labels == i]
        plt.scatter(points[:, 0], points[:, 1], s=50, c=colors[i], label=f'Cluster {i}')

    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='blue', label='Centroids', marker='X')
    plt.title('Clustered Data Points')
    plt.xlabel('x1 Axis')
    plt.ylabel('x2 Axis')
    plt.legend()
    plt.show()
    
labels, final_centroids = k_means(data, centroids)
plot_k_means(data, final_centroids, labels)

print("Final centroids' coordinates:")
print(final_centroids)

def compute_cost_function(data, centroids, labels):
    m = data.shape[0] # number of data points
    J = 0
    for i in range(len(centroids)):
        cluster_data = data[labels == i]
        J += np.sum((cluster_data - centroids[i]) ** 2)
    J /= m # divide by the number of data points to average
    return J

J = compute_cost_function(data, final_centroids, labels)
print(f"The overall error (cost function J) for the dataset is: {J}")