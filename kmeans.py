##############################################################################
#         TODO: Write the code for reading data from files                    #
##############################################################################
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

##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################

##############################################################################
#                TODO: Implement the K-means clustering                      #
##############################################################################
def k_means(data, initial_centroids, max_iters=100):
    centroids = np.array(initial_centroids)
    for _ in range(max_iters):
        expanded_centroids = np.expand_dims(centroids, axis=1)
        expanded_data = np.expand_dims(data, axis=0)
        distances = np.sqrt(np.sum((expanded_data - expanded_centroids) ** 2, axis=2))
        closest_centroids = np.argmin(distances, axis=0)
        new_centroids = []

        for k in range(len(centroids)):
            cluster_data = data[closest_centroids == k]
            if cluster_data.size:
                new_centroids.append(cluster_data.mean(axis=0))
            else:
                new_centroids.append(centroids[k])
        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids, atol=1e-5):
            break
        centroids = new_centroids

    return closest_centroids, centroids
##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################

##############################################################################
#                TODO: Plot the clustering result                           #
##############################################################################
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
##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################

##############################################################################
#       TODO: print out the final centroids' coordinates                     #
##############################################################################
print("Final centroids' coordinates:")
print(final_centroids)
##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################

##############################################################################
#                           TODO: Model Evaluation                           #
##############################################################################
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
##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################