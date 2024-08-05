import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def read_data(file_path):
    with open(file_path, 'r') as f:
        m, n = map(int, f.readline().split('\t'))
        df = pd.read_csv(f, header=None, delimiter='\t', engine='python')
        features = df.iloc[:, :n].to_numpy()
        y_true = df.iloc[:, n].to_numpy()
        return m, n, features, y_true

def generate_polynomial_features(X, max_degree=2):
    x1 = X[:, 0]
    x2 = X[:, 1]
    m = len(x1)
    new_features = [np.ones(m)]
    for i in range(1, max_degree + 1):
        for j in range(i + 1):
            new_features.append((x1 ** (i - j)) * (x2 ** j))
    return np.column_stack(new_features)

def feature_scaling(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-10
    X_norm = (X - mean) / std
    return X_norm, mean, std

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def gradient_func(X_train, y_train, w):
    y_pred = sigmoid(np.dot(X_train, w))
    gradient = np.dot(X_train.T, (y_pred - y_train)) / len(y_train)
    return gradient

def Vanilla_GD(X_train, y_train, epoch_num, lr, w, J):
    for epoch in range(epoch_num):
        y_pred = sigmoid(np.dot(X_train, w))
        loss = cross_entropy_loss(y_train, y_pred)
        J.append(loss)
        gradient = gradient_func(X_train, y_train, w)
        w -= lr * gradient
    return w, J

def logistic_regression(X, y, learning_rate, num_iterations):
    w = np.zeros((X.shape[1], 1))
    J = []
    w, J = Vanilla_GD(X, y, num_iterations, learning_rate, w, J)
    return w, J


def evaluate_model(y_true, y_pred):
    y_pred_class = (y_pred >= 0.5).astype(int)
    accuracy = np.mean(y_pred_class == y_true)
    true_positives = np.sum((y_pred_class == 1) & (y_true == 1))
    predicted_positives = np.sum(y_pred_class == 1)
    precision = true_positives / predicted_positives if predicted_positives != 0 else 0
    actual_positives = np.sum(y_true == 1)
    recall = true_positives / actual_positives
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    print(f"Accuracy: {accuracy:.0%}")
    print(f"Precision: {precision:.0%}")
    print(f"Recall: {recall:.0%}")
    print(f"F1 Score: {f1:.0%}")

# Function to plot decision boundaries
def plot_decision_boundary(X, y, classifier, title):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Apply the same transformations as you did for training data
    Z = generate_polynomial_features(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z - mean_train) / std_train
    
    if title == 'Logistic Regression':
        Z = sigmoid(np.dot(Z, w))
        Z = (Z >= 0.5).astype(int)
    else:
        Z = classifier.predict(Z)
    
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(('red', 'blue')))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50)
    plt.title(title)
    plt.show()

# Main Code
epoch_num = 45000
lr = .01

# Read and preprocess training data
train_m, train_n, train_features_original, train_y_true = read_data('P3train.txt')  # Save original features here
train_features = generate_polynomial_features(train_features_original)  # Generate polynomial features
train_features, mean_train, std_train = feature_scaling(train_features)  # Feature scaling
train_y_true = train_y_true.reshape(-1, 1)

# Read and preprocess testing data
test_m, test_n, test_features, test_y_true = read_data('P3test.txt')
test_features = generate_polynomial_features(test_features)
test_features = (test_features - mean_train) / std_train
test_y_true = test_y_true.reshape(-1, 1)

w, J = logistic_regression(train_features, train_y_true, lr, epoch_num)
test_y_pred = sigmoid(np.dot(test_features, w))
evaluate_model(test_y_true, test_y_pred)

print("Weights:")
print(w)
plt.plot(J)
plt.xlabel('Iteration Number')
plt.ylabel('Loss Value')
plt.title('J-curve/Loss curve over Training Iterations')
plt.show()

def evaluate_model_svm(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Accuracy: {round(accuracy * 100)}%")
    print(f"Precision: {round(precision * 100)}%")
    print(f"Recall: {round(recall * 100)}%")
    print(f"F1 Score: {round(f1 * 100)}%\n")

# Train SVM models using different kernels
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
best_model = None
best_accuracy = 0

for kernel in kernels:
    clf = svm.SVC(kernel=kernel)
    if (kernel == 'rbf'):
        clf = svm.SVC(kernel=kernel, gamma=.1)
    clf.fit(train_features, train_y_true.ravel())
    test_y_pred = clf.predict(test_features)
    
    # Evaluate model
    print(f"Kernel: {kernel}")
    evaluate_model_svm(test_y_true, test_y_pred)
    
    accuracy = accuracy_score(test_y_true, test_y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = clf

# Print weights of the best model
print("Best Model Information:")
print(f"Best Model: {best_model.kernel}")
if best_model.kernel == 'linear':
    print("Weights:", best_model.coef_)
elif best_model.kernel == 'rbf':
    if best_model.gamma == 'scale':
        # Calculate the gamma value
        n_features = train_features.shape[1]
        var_data = np.var(train_features)
        calculated_gamma = 1 / (n_features * var_data)
        print(f"Calculated Gamma (scale): {calculated_gamma}")
    else:
        print(f"Gamma: {best_model.gamma}")
    print("Note: RBF kernel doesn't have weights")
else:
    print("This kernel doesn't provide weights")


    
# Plot decision boundary for Logistic Regression
plot_decision_boundary(train_features_original, train_y_true.ravel(), None, 'Logistic Regression')

# Plot decision boundary for the best SVM model
plot_decision_boundary(train_features_original, train_y_true.ravel(), best_model, f'SVM ({best_model.kernel})')
