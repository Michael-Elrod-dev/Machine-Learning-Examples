##############################################################################
#         TODO: Write the code for reading data from file                    #
##############################################################################
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np

def read_data(read_file):
    with open(read_file, 'r') as file:
        first_line = file.readline().strip().split()
        entries, features = int(first_line[0]), int(first_line[1])
         
        x, y = [], []
        for _ in range(entries):
            line = file.readline().strip().split()
            x.append(int(line[0])+1900)
            y.append(float(line[1]))
       
        return x, y, entries, features
    
x_values, y_values, entries, features = read_data("W100MTimes.txt")
##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################

##############################################################################
#                 TODO: Define the regression models                         #
##############################################################################
def calculate_weights(X, Y):
    X_transpose = np.transpose(X)
    X_transpose_X = np.dot(X_transpose, X)
    X_transpose_X_inverse = np.linalg.inv(X_transpose_X)
    X_transpose_y = np.dot(X_transpose, Y)
    w = np.dot(X_transpose_X_inverse, X_transpose_y)
    return w

def linear_model(x_values, y_values):
    X = np.vstack([np.ones(len(x_values)), x_values]).T
    Y = np.array(y_values).reshape(-1, 1)
    w = calculate_weights(X, Y)
    linear = [w[0] + w[1]*x for x in x_values]
    return linear, w

def quadratic_model(x_values, y_values):
    X = np.vstack([np.ones(len(x_values)), x_values, np.power(x_values, 2)]).T
    Y = np.array(y_values).reshape(-1, 1)
    w = calculate_weights(X, Y)
    quadratic = [w[0] + w[1]*x + w[2]*x**2 for x in x_values]
    return quadratic, w

def cubic_model(x_values, y_values):
    X = np.vstack([np.ones(len(x_values)), x_values, np.power(x_values, 2), np.power(x_values, 3)]).T
    Y = np.array(y_values).reshape(-1, 1)
    w = calculate_weights(X, Y)
    cubic = [w[0] + w[1]*x + w[2]*x**2 + w[3]*x**3 for x in x_values]
    return cubic, w

def fourth_degree_model(x_values, y_values):
    X = np.vstack([np.ones(len(x_values)), x_values, np.power(x_values, 2), np.power(x_values, 3), np.power(x_values, 4)]).T
    Y = np.array(y_values).reshape(-1, 1)
    w = calculate_weights(X, Y)
    fourth = [w[0] + w[1]*x + w[2]*x**2 + w[3]*x**3 + w[4]*x**4 for x in x_values]
    return fourth, w
##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################

##############################################################################
#                     TODO: 5-fold cross-validation                         #
##############################################################################
def calculate_error(X, Y, w):
    m = len(Y)
    difference = np.dot(X, w) - Y
    J = (1/m) * np.dot(difference.T, difference)
    return J[0][0]

def k_fold_cross_validation(x_values, y_values, k=5):
    data_size = len(x_values)
    fold_size = data_size // k
    results = []
    training_errors = []
    testing_errors = []

    for i in range(k):
        test_indices = list(range(i*fold_size, (i+1)*fold_size))
        train_indices = list(range(0, i*fold_size)) + list(range((i+1)*fold_size, data_size))

        x_train = [x_values[j] for j in train_indices]
        y_train = [y_values[j] for j in train_indices]
        x_test = [x_values[j] for j in test_indices]
        y_test = [y_values[j] for j in test_indices]

        # Train models using training data
        linear, linear_weights = linear_model(x_train, y_train)
        quadratic, quadratic_weights = quadratic_model(x_train, y_train)
        cubic, cubic_weights = cubic_model(x_train, y_train)
        fourth, fourth_weights = fourth_degree_model(x_train, y_train)

        # Define the X and Y matrices for each model
        Y_train = np.array(y_train).reshape(-1, 1)
        Y_test = np.array(y_test).reshape(-1, 1)

        X_train_linear = np.vstack([np.ones(len(x_train)), x_train]).T
        X_train_quadratic = np.vstack([np.ones(len(x_train)), x_train, np.power(x_train, 2)]).T
        X_train_cubic = np.vstack([np.ones(len(x_train)), x_train, np.power(x_train, 2), np.power(x_train, 3)]).T
        X_train_fourth = np.vstack([np.ones(len(x_train)), x_train, np.power(x_train, 2), np.power(x_train, 3), np.power(x_train, 4)]).T

        X_test_linear = np.vstack([np.ones(len(x_test)), x_test]).T
        X_test_quadratic = np.vstack([np.ones(len(x_test)), x_test, np.power(x_test, 2)]).T
        X_test_cubic = np.vstack([np.ones(len(x_test)), x_test, np.power(x_test, 2), np.power(x_test, 3)]).T
        X_test_fourth = np.vstack([np.ones(len(x_test)), x_test, np.power(x_test, 2), np.power(x_test, 3), np.power(x_test, 4)]).T

        # Calculate training errors
        linear_train_err = calculate_error(X_train_linear, Y_train, linear_weights)
        quadratic_train_err = calculate_error(X_train_quadratic, Y_train, quadratic_weights)
        cubic_train_err = calculate_error(X_train_cubic, Y_train, cubic_weights)
        fourth_train_err = calculate_error(X_train_fourth, Y_train, fourth_weights)

        # Calculate testing errors
        linear_test_err = calculate_error(X_test_linear, Y_test, linear_weights)
        quadratic_test_err = calculate_error(X_test_quadratic, Y_test, quadratic_weights)
        cubic_test_err = calculate_error(X_test_cubic, Y_test, cubic_weights)
        fourth_test_err = calculate_error(X_test_fourth, Y_test, fourth_weights)

        training_errors.append([linear_train_err, quadratic_train_err, cubic_train_err, fourth_train_err])
        testing_errors.append([linear_test_err, quadratic_test_err, cubic_test_err, fourth_test_err])

        train_folds = "".join([str(j+1) for j in range(k) if j != i])
        fold_representation = train_folds + "\n" + str(i+1)
        results.append([fold_representation, linear_test_err, quadratic_test_err, cubic_test_err, fourth_test_err])

    # Calculate mean errors
    mean_training_errors = np.mean(training_errors, axis=0)
    mean_testing_errors = np.mean(testing_errors, axis=0)

    results.append(["Mean for Training", mean_training_errors[0], mean_training_errors[1], mean_training_errors[2], mean_training_errors[3]])
    results.append(["Mean for Testing", mean_testing_errors[0], mean_testing_errors[1], mean_testing_errors[2], mean_testing_errors[3]])

    # Create table
    headers = ["", "Linear", "Quadratic", "Cubic", "Fourth"]
    print(tabulate(results, headers, tablefmt="grid"))
    
    # Create plot
    plot_errors(training_errors, testing_errors)
##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################

##############################################################################
#                     TODO: Print out the table chart                       #
##############################################################################
# This method also prints the table!
def k_fold_cross_validation(x_values, y_values, k=5):
    data_size = len(x_values)
    fold_size = data_size // k
    results = []
    training_errors = []
    testing_errors = []

    for i in range(k):
        test_indices = list(range(i*fold_size, (i+1)*fold_size))
        train_indices = list(range(0, i*fold_size)) + list(range((i+1)*fold_size, data_size))

        x_train = [x_values[j] for j in train_indices]
        y_train = [y_values[j] for j in train_indices]
        x_test = [x_values[j] for j in test_indices]
        y_test = [y_values[j] for j in test_indices]

        # Train models using training data
        linear, linear_weights = linear_model(x_train, y_train)
        quadratic, quadratic_weights = quadratic_model(x_train, y_train)
        cubic, cubic_weights = cubic_model(x_train, y_train)
        fourth, fourth_weights = fourth_degree_model(x_train, y_train)

        # Define the X and Y matrices for each model
        Y_train = np.array(y_train).reshape(-1, 1)
        Y_test = np.array(y_test).reshape(-1, 1)

        X_train_linear = np.vstack([np.ones(len(x_train)), x_train]).T
        X_train_quadratic = np.vstack([np.ones(len(x_train)), x_train, np.power(x_train, 2)]).T
        X_train_cubic = np.vstack([np.ones(len(x_train)), x_train, np.power(x_train, 2), np.power(x_train, 3)]).T
        X_train_fourth = np.vstack([np.ones(len(x_train)), x_train, np.power(x_train, 2), np.power(x_train, 3), np.power(x_train, 4)]).T

        X_test_linear = np.vstack([np.ones(len(x_test)), x_test]).T
        X_test_quadratic = np.vstack([np.ones(len(x_test)), x_test, np.power(x_test, 2)]).T
        X_test_cubic = np.vstack([np.ones(len(x_test)), x_test, np.power(x_test, 2), np.power(x_test, 3)]).T
        X_test_fourth = np.vstack([np.ones(len(x_test)), x_test, np.power(x_test, 2), np.power(x_test, 3), np.power(x_test, 4)]).T

        # Calculate training errors
        linear_train_err = calculate_error(X_train_linear, Y_train, linear_weights)
        quadratic_train_err = calculate_error(X_train_quadratic, Y_train, quadratic_weights)
        cubic_train_err = calculate_error(X_train_cubic, Y_train, cubic_weights)
        fourth_train_err = calculate_error(X_train_fourth, Y_train, fourth_weights)

        # Calculate testing errors
        linear_test_err = calculate_error(X_test_linear, Y_test, linear_weights)
        quadratic_test_err = calculate_error(X_test_quadratic, Y_test, quadratic_weights)
        cubic_test_err = calculate_error(X_test_cubic, Y_test, cubic_weights)
        fourth_test_err = calculate_error(X_test_fourth, Y_test, fourth_weights)

        training_errors.append([linear_train_err, quadratic_train_err, cubic_train_err, fourth_train_err])
        testing_errors.append([linear_test_err, quadratic_test_err, cubic_test_err, fourth_test_err])

        train_folds = "".join([str(j+1) for j in range(k) if j != i])
        fold_representation = train_folds + "\n" + str(i+1)
        results.append([fold_representation, linear_test_err, quadratic_test_err, cubic_test_err, fourth_test_err])

    # Calculate mean errors
    mean_training_errors = np.mean(training_errors, axis=0)
    mean_testing_errors = np.mean(testing_errors, axis=0)

    results.append(["Mean for Training", mean_training_errors[0], mean_training_errors[1], mean_training_errors[2], mean_training_errors[3]])
    results.append(["Mean for Testing", mean_testing_errors[0], mean_testing_errors[1], mean_testing_errors[2], mean_testing_errors[3]])

    # Create table
    headers = ["", "Linear", "Quadratic", "Cubic", "Fourth"]
    print(tabulate(results, headers, tablefmt="grid"))
    
    # Create plot
    plot_errors(training_errors, testing_errors)
##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################

##############################################################################
#                     TODO: Plot the J curve                                 #
##############################################################################
def plot_errors(training_errors, testing_errors):
    # Calculate mean errors
    mean_training_errors = np.mean(training_errors, axis=0)
    mean_testing_errors = np.mean(testing_errors, axis=0)
    
    # Plotting the training and testing J with respect to the polynomial degree
    degrees = [1, 2, 3, 4]
    plt.figure()
    plt.plot(degrees, mean_training_errors, marker='o', label='Training Error')
    plt.plot(degrees, mean_testing_errors, marker='o', label='Testing Error')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Error (J)')
    plt.title('Training and Testing Error vs Polynomial Degree')
    plt.legend()
    plt.grid(True)
    plt.show()

def k_fold_cross_validation(x_values, y_values, k=5):
    data_size = len(x_values)
    fold_size = data_size // k
    results = []
    training_errors = []
    testing_errors = []

    for i in range(k):
        test_indices = list(range(i*fold_size, (i+1)*fold_size))
        train_indices = list(range(0, i*fold_size)) + list(range((i+1)*fold_size, data_size))

        x_train = [x_values[j] for j in train_indices]
        y_train = [y_values[j] for j in train_indices]
        x_test = [x_values[j] for j in test_indices]
        y_test = [y_values[j] for j in test_indices]

        # Train models using training data
        linear, linear_weights = linear_model(x_train, y_train)
        quadratic, quadratic_weights = quadratic_model(x_train, y_train)
        cubic, cubic_weights = cubic_model(x_train, y_train)
        fourth, fourth_weights = fourth_degree_model(x_train, y_train)

        # Define the X and Y matrices for each model
        Y_train = np.array(y_train).reshape(-1, 1)
        Y_test = np.array(y_test).reshape(-1, 1)

        X_train_linear = np.vstack([np.ones(len(x_train)), x_train]).T
        X_train_quadratic = np.vstack([np.ones(len(x_train)), x_train, np.power(x_train, 2)]).T
        X_train_cubic = np.vstack([np.ones(len(x_train)), x_train, np.power(x_train, 2), np.power(x_train, 3)]).T
        X_train_fourth = np.vstack([np.ones(len(x_train)), x_train, np.power(x_train, 2), np.power(x_train, 3), np.power(x_train, 4)]).T

        X_test_linear = np.vstack([np.ones(len(x_test)), x_test]).T
        X_test_quadratic = np.vstack([np.ones(len(x_test)), x_test, np.power(x_test, 2)]).T
        X_test_cubic = np.vstack([np.ones(len(x_test)), x_test, np.power(x_test, 2), np.power(x_test, 3)]).T
        X_test_fourth = np.vstack([np.ones(len(x_test)), x_test, np.power(x_test, 2), np.power(x_test, 3), np.power(x_test, 4)]).T

        # Calculate training errors
        linear_train_err = calculate_error(X_train_linear, Y_train, linear_weights)
        quadratic_train_err = calculate_error(X_train_quadratic, Y_train, quadratic_weights)
        cubic_train_err = calculate_error(X_train_cubic, Y_train, cubic_weights)
        fourth_train_err = calculate_error(X_train_fourth, Y_train, fourth_weights)

        # Calculate testing errors
        linear_test_err = calculate_error(X_test_linear, Y_test, linear_weights)
        quadratic_test_err = calculate_error(X_test_quadratic, Y_test, quadratic_weights)
        cubic_test_err = calculate_error(X_test_cubic, Y_test, cubic_weights)
        fourth_test_err = calculate_error(X_test_fourth, Y_test, fourth_weights)

        training_errors.append([linear_train_err, quadratic_train_err, cubic_train_err, fourth_train_err])
        testing_errors.append([linear_test_err, quadratic_test_err, cubic_test_err, fourth_test_err])

        train_folds = "".join([str(j+1) for j in range(k) if j != i])
        fold_representation = train_folds + "\n" + str(i+1)
        results.append([fold_representation, linear_test_err, quadratic_test_err, cubic_test_err, fourth_test_err])

    # Calculate mean errors
    mean_training_errors = np.mean(training_errors, axis=0)
    mean_testing_errors = np.mean(testing_errors, axis=0)

    results.append(["Mean for Training", mean_training_errors[0], mean_training_errors[1], mean_training_errors[2], mean_training_errors[3]])
    results.append(["Mean for Testing", mean_testing_errors[0], mean_testing_errors[1], mean_testing_errors[2], mean_testing_errors[3]])

    # Create table
    headers = ["", "Linear", "Quadratic", "Cubic", "Fourth"]
    print(tabulate(results, headers, tablefmt="grid"))
    
    # Create plot
    plot_errors(training_errors, testing_errors)
    
k_fold_cross_validation(x_values, y_values)

##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################

##############################################################################
#              TODO: Calculate the final regression model                    #
##############################################################################
def compute_final_weights(x_values, y_values, model_type):
    # Convert x_values and y_values to appropriate numpy arrays
    X = np.array(x_values).reshape(-1, 1)
    Y = np.array(y_values).reshape(-1, 1)
    
    # Depending on the model_type, construct the X matrix with appropriate polynomial features
    if model_type == 'linear':
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    elif model_type == 'quadratic':
        X = np.hstack([np.ones((X.shape[0], 1)), X, X**2])
    elif model_type == 'cubic':
        X = np.hstack([np.ones((X.shape[0], 1)), X, X**2, X**3])
    elif model_type == 'fourth':
        X = np.hstack([np.ones((X.shape[0], 1)), X, X**2, X**3, X**4])
    
    # Use the calculate_weights function to compute the final weights
    final_weights = calculate_weights(X, Y)
    
    return final_weights

best_model_type = "linear"

# Compute the final weights using the entire dataset
final_weights = compute_final_weights(x_values, y_values, best_model_type)

print(f"Final weights of the {best_model_type} model: w0={final_weights[0][0]}, w1={final_weights[1][0]}")
##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################

##############################################################################
#         TODO: Predict the race time using the best model                   #
##############################################################################
def predict_time(year, weights, model_type):
    # Using the model equation to predict the time
    if model_type == 'linear':
        return round((weights[0].item() + weights[1].item() * year), 2)
    elif model_type == 'quadratic':
        return round((weights[0].item() + weights[1].item() * year + weights[2].item() * year**2), 2)
    elif model_type == 'cubic':
        return round((weights[0].item() + weights[1].item() * year + weights[2].item() * year**2 + weights[3].item() * year**3), 2)
    elif model_type == 'fourth':
        return round((weights[0].item() + weights[1].item() * year + weights[2].item() * year**2 + weights[3].item() * year**3 + weights[4].item() * year**4), 2)

best_model_type = "linear"
# Predict the winning times for the specified years
years_to_predict = [1980, 2000, 2024]
for year in years_to_predict:
    predicted_time = predict_time(year, final_weights, best_model_type)
    print(f"The predicted winning time in the year {year} is {predicted_time} seconds")
##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################