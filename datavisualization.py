##############################################################################
#         TODO: Write the code for reading data from file                    #
##############################################################################
file_name = input("Enter the name of your data file: ")
print("You can do a plot of any two features of the Iris Data set. The feature codes are:")
print("0 = sepal length\n1 = sepal width\n2 = petal length\n3 = petal width\n")

feature_x = int(input("Enter feature code for the horizontal axis: "))
feature_y = int(input("Enter feature code for the vertical axis: "))

data = {
    "setosa": [],
    "versicolor": [],
    "virginica": []
}

# Separate the data into lists
with open(file_name, 'r') as file:
    for line in file:
        # Separate each line into words
        items = line.strip().split()
        if len(items) == 5:
            features = [float(i) for i in items[:4]]
            label = items[4]
            data[label].append(features)

##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################

##############################################################################
#   TODO: Write the function for plotting using two features of the data     #
##############################################################################
import matplotlib.pyplot as plot
def plot_features(file_name, feature_x, feature_y):
    # Objects for plots
    colors = {
        "setosa": 'r',
        "versicolor": 'g',
        "virginica": 'b'
    }
    markers = {
        "setosa": 'o',
        "versicolor": 's',
        "virginica": 'd'
    }
    data = {
        "setosa": [],
        "versicolor": [],
        "virginica": []
    }

    # Separate the data into lists
    with open(file_name, 'r') as file:
        for line in file:
            # Separate each line into words
            items = line.strip().split()
            if len(items) == 5:
                features = [float(i) for i in items[:4]]
                label = items[4]
                data[label].append(features)

    for label, features in data.items():
        if features:
            x = [feature[feature_x] for feature in features]
            y = [feature[feature_y] for feature in features]
            plot.scatter(x, y, color=colors[label], marker=markers[label], label=label)
    
    # Plot labels
    plot.xlabel(['sepal length', 'sepal width', 'petal length', 'petal width'][feature_x])
    plot.ylabel(['sepal length', 'sepal width', 'petal length', 'petal width'][feature_y])
    plot.title('Iris Data Set')
    plot.legend()
    plot.show()
##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################

##############################################################################
#          TODO: Use the loop function for plotting figures                 #
##############################################################################
import matplotlib.pyplot as plot
def plot_points(file_name, feature_x, feature_y):
    # Objects for plots
    colors = {
        "setosa": 'r',
        "versicolor": 'g',
        "virginica": 'b'
    }
    markers = {
        "setosa": 'o',
        "versicolor": 's',
        "virginica": 'd'
    }
    data = {
        "setosa": [],
        "versicolor": [],
        "virginica": []
    }

    # Separate the data into lists
    with open(file_name, 'r') as file:
        for line in file:
            # Separate each line into words
            items = line.strip().split()
            if len(items) == 5:
                features = [float(i) for i in items[:4]]
                label = items[4]
                data[label].append(features)

    for label, features in data.items():
        if features:
            x = [feature[feature_x] for feature in features]
            y = [feature[feature_y] for feature in features]
            plot.scatter(x, y, color=colors[label], marker=markers[label], label=label)
    
    # Plot labels
    plot.xlabel(['sepal length', 'sepal width', 'petal length', 'petal width'][feature_x])
    plot.ylabel(['sepal length', 'sepal width', 'petal length', 'petal width'][feature_y])
    plot.title('Iris Data Set')
    plot.legend()
    plot.show()


file_name = input("Enter the name of your data file: ")
while True:
    print("You can do a plot of any two features of the Iris Data set. The feature codes are:")
    print("0 = sepal length\n1 = sepal width\n2 = petal length\n3 = petal width\n")
    
    feature_x = int(input("Enter feature code for the horizontal axis: "))
    feature_y = int(input("Enter feature code for the vertical axis: "))
    
    plot_points(file_name, feature_x, feature_y)
    
    plot_again = input("\nWould you like to do another plot? (y/n) ").lower()
    if plot_again == 'n':
        break
##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################

