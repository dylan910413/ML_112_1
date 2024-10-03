# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iteration=100):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.intercept = None

    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y):
        intercept_column = np.ones((X.shape[0], 1))
        X_intercept = np.hstack((intercept_column, X))
        sample_num, feature_num = X_intercept.shape
        self.weights = np.zeros(feature_num)
        for t in range(self.iteration):
            gradient_accum = np.zeros(feature_num)
            for i in range(sample_num):
                x_i = X_intercept[i]
                y_i = y[i]
                log_odds = np.dot(x_i, self.weights)
                probability = self.sigmoid(log_odds)
                loss = probability - y_i
                gradient = x_i * loss
                gradient_accum += gradient

            self.weights -= self.learning_rate * gradient_accum / sample_num
            self.intercept = self.weights[0]
            if t % 200 == 0:
                self.learning_rate /= 2
    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        X_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
        probabilities = self.sigmoid(np.dot(X_intercept, self.weights))
        return np.round(probabilities)

    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None
        self.intercept = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        X0 = X[y == 0]
        X1 = X[y == 1]
        self.m0 = np.mean(X0, axis=0)
        self.m1 = np.mean(X1, axis=0)
        self.sw = np.dot((X0 - self.m0).T, X0 - self.m0) + np.dot((X1 - self.m1).T, X1 - self.m1)
        self.sb = np.outer((self.m1 - self.m0), (self.m1 - self.m0))
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(self.sw).dot(self.sb))
        self.w = eigenvectors[:, np.argmax(eigenvalues)]
        self.slope = self.w[1] / self.w[0]
        self.intercept = np.dot(self.m0, self.w) / np.dot(self.w, self.w) 
    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def project(self, X):
        # Project data onto the optimal direction
        return np.dot(X, self.w)

    def predict(self, X):
        projected_X = self.project(X)
        distance_to_m0 = np.abs(projected_X - np.dot(self.m0, self.w))
        distance_to_m1 = np.abs(projected_X - np.dot(self.m1, self.w))
        predictions = np.where(distance_to_m0 < distance_to_m1, 0, 1)
        return predictions

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X):
        plt.figure(figsize=(10, 5))
        plt.axline((self.intercept + 220, 0), slope=self.slope, color='green', linestyle='-', linewidth=2, label='Projection Line (Training Set)')
        predictions = self.predict(X)
        plt.scatter(X[predictions == 0][:, 0], X[predictions == 0][:, 1], label='Predicted Class 0 (Testing Set)', marker='o', color='blue')
        plt.scatter(X[predictions == 1][:, 0], X[predictions == 1][:, 1], label='Predicted Class 1 (Testing Set)', marker='o', color='red')
        for test_point in X:
            projected_point = self.project(test_point.reshape(1, -1))
        plt.plot([test_point[0], projected_point[0]], [test_point[1], projected_point[0] * self.slope + self.intercept + 220], color='gray', linestyle='dashed', linewidth=1)
        plt.title(f"Projection Line (Training Set)\nSlope: {self.slope:.2f}, Intercept: {self.intercept:.2f}")
        plt.xlabel('Age')
        plt.ylabel('thalach')
        plt.legend()
        plt.axis('equal')
        plt.show()




     
# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    LR = LogisticRegression(learning_rate=0.026, iteration=1600)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"

