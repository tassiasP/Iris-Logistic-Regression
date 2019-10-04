# Imports
import numpy as np
from sklearn.metrics import accuracy_score
import scipy.optimize as op
from math import e

# We will use the Iris Dataset for the LogisticRegression algorithm
def load_iris_data(dataset_path):
    import pandas as pd
    
    iris_data = pd.read_table(dataset_path+"/iris.data", delimiter=',', header=None)
    
    return iris_data
    
iris_data = load_iris_data("../Dataset/iris")

"""
Function edit_data() splits the data in Data-Values and Class-Values and also removes
random class from dataset in order to avoid multinomial LR and use the sigmoid function instead
"""

def edit_data(iris_data, multinomial=False):
    import random
    class_num = random.randint(0,2)
    y_class = 'Iris-setosa'
    
    if class_num == 1:
        y_class = 'Iris-versicolor'
    elif class_num == 2:
        y_class = 'Iris-virginica'
  
    if (multinomial==False):
        iris_data = iris_data[iris_data.iloc[:, 4] != y_class]
    
     # Split data in X_train that contains values and y_train that contains the class
    X_train = iris_data.drop(iris_data.columns[[4]], axis=1)
    y_train = iris_data.drop(iris_data.columns[[0, 1, 2, 3]], axis=1)

    # Rename pandas' columns
    iris_data.columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class']
    X_train.columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    y_train.columns = ['Class']

    return iris_data, X_train, y_train

iris_data, X_train, y_train = edit_data(iris_data)    
# iris_data, X_train, y_train = edit_data(iris_data, True)

# Function to scatter the data in 3-d space
def scatter_circle(iris):
    from matplotlib import pyplot as plt
    from pandas.tools.plotting import radviz
    
    plt.figure()
    radviz(iris, 'Class')
    
scatter_circle(iris_data)

# We use LabelEncoder to convert non-categorical values
def encoding_data(data):
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    le.fit(data.iloc[:,0])
    data.iloc[:,0] = le.transform(data.iloc[:,0])
    
    return data
    
encoding_data(y_train) # Encode only y_train object which includes non-categorical values

# Some statistical info for our data
print "Data Stats\n" , X_train.describe()
print "Class Stats\n" , y_train.describe()

# Convert pandas objects to np arrays
X_train = X_train.as_matrix()
y_train = y_train.as_matrix()

# Sigmoid Function which will use for the Logistic Regression
def sigmoid(z):
    val = 1.0/(1+e**(-z))
    
    return val
    
# Function that calculates the error
def compute_error(theta, X, y):
    J = 0 # initialize error
    n = X.shape[0] #size of array

    # Body of for function
    for i in range(n):
        z = np.dot(X[i, :], theta)
        J -= y[i]*np.log(sigmoid(z)) + (1-y[i])*np.log(1-sigmoid(z))

    return J
    
def compute_grads(theta, X, y):
    grads = np.zeros(np.size(theta)) # initialize gradient
    n = X.shape[0]
    # Body of for function
    for i in range(n):
        z = np.dot(X[i, :], theta)
        for j in range(np.size(theta)):
            grads[j] -= (y[i]-sigmoid(z)) * X[i, j]

    return grads
    
def predict(v, theta):
    y_pred = sigmoid(np.dot(v, theta))
    
    if y_pred > 0.5:
        return 1
    else:
        return 0

def binomial_LR(X_train, y_train):
    x_new = np.ones((X_train.shape[0], 5))
    x_new[:, 1:] = X_train
    n = x_new.shape[1]
    
    initial_theta = np.zeros(n)
    for i in range(n):
        initial_theta[i] = np.random.randn()*0.1
    
    Result = op.minimize(fun = compute_error, x0 = initial_theta, args = (x_new, y_train), method = 'TNC',jac = compute_grads)
    theta = Result.x;
    
    y_pred = []
    for i in range(x_new.shape[0]):
        y_pred.append(predict(x_new[i, :], theta))
        
    print "Accuracy Score: ", accuracy_score(y_pred, y_train)

def multinomial_LR(X_train, y_train):
    x_new = np.ones((X_train.shape[0], 5))
    x_new[:, 1:] = X_train
    n = x_new.shape[1]
    
    initial_theta = np.zeros(n)
    for i in range(n):
        initial_theta[i] = np.random.randn()*0.1
    
    y_1 = np.array(y_train)
    y_2 = np.array(y_train)
    y_3 = np.array(y_train)
    
    y_1[y_1 == 1] = 2
    y_2[y_2 == 0] = 2
    y_3[y_3 == 1] = 0
    
    Result = op.minimize(fun = compute_error, x0 = initial_theta, args = (x_new, y_1), method = 'TNC',jac = compute_grads);
    theta1 = Result.x;
    
    Result = op.minimize(fun = compute_error, x0 = initial_theta, args = (x_new, y_2), method = 'TNC',jac = compute_grads);
    theta2 = Result.x;
    
    Result = op.minimize(fun = compute_error, x0 = initial_theta, args = (x_new, y_3), method = 'TNC',jac = compute_grads);
    theta3 = Result.x;
    
    y_pred = []
    for i in range(x_new.shape[0]):
        reg_term = ((e**(-(np.dot(theta1, x_new[i]))))+(e**(-(np.dot(theta3, x_new[i])))))
        pr_cl1 = (e**(-(np.dot(theta1, x_new[i]))))/(1 + reg_term)
        pr_cl2 = (1)/(1 + reg_term)
        pr_cl3 = (e**(-(np.dot(theta3, x_new[i]))))/(1 + reg_term)
                       
        pr_max = max(pr_cl1, pr_cl2, pr_cl3)
                 
        if pr_cl1 == pr_max:
            y_pred.append(2)
        elif pr_cl2 == pr_max:
            y_pred.append(1)
        else:
            y_pred.append(0)
            
    print "Accuracy Score: ", accuracy_score(y_train, y_pred)

binomial_LR(X_train, y_train)    
#multinomial_LR(X_train, y_train)