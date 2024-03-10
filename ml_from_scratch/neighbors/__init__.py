import numpy as np

class KNeighborsRegression():
    """
    Building a KNN class for regression
    """
    
    def __init__(self, k_neighbor=5, p=2, rounding = False):
        """
        Class initiation:
        k_neighbor = number of neighbor calculated for average
        p = the variable to determine the distance measuring method. p = 2 for Eucladian, p = 1 for Manhattan
        rounding = determine whether the results will be rounded to nearest integer
        """
        self.k_neighbor = k_neighbor
        self.p = p
        self.rounding = rounding

    def distance(self, point1, point2, point2value, p):
        """
        Method used for calculating the distance between two point. It varies according to the p value.
        """
        distance_value = np.power(np.sum((np.absolute(point1 - point2))**p), 1/p)
        distance_pair = [distance_value, point2value]
        return [distance_pair]


    def fit(self, X_train, y_train):
        """
        Method for model fitting. Saving X and y train to the model
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Method for predicting. The input is X_test and it will return ypred. It will use X_train and y_train
        that has been stored before.
        """
        y_pred = []
        X_test_shape = X_test.shape[0]
        X_train_shape = self.X_train.shape[0]

        for i in range(X_test_shape):       #For each X point in X_test
            distance_list = []              #Creating empty list for the distances
            for j in range(X_train_shape):  #For each X point in X_train
                d = self.distance(X_test[i,:],self.X_train[j,:], self.y_train[j], self.p) #Measuring distances
                distance_list.append(d)     #Add measured distance to the list
            distance_list = sorted(distance_list)   #Sort the distances
            neighbors = []
            for neighbor in range(self.k_neighbor):
                neighbors.append(distance_list[neighbor][0][1]) #Taking K number of closest neighbors
            
            if self.rounding == True:       #Rounding (or not rounding) the results
                y_pred.append(round(np.mean(neighbors)))        
            else:
                 y_pred.append(np.mean(neighbors))  #Averaging the y_train data for all nearest neighbors
        return y_pred          #Return the average results


    

class KNeighborsClassifier():
    """
    Building a KNN class for Classification
    """
    def __init__(self, k_neighbor=5, p=2):
        """
        Class initiation:
        k_neighbor = number of neighbor calculated for majority vote
        p = the variable to determine the distance measuring method. p = 2 for Eucladian, p = 1 for Manhattan
        """
        self.k_neighbor = k_neighbor
        self.p = p

    def distance(self, point1, point2, p):
        """
        Method used for calculating the distance between two point. It varies according to the p value.
        """
        return np.power(np.sum((np.absolute(point1 - point2))**p, axis=1), 1/p)
    
    def majority(self, set_neighbors):
        """
        Method used for choosing the output value according to majority vote
        """
        return max(set(set_neighbors), key=set_neighbors.count)
        
    def fit(self, X_train, y_train):
        """
        Method for model fitting. Saving X and y train to the model
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Method for predicting. The input is X_test and it will return ypred. It will use X_train and y_train
        that has been stored before.
        """
        neighbors = []
        for x in X_test:
            distances = self.distance(x, self.X_train, self.p)
            y_sort = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sort[:self.k_neighbor])
        return list(map(self.majority, neighbors))



    def score(self, X_test, y_test):
        """
        Method for evaluating the model
        """
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy
    
    