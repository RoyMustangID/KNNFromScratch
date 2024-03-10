import numpy as np

class KNeighborsRegression():
    """
    Building a KNN class for regression
    """
    
    def __init__(self, k_neighbor=5, p=2, rounding = False):
        """
        Class initiation
        k_neighbor = number of neighbor calculated for average
        p = the variable to determine the distance measuring method. p = 2 for Eucladian, p = 1 for Manhattan
        rounding = determine whether the results will be rounded to nearest integer
        """
        self.k_neighbor = k_neighbor
        self.p = p
        self.rounding = rounding

    def distance(self, point1, point2, point2value, p):
        
        distance_value = np.power(np.sum((np.absolute(point1 - point2))**p), 1/p)
        distance_pair = [distance_value, point2value]
        return [distance_pair]


    
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        X_test_shape = X_test.shape[0]
        X_train_shape = self.X_train.shape[0]

        for i in range(X_test_shape):
            distance_list = []
            for j in range(X_train_shape):
                d = self.distance(X_test[i,:],self.X_train[j,:], self.y_train[j], self.p)
                distance_list.append(d)
            distance_list = sorted(distance_list)
            distance_list
            neighbors = []
            for neighbor in range(self.k_neighbor):
                neighbors.append(distance_list[neighbor][0][1])
            
            if self.rounding == True:
                y_pred.append(round(np.mean(neighbors)))
            else:
                 y_pred.append(np.mean(neighbors))
        return y_pred


    

class KNeighborsClassifier():
    def __init__(self, k_neighbor=5, p=2):
        self.k_neighbor = k_neighbor
        self.p = p

    def distance(self, point1, point2, p):
        return np.power(np.sum((np.absolute(point1 - point2))**p, axis=1), 1/p)
    
    def majority(self, set_neighbors):
        return max(set(set_neighbors), key=set_neighbors.count)
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.distance(x, self.X_train, self.p)
            y_sort = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sort[:self.k_neighbor])
        return list(map(self.majority, neighbors))



    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy
    
    