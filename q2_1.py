'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
from scipy.stats import mode
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        distances=self.l2_distance(test_point)
        indice_larg=distances.argsort()[:k]
        digits = self.train_labels[indice_larg]
        temp=np.bincount(digits.astype(int))
        max=[]
        digit=0
        for i in range(temp.shape[0]):

            if len(max)>0 and temp[i] > max[0]:
                max = []
                max.append(temp[i])
                digit = i
            elif len(max)==0 or temp[i] == max[0]:
                max.append(temp[i])
        if len(max)==1:
            return digit
        return self.query_knn(test_point, k-1)

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    i=1
    for k in k_range:
        #Loop over folds
        #Evaluate k-NN
        ...
        kf = KFold(n_splits=10)
        kf.get_n_splits(train_data)
        KFold(n_splits=10, random_state=None, shuffle=False)
        accu_sum=0
        for train_index, test_index in kf.split(train_data):
            X_train, X_test = train_data[train_index], train_data[test_index]
            y_train, y_test=  train_labels[train_index], train_labels[test_index]
            knn = KNearestNeighbor(X_train, y_train)
            accu_sum+=classification_accuracy(knn, k, X_test, y_test)
        avg_accu=accu_sum/10
        print("Average accuracy "+str(i)+"："+ str(avg_accu))
        i+=1
        pass

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    count=0
    for i in range(eval_data.shape[0]):
        predicted_label=knn.query_knn(eval_data[i], k)
        if eval_labels[i]==predicted_label:
            count+=1
    res=count/eval_data.shape[0]
    return res
    pass

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # Example usage:
    #predicted_label = knn.query_knn(test_data[0], 1)

    ac_train1=classification_accuracy(knn, 1, train_data, train_labels)
    ac_test1=classification_accuracy(knn, 1, test_data, test_labels)
    print("Train Accuracy when k=1: "+str(ac_train1))
    print("Test Accuracy when k=1: "+str(ac_test1))

    ac_train15=classification_accuracy(knn, 15, train_data, train_labels)
    ac_test15=classification_accuracy(knn, 15, test_data, test_labels)
    print("Train Accuracy when k=15: "+str(ac_train15))
    print("Test Accuracy when k=15: "+str(ac_test15))


    cross_validation(train_data, train_labels, k_range=np.arange(1, 16))

    res=classification_accuracy(knn, 3, train_data, train_labels)
    print("Train Accuracy: "+str(res))

    res = classification_accuracy(knn, 3, test_data, test_labels)
    print("Test Accuracy: " + str(res))

if __name__ == '__main__':
    main()