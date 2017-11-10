'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(0,10):
        i_digits=data.get_digits_by_label(train_data, train_labels, i)
        means+=np.sum(i_digits,axis=0)/len(i_digits)
    #print(means)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    for i in range(0,10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)

        mean=np.sum(i_digits,axis=0)/len(i_digits)
        means=np.zeros((i_digits.shape[0],64))
        for j in range(0,i_digits.shape[0]):
            means+=mean

        temp1=np.subtract(i_digits,means)
        temp2=np.dot(np.transpose(temp1),temp1)
        temp3=np.divide(temp2,len(i_digits))
        temp4=np.add(temp3,np.full((64,64),0.01))
        covariances[i,:,:]=temp4
    # Compute covariances
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    covs=[]
    for i in range(10):
        cov_diag = np.log(np.diag(covariances[i]))
        # ...
        cov = np.reshape(cov_diag, (8, 8))
        covs.append(cov)
    all_concat = np.concatenate(covs, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    res=np.zeros((digits.shape[0],10))
    for i in range(0,10):
        mean = means[i]
        covariance = covariances[i]
        temp1 = (np.linalg.det(covariance)) ** (-0.5)
        temp2 = np.subtract(digits, mean)
        temp0 = np.linalg.inv(covariance)
        temp3 = np.dot(temp2, temp0)
        temp4 = np.dot(temp3, np.transpose(temp2))
        temp5=np.diag(temp4)
        temp6 = -32*np.log(2*np.math.pi)-0.5*np.log(temp1)-0.5*np.log(temp5)
        res[:,i]=temp6
    return res

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    ptk=np.full((digits.shape[0],10),0.1)
    res=generative_likelihood(digits,means,covariances)+np.log(ptk)
    #print(res)
    return res

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    for i in range(0,digits.shape[])

    # Compute as described above and return
    return res

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    pass

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    plot_cov_diagonal(covariances)
    train_con_like=avg_conditional_likelihood(train_data, train_labels, means, covariances)
    print(train_con_like)

    test_means=compute_mean_mles(test_data, test_labels)
    test_covariances=compute_sigma_mles(test_data, test_labels)
    test_con_like=avg_conditional_likelihood(test_data, test_labels, test_means, test_covariances)
    print(test_con_like)
    # Evaluation

if __name__ == '__main__':
    main()