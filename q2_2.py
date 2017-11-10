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
        means[i,:]=np.sum(i_digits,axis=0)/i_digits.shape[0]
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
        temp1=i_digits-mean
        temp2=np.dot(np.transpose(temp1),temp1)
        temp3=np.divide(temp2,len(i_digits))
        temp4=np.add(temp3,0.01*np.identity(64))
        covariances[i,:,:]=temp4
    # Compute covariances
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    covs=[]
    for i in range(10):
        cov_diag = np.log(np.diagonal(covariances[i]))
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
    res=np.zeros((10,digits.shape[0]))
    for i in range(0,10):
        mean = means[i]
        covariance = covariances[i]
        temp1 = np.linalg.det(covariance)
        temp0 = np.linalg.inv(covariance)
        temp3 = np.dot(digits-mean, temp0)
        temp4 = np.dot(temp3, np.transpose(digits-mean))
        temp5=np.diagonal(temp4)
        temp6 = -(digits.shape[1]/2)*np.log(2*np.pi)-0.5*np.log(temp1)-0.5*temp5
        temp6=temp6.reshape(1,-1)
        res[i]=temp6
    tt=np.transpose(res)
    return np.transpose(res)

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    temp = generative_likelihood(digits, means, covariances)
    temp=np.exp(temp)
    sum=np.sum(temp,axis=1)
    res=np.zeros((digits.shape[0],10))
    for i in range(digits.shape[0]):
        sum_temp=sum[i]
        temp_res=temp[i,:]
        res[i,:]=temp[i,:]/sum[i]
    return np.log(res)

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    sum=0
    for i in range(0,digits.shape[0]):
        label=labels[i]
        sum+=cond_likelihood[i,int(label)]
    res=sum/digits.shape[0]

    # Compute as described above and return
    return res

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    temp=np.argmax(cond_likelihood,axis=1)
    return np.argmax(cond_likelihood,axis=1)
    pass

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    plot_cov_diagonal(covariances)
    train_con_like=avg_conditional_likelihood(train_data, train_labels, means, covariances)
    print(train_con_like)

    test_con_like=avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print(test_con_like)
    predict_train_labels=classify_data(train_data, means, covariances)
    predict_test_labels=classify_data(test_data, means, covariances)
    train_accuracy=np.sum(predict_train_labels == train_labels)/predict_train_labels.shape[0]
    test_accuracy = np.sum(predict_test_labels == test_labels) / predict_test_labels.shape[0]
    print(train_accuracy)
    print(test_accuracy)
    # Evaluation

if __name__ == '__main__':
    main()