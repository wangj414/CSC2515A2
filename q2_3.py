'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    temp=binarize_data(train_data)
    eta = np.zeros((10, 64))
    for i in range(10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        ones = np.zeros((1,64))
        zeros = np.zeros((1,64))
        for j in range(0,64):
            ones[:,j]=1
        i_digits = np.append(i_digits, zeros, axis=0)
        i_digits = np.append(i_digits, ones,axis=0)
        N=i_digits.shape[0]
        Nc=np.sum(i_digits,axis=0)
        temp2=(Nc+2)/(N+4)
        eta[i,:]=temp2
    #print(eta)
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    images=[]
    for i in range(10):
        img_i = class_images[i]
        # ...
        img_i = np.reshape(img_i, (8, 8))
        images.append(img_i)
    all_concat = np.concatenate(images, 1)
    #plt.imshow(all_concat, cmap='gray')
    #plt.show()


def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    temp1=np.dot(bin_digits,np.transpose(np.log(eta)))
    temp2=np.dot(1-bin_digits,np.transpose(np.log(1-eta)))
    res=temp1+temp2
    return res

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    temp = generative_likelihood(bin_digits, eta)
    temp = np.exp(temp)
    sum = np.sum(temp, axis=1)
    res = np.zeros((bin_digits.shape[0], 10))
    for i in range(bin_digits.shape[0]):
        res[i, :] = temp[i, :] / sum[i]
    return np.log(res)

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    sum = 0
    for i in range(0, bin_digits.shape[0]):
        label = labels[i]
        sum += cond_likelihood[i, int(label)]
    res = sum / bin_digits.shape[0]

    # Compute as described above and return
    return res


def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)
    pass

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)

    generate_new_data(eta)

    train_con_like = avg_conditional_likelihood(train_data, train_labels, eta)
    print(train_con_like)

    test_con_like = avg_conditional_likelihood(test_data, test_labels, eta)
    print(test_con_like)
    predict_train_labels = classify_data(train_data, eta)
    predict_test_labels = classify_data(test_data, eta)
    train_accuracy = np.sum(predict_train_labels == train_labels) / predict_train_labels.shape[0]
    test_accuracy = np.sum(predict_test_labels == test_labels) / predict_test_labels.shape[0]
    print(train_accuracy)
    print(test_accuracy)

if __name__ == '__main__':
    main()
