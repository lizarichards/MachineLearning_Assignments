#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
# Completed with help from Tammas Hicks
#
#
import sys
import re
from math import log
from math import exp
from math import sqrt
import math
#import numpy as np

MAX_ITERS = 100


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        example = [int(x) for x in p.split(l.strip())]
        x = example[0:-1]
        y = example[-1]
        data.append((x, y))
    return (data, varnames)


def sigmoid(z):
    return 1 / (1 + exp(z))

def mult(l1, l2):
    result = 0
    if len(l1) != len(l2):
        print("lists must be of same length")
        return None
    for i in range(len(l1)):
        result += (l1[i] * l2[i])
    return result


# I have two gradient calculators, but I couldn't figure out how to implement them correctly
def calc_weight_gradient(w, data, b, l2_reg_weight):
    gradient = [0.0] * len(w)
    for i in range(len(w)):
        gradient_sum = 0
        for sample, value in data:
            dot_prod = np.dot(w, sample)
            gradient_sum += sigmoid(value * (dot_prod + b))
        gradient[i] = -(gradient_sum * value * sample[i]) + (w[i] * l2_reg_weight)
    return gradient

def calc_bias_grad(w, data, b):

    gradient = 0
    for i in range(len(w)):
        gradient_sum = 0
        for sample, value in data:
            dot_prod = np.dot(w, sample)
            gradient_sum += sigmoid(value * (dot_prod + b))
        gradient = -(gradient_sum * value)
    return gradient



def calc_mag(w, b):
    magnitude = 0
    for item in w:
        magnitude += (item ** 2)
    magnitude += b ** 2
    return sqrt(magnitude)


# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
    # add regularizer gradient to gradient of logisitic loss
    numvars = len(data[0][0])
    w = [0.0] * numvars
    b = 0.0

    for item in range(MAX_ITERS):
        weight_gradient = [0.0] * numvars
        bias_gradient = 0

        for sample, value in data:
            # x = sample = data[sample][0]
            # y = value = data[sample][1]
            # bias_gradient = (sigmoid(-value * (dot_product + b))) * value
            #sigmoid_result = sigmoid(value * (np.dot(w, sample) + b))
            sigmoid_result = sigmoid(value * (mult(w, sample) + b))
            bias_gradient -= value * sigmoid_result

            for k in range(numvars):
                # weight_gradient = ((sigmoid(-value * (dot_product + b))) * (value * x[k])) + (l2_reg_weight * w[k])
                # l2_reg_gradient = l2_reg_weight * w[k]
                # weight_gradient[k] += -error * value * sample[k] + l2_reg_gradient
                # -> can't do this because not fully updated yet...
                weight_gradient[k] += -sigmoid_result * value * sample[k]  # gets weight gradients

        for i in range(numvars):
            # add the gradient of the regularization (lambda * w[j])
            weight_gradient[i] += w[i] * l2_reg_weight

        for j in range(numvars):
            w[j] -= eta * weight_gradient[j]
        b -= eta * bias_gradient

        mag = calc_mag(weight_gradient, bias_gradient)
        if mag < 0.0001:
            break

                # if abs(loss) >= 0.0001:
                #     w[k] -= eta * weight_gradient
                #     b -= eta * bias_gradient
                # else:
                #     return (w, b)

    return (w,b)



# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
    (w, b) = model
   #return np.dot(w, x) + b
    return mult(w, x) + b



# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 5):
        print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    eta = float(argv[2])
    lam = float(argv[3])
    modelfile = argv[4]

    # Train model
    (w, b) = train_lr(train, eta, lam)

    # Write model file
    f = open(modelfile, "w+")
    f.write('%f\n' % b)
    for i in range(len(w)):
        f.write('%s %f\n' % (varnames[i], w[i]))

    # Make predictions, compute accuracy
    correct = 0
    for (x, y) in test:
        prob = predict_lr((w, b), x)
        print(prob)
        if (prob - 0.5) * y > 0:
            correct += 1
    acc = float(correct) / len(test)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
