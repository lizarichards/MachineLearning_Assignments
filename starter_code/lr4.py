#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import sys
import re
from math import log
from math import exp
from math import sqrt
import math
import numpy as np

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
    return 1/(1+exp(-z))


def calc_weight_gradient(w, data, b, l2_reg_weight):
    #gradient = [0.0] * len(w)
    gradient = 0
    for i in range(len(w)):
        gradient_sum = 0
        for row in range(len(data)):
            dot_prod = np.dot(w, data[row][0])
           # gradient_sum += ((row[1] * row[0][i]) / (1 + exp(row[1] * (dot_prod + b))))
            gradient_sum += sigmoid(data[row][1] * (dot_prod + b))
            #gradient[i] = -(gradient_sum + (l2_reg_weight * w[i]))
            gradient = -(gradient_sum)
    return gradient

def calc_bias_grad(w, data, b):
    # # gradient = 0
    # gradient_sum = 0
    # for row in data:
    #     dot_prod = np.dot(w, data[row][0])
    #     #gradient_sum += ((row[1] * row[0][0]) / (1 + exp(row[1] * (dot_prod + b))))
    #     gradient_sum += 1 / (1 + exp(row[1] * (dot_prod + b)))
    # gradient = -(gradient_sum * data[row][1])
    # return gradient
    #gradient = [0.0] * len(w)
    gradient = 0
    for i in range(len(w)):
        gradient_sum = 0
        for row in range(len(data)):
            dot_prod = np.dot(w, data[row][0])
           # gradient_sum += ((row[1] * row[0][i]) / (1 + exp(row[1] * (dot_prod + b))))
            gradient_sum += sigmoid(data[row][1] * (dot_prod + b))
        gradient = -(gradient_sum)
    return gradient



# def calc_loss_gradient(data, w, l2_reg_weight, b):
#     gradient = [0.0] * len(w)
#     loss = calc_weight_gradient(w, data, b, l2_reg_weight) + calc_bias_grad(w, data, b)
#     for i in range(len(w)):
#         gradient_sum = 0
#         for row in data:
#             dot_prod = np.dot(w, row[0])
#             gradient_sum += ((row[1] * row[[0][i]]) / (1 + exp(row[1] * (dot_prod + b))))
#         gradient[i] = -(gradient_sum + l2_reg_weight * w[i])
#     return gradient



# def calc_regularizer(data, w, l2_reg_weight):
#     #numvars = len(data[0][0])
#     #w = [0.0] * numvars
#     gradient_sum = 0
#
#     for weight in range(len(w)):
#         for i in data:
#             gradient_sum += (w[i] ** 2)
#     reg = (l2_reg_weight / 2) * gradient_sum
#     return reg


# def calc_reg_gradient(data, w, l2_reg_weight):
#     #reg_gradient = 0 # do i want this to be a list?
#     reg_gradient = [0.0] * len(w)
#     reg = calc_regularizer(data, w, l2_reg_weight)
#     for i in range(len(data)):
#         for j in range(len(data[i][0])):
#             reg_gradient[i] = l2_reg_weight * reg[i][0][j]
#
#     return reg_gradient


def calc_mag(w, b):
    magnitude = 0
    for item in w:
        magnitude += (item ** 2)
    magnitude += b ** 2
    return sqrt(magnitude)


# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
    # add regularizer gradient to gradient of logisitic loss
    # logistic loss gradient is the bias gradient + the weight gradient
    numvars = len(data[0][0])
    w = [0.0] * numvars
    b = 0.0


    #gradient = [0.0] * len(w)
    #gradient_sum = 0

    # logistic_loss = calc_weight_gradient(w, data, b, l2_reg_weight) + calc_bias_grad(w, data, b)
    for item in range(MAX_ITERS):
        for row in range(len(data)):
            for k in range(numvars):
                dot_prod = np.dot(w, data[row][0])
                weight_gradient = calc_weight_gradient(w, data, b, l2_reg_weight) * (data[row][1] * data[row][0][k]) + (l2_reg_weight * w[k])
                bias_gradient = calc_bias_grad(w, data, b) * (data[row][1])
                loss = log(sigmoid(-data[row][1] * (dot_prod + b)))

                if abs(loss) >= 0.0001:
                    w[k] -= eta * weight_gradient
                    b -= eta * bias_gradient
                else:
                    return (w, b)
    return (w,b)

    # eta is the learning rate
    # for i in range(len(data)):
    #     for row in range(len(data)):
    #         for y in range(len(data[row][0])):
    #             logistic_loss_grad = calc_weight_gradient(w, data, b, l2_reg_weight) + calc_bias_grad(w, data, b)
    #             #reg_gradient = calc_reg_gradient(data, w, l2_reg_weight)
    #             #res = logistic_loss + reg_gradient
    #             dot_prod = np.dot(w, data[row][0])
    #             gradient_sum += log(1 + exp(-data[row][0][y] *(data[row][1] * (dot_prod + b))))
    #             w[i] = w[i - 1] - eta * gradient_sum
    #             b += data[row][1]
    #     gradient[i] = -(gradient_sum + l2_reg_weight * w[i])
    #
    #     mag = calc_mag(w, b)
        #if mag < 0.0001:
            # means it has converged, what do we do now
            # return (w, b)?
    # return gradient

    # for i in range(MAX_ITERS):
    #     for row in range(len(data)):
    #         for i in range(len(data[row][0])):
    #             # activation += (w[i] * data[row][0][i]) + b

    #return (w, b)


# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
    (w, b) = model
    prediction = b

    for i in range(len(x)):
        prediction += (w[i] * x[i])

    # this needs to return a probability
    if prediction == 1:
        return 1
    else:
        return -1

    # return 0.5 # This is an random probability, fix this according to your solution


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
