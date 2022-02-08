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


# Helper function that computes the sigmoid function for an argument z
def sigmoid(z):
    return 1/(1+exp(-z))


# def mult(l1, l2):
#     result = 0
#     if len(l1) != len(l2):
#         print("lists must be of same length")
#         return None
#     for i in range(len(l1)):
#         result += (l1[i] * l2[i])
#     return result


# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
    numvars = len(data[0][0])
    w = [0.0] * numvars
    b = 0.0
    for _ in range(MAX_ITERS):
        for d in range(len(data)):
            x = data[d][0]
            y = data[d][1]
            for k in range(numvars):
                dot_product = np.dot(w, x)

                # Where does regularization come into play??? it comes into play in the weight gradient when you add
                # the l2_reg_weight and w[k]

                cost = log(sigmoid(y * (dot_product + b)))

                # weights gradient
                dw = ((sigmoid(-y * (dot_product + b))) * (y * x[k])) + (l2_reg_weight * w[k])

                #bias gradient
                db = (sigmoid(-y * (dot_product + b))) * y

                if abs(cost) >= 0.0001:
                    w[k] -= eta * dw
                    b -= eta * db
                    # b += y
                else:
                    return w, b

    return w, b


# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
    (w, b) = model
    return sigmoid(np.dot(w, x) + b)

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