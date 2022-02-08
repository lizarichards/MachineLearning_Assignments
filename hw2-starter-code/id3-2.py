#!/usr/bin/python3
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
#
import sys
import re
# Node class for the decision tree
import node
import math

train = None
varnames = None
test = None
testvarnames = None
root = None


# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
    if (p == 0) or (p == 1):  # 0.0
        entropy_res = 0  # set to zero
    else:
        p_2 = 1 - p
        entropy_res = -1 * (p * math.log2(p)) - (p_2 * math.log2(p_2))

    return entropy_res


# Compute information gain for a particular split, given the counts
# py_pxi : number of occurrences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of occurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):

    if total == 0:
        return 0
    parent_entropy = entropy(py / total)

    if (pxi == total) or (pxi == 0):
        return 0
    else:
        res1 = entropy(py_pxi / pxi)
        res2 = entropy((py - py_pxi) / (total - pxi))
        overall_res = parent_entropy - ((pxi / total) * res1) - (((total - pxi) / total) * res2)
        return overall_res



# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
def collectCounts(data):
    counts = {}
    for var in varnames:
        counts[var] = [0, 0]

    for i, var in enumerate(varnames):
        for row in data:
            counts[var][row[i]] += 1

    return counts

# - find the best variable to split on, according to mutual information
def bestVar(data, vars):
    best = vars[0]
    maxGain = 0
    idx = 0
    for i, var in enumerate(vars[:-1]):
        py_pxi, pxi, py, total = counter(data, var)
        newGain = infogain(py_pxi, pxi, py, total)
        if newGain > maxGain:
            maxGain = newGain
            best = var
            idx = i
    return idx, best, maxGain

    # max_gain = 0
    # best_index = 0
    # size = len(data) - 1
    # # go through each index of data, the index with the highest info gain is the attribute that is the best
    # for index in range(0, size):
    #     py, pxi, py_pxi, total_samples = counter(data, index)
    #     temp_max_gain = infogain(py, pxi, py_pxi, total_samples)
    #     if temp_max_gain > max_gain:
    #         max_gain = temp_max_gain
    #         best_index = index
    # return best_index # this index will be used in the splitting function


# - partition data based on a given variable
def counter(data, var):
    counts = collectCounts(data)

    total = len(data)
    pxi = counts[var][1]
    py_pxi = 0
    py = 0

    for sample in data:
        if sample[-1] == 1:
            py += 1

    for i, name in enumerate(varnames):
        if name == var:
            for sample in data:
                if sample[i] == 1 and sample[-1] == 1:
                    py_pxi += 1

    return py_pxi, pxi, py, total

def split(data, index):

    sublist1 = []
    sublist2 = []
    for sample in data:
        if sample[index] == 0:
            sublist1.append(sample)
        else:
            sublist2.append(sample)

    return sublist1, sublist2

# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    # namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return data, varnames


# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)


# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames):

    newVarnames = varnames[:]
    root = node.Node(newVarnames)
    idx, best, maxGain = bestVar(data, newVarnames)

    counts = collectCounts(data)
    if len(data) == 0 or ((len(newVarnames) - 1) <= 0):
        root = node.Leaf(newVarnames, 1)

    elif counts[best].count(0) == 1:
        if counts[best].index(0) == 0:
            root = node.Leaf(newVarnames, 1)
        else:
            root = node.Leaf(newVarnames, 0)

    elif counts[best][0] == counts[best][1]:
        root = node.Leaf(newVarnames, 1)

    else:
        leftData, rightData = split(data, idx)
        left = build_tree(leftData, newVarnames)
        right = build_tree(rightData, newVarnames)
        root = node.Split(newVarnames, idx, left, right)

    return root

# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS, testS, modelS):
    global train
    global varnames
    global test
    global testvarnames
    global root
    (train, varnames) = read_data(trainS)
    (test, testvarnames) = read_data(testS)
    modelfile = modelS

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)



def runTest():
    correct = 0
    # The position of the class label is the last element in the list.
    yi = len(test[0]) - 1
    for x in test:
        # Classification is done recursively by the node class.
        # This should work as-is.
        # print(root.var)
        pred = root.classify(x)
        if pred == x[yi]:
            correct += 1
    acc = float(correct) / len(test)
    return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
        print('Usage: python3 id3.py <train> <test> <model>')
        sys.exit(2)
    loadAndTrain(argv[0], argv[1], argv[2])

    acc = runTest()
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])