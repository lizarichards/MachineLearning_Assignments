#!/usr/bin/python3
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
#
import sys
import re
import math
# Node class for the decision tree
import node

train = None
varnames = None
test = None
testvarnames = None
root = None


# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
    # >>>> YOUR CODE GOES HERE <<<<

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
    # >>>> YOUR CODE GOES HERE <<<<

    parent_entropy = entropy(py / total)

    if (pxi == total) or (pxi == 0):
        return 0
    else:
        res1 = entropy(py_pxi/pxi)
        res2 = entropy((py - py_pxi)/(total - pxi))
        overall_res = parent_entropy - ((pxi/total) * res1) - (((total - pxi)/total) * res2)
        return overall_res


# OTHER SUGGESTED HELPER FUNCTIONS:

def collectCounts(data):
    counts = {}
    for var in varnames:
        counts[var] = [0, 0]

    for i, var in enumerate(varnames):
        for row in data:
            counts[var][row[i]] += 1

    return counts

# - collect counts for each variable value with each class label
# def counter(data, index):
#     # data is a list of lists
#     # determine how many samples there are
#     total_samples = len(data)
#     py = 0
#     pxi = 0
#     py_pxi = 0
#
#     for sample in data:
#         # determine py which is how many samples there are where y = 1
#         if sample[index] == 1:
#             pxi += 1
#
#         # determine how many are positive for the attribute being looked at which is pxi (x_i = 1)
#         if (sample[index] == 1) and (sample[-1] == 1):
#             py_pxi += 1
#
#         # determine where x_i = 1 and y = 1
#         if sample[-1] == 1:
#             py += 1
#
#     return py, pxi, py_pxi, total_samples

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

# - find the best variable to split on, according to mutual information
def find_split_attr(data, vars):  # what arguments go in here?
    best_attribute = vars[0]
    max_gain = 0
    best_index = 0
    # go through each index of data, the index with the highest info gain is the attribute that is the best
    for i, var in enumerate(vars[:-1]):
        py, pxi, py_pxi, total_samples = counter(data, var)
        temp_max_gain = infogain(py, pxi, py_pxi, total_samples)
        if temp_max_gain > max_gain:
            max_gain = temp_max_gain
            best_index = i
            best_attribute = var
    return best_attribute, best_index # this index will be used in the splitting function


# - partition data based on a given variable - will be the best one from previous helper
# the variable that is being split on is the one being handed into the function

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
    namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)


# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)


# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames): # what exactly is varnames?

    # >>>> YOUR CODE GOES HERE <<<<
    # NOTE: initialize the root of the tree

    # 1. the leaf is pure: all sublist[-1] values are either 1 or 0 return a leaf with that value

    # 2. No attributes can be split on that yield information gain.
    # Return a leaf with the dominant value from the -1 index

    # 3. The data can be split gainfully at an index (info gain > 0). Split it.
    # Call build tree to create the left and right children/roots
    # feed them into a new split node and return that

    # how do you stop recursion when the info gain does not meet threshold
    newVarnames = varnames[:]
    root = node.Node(varnames)
    sample_count = len(data)
    last_item_0_count = 0
    last_item_1_count = 0

    # checks if the leaf is pure
    for sample in data:
        if sample[-1] == 0:
            last_item_0_count += 1
        else:
            last_item_1_count += 1

    if last_item_0_count == sample_count:
        root = node.Leaf(newVarnames, 0)
    if last_item_1_count == sample_count:
        root = node.Leaf(newVarnames, 1)

    best_attribute, best_idx = find_split_attr(data, newVarnames)

    # case where no attributes can be split on the yield information gain
    if best_idx == 0:
        if last_item_1_count >= last_item_0_count:
            root = node.Leaf(newVarnames, 1)
        else:
            root = node.Leaf(newVarnames, 0)

    # case where a split can be done
    else:
        #newVarnames.remove(best_attribute)
        sublist1, sublist2 = split(data, best_idx)
        left = build_tree(sublist1, newVarnames)
        right = build_tree(sublist2, newVarnames)
        root = node.Split(newVarnames, best_idx, left, right)

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
