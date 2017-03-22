import numpy as np
import pydot
import math
import time

def draw(parent_name, child_name):
    edge = pydot.Edge(parent_name, child_name)
    graph.add_edge(edge)

def visit(node, parent=None):
    for k,v in node.iteritems():
        if isinstance(v, dict):
            # We start with the root node whose parent is None
            # we don't want to graph the None node
            if parent:
                draw(parent, k)
            visit(v, k)
        else:
            draw(parent, k)
            # drawing the label using a distinct name
            draw(k, k+'_'+v)

def buildTree(dataset, features):
    # get oracle total values
    classList = [sample[-1] for sample in dataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    max_index = calcMaxAttributeGain(dataset)
    attribute_max = features[max_index]
    myTree = {attribute_max: {}}
    fea_val = [sample[max_index] for sample in dataset]
    unique = set(fea_val);
    del (features[max_index])

    for values in unique:
        sub_dataset = splitDataSet(dataset, max_index, values)
        myTree[attribute_max][values] = buildTree(sub_dataset, features)
    features.insert(max_index, attribute_max)

    return myTree


def calcMaxAttributeGain(data_set):
    numFeatures = len(data_set[1, :])
    gain = 1
    attribute_max = -1
    for value in range(numFeatures - 1):
        InfoGain = featureInfo(data_set[:, [value, -1]])
        if (gain > InfoGain):
            gain = InfoGain
            attribute_max = value
    return attribute_max

def calc_entropy(oracleValues):
    entTotal = 0.0

    for key in oracleValues:
        s = 0.0
        for label in oracleValues[key]:
            s += oracleValues[key][label]
        featAttribute = 0.0
        for label in oracleValues[key]:
            prob = float(oracleValues[key][label] / s)
            if prob != 0:
                featAttribute -= prob * math.log(prob, 2)

        entTotal += s / len(data[:, 0]) * featAttribute
    return entTotal


def featureInfo(data):
    # get feature attributes & oracle values
    valueDic = {}
    for value in data:
        if value[0] not in valueDic.keys():
            valueDic[value[0]] = {}
            valueDic[value[0]][value[1]] = 1
        elif value[1] not in valueDic[value[0]]:
            valueDic[value[0]][value[1]] = 1
        else:
            valueDic[value[0]][value[1]] += 1
    return calc_entropy(valueDic)


def splitDataSet(dataSet, featureIndex, value):
    subDataSet = []
    dataSet = dataSet.tolist()
    for sample in dataSet:
        if sample[featureIndex] == value:
            reducedSample = sample[:featureIndex]
            reducedSample.extend(sample[featureIndex + 1:])
            subDataSet.append(reducedSample)
    return np.asarray(subDataSet)


def print_dict(dictionary, indent = '', braces=1):
    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            print '%s%s%s%s' %(indent, 'IF ', key, ':')
            print_dict(value, indent+'  ', braces+1)
        else:
            print indent+'%s = %s' %(key, value)


if __name__ == "__main__":
    with open('car.csv', 'r') as inputFile:
        lines = inputFile.readlines()

    data = [line.strip().split(',') for line in lines]
    data = np.array(data)

    '''Feature values for car.csv'''
    featureSet_car = ["Cost", "Maintenance", "Doors", "Persons", "Trunk", "Safety"]
    '''Feature values for fishing.csv'''
    featureSet_fishing = ["Wind", "Water", "Air", "Forecast"]
    '''Feature values for contacts.csv'''
    featureSet_contacts = ["Age", "Prescription", "Astigmatism", "Tear-Rate"]
    start = time.clock()
    tree = buildTree(data, featureSet_car)
    print "\tTime: %s Seconds" % (time.clock() - start)
    pydot.Dot(type=tree)

    graph = pydot.Dot(graph_type='graph')
    visit(tree)
    graph.write_png('car.png')

