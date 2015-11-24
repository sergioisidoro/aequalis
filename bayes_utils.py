import math
import csv
import string
import scipy.stats as stats
from collections import Counter

import multiprocessing
pool = multiprocessing.Pool()

# Reused some code to avoid boilerplate code
# Credit to:
# http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/


def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [x for x in dataset[i]]
    return dataset


def uniform_data(dataset):
    return [[t.strip().replace(".", "") for t in i] for i in dataset]


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg, 2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)


def separateByClass(dataset):
    # Dataset is a list of vectors, with label being the last index
    # [feature1, feature2, feature3, label]
    # Separates a dataset into a dict with each label
    # { label1: [feature1, feature2, feature3] }
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def summarize(dataset):
    # Dataset is a list of lists. Zip will aggregate all the same features
    # eg. [a,b,c] [g,t,e] -> [(a,g), (b,t), (c,e)]
    summaries = [
        (mean(attribute), stdev(attribute))
        for attribute in zip(*dataset)]
    # The las summary is the summary of the labels.
    del summaries[-1]
    return summaries


def discretize_variable(dataset, index):
    # Inneficiently puts a constant variable into one of the 4 percentiles

    print "Discretizing... variable %s" % index
    variable_values = [sample[index] for sample in dataset]
    new_values = stats.rankdata(variable_values, "average")/len(variable_values)

    for i in range(0, len(dataset)):
        dataset[i][index] = str(int(round((new_values[i]*100)/25)))
    return dataset


def discrete_summarize(dataset):
    # Dataset is a list of lists. Zip will aggregate all the same features
    # eg. [a,b,c] [g,t,e] -> [(a,g), (b,t), (c,e)]
    summaries = [
        dict(Counter(attribute))
        for attribute in zip(*dataset)]
    # The las summary is the summary of the labels.
    del summaries[-1]
    return (summaries, len(dataset))


def summarizeByClass(dataset):
    print dataset
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries


def discrete_summarize_by_class(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = discrete_summarize(instances)
    return summaries

def discrete_summarize_total(dataset):
    summaries = discrete_summarize(dataset)
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev, 2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateDiscreteProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev, 2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(
                x, mean, stdev)
    return probabilities
