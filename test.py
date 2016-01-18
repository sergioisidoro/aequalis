#!/usr/bin/env python

from bayes_utils import *
from bayes import *
import matplotlib.pyplot as plt

data_file = './data/adult.data'
data = loadCsv(data_file)

data = discretize_variable(data, 0)
data = discretize_variable(data, 2)
data = discretize_variable(data, 4)
data = discretize_variable(data, 10)
data = discretize_variable(data, 11)
data = discretize_variable(data, 12)

data = uniform_data(data)

test_data_file = './data/adult.test'
test_data = loadCsv(test_data_file)

test_data = discretize_variable(test_data, 0)
test_data = discretize_variable(test_data, 2)
test_data = discretize_variable(test_data, 4)
test_data = discretize_variable(test_data, 10)
test_data = discretize_variable(test_data, 11)
test_data = discretize_variable(test_data, 12)

test_data = uniform_data(test_data)

# BAYES MODEL
basic_model = BinaryBayesModel()
basic_model.train(data)

print ("NORMAL MODEL:")
print ("Accuracy: %s " % basic_model.test(test_data)[0])
print ("Discrimination score: %s " %  basic_model.discrimination_measure(9, '>50K', test_data))

# 2M BAYES MODEL
two_m_model = SplitFairBayesModel(sensitive_parameter_indexes=[9])
two_m_model.train(data)

print ("2M MODEL:")
print ("Accuracy: %s " % two_m_model.test(test_data)[0])
print ("Discrimination score: %s " %  two_m_model.discrimination_measure(9, '>50K', test_data))


# print(basic_model.discrimination_measure(9, '>50K', test_data))

modified_model = BalancedBayesModel()
modified_model.train(data)

plot_data = modified_model.balance_model(
        index=9, discriminated_class="Female", privileged_class="Male",
        positive_label='>50K', negative_label='<=50K', train_data=data)

print ("MODIFIEND MODEL:")
print ("Accuracy: %s " % modified_model.test(test_data)[0])
print ("Discrimination score: %s " %
       modified_model.discrimination_measure(
            index=9, discriminated_class="Female", privileged_class="Male",
            positive_label='>50K', test_data=test_data)[0])

fig, ax1 = plt.subplots()
line = ax1.plot(list((t[0] for t in plot_data)))
plt.setp(line, color='b')
ax1.set_ylabel('Discrimination score')

for tl in ax1.get_yticklabels():
        tl.set_color('b')

ax2 = ax1.twinx()
line2 = ax2.plot(list((t[1] for t in plot_data)))

for tl in ax2.get_yticklabels():
        tl.set_color('r')
plt.setp(line2, color='r')
ax2.set_ylabel('Accuracy')

ax2.set_xlabel('Runs')
plt.show()
