from bayes_utils import *
from bayes import BinaryBayesModel, BalancedFairBayesModel

data_file = './data/adult.data'
data = loadCsv(data_file)

data = discretize_variable(data, 0)
data = discretize_variable(data, 2)
data = discretize_variable(data, 4)
data = discretize_variable(data, 10)
data = discretize_variable(data, 11)
data = discretize_variable(data, 12)

data = uniform_data(data)

summary = discrete_summarize_by_class(data)

global_summary = discrete_summarize_total(data)

basic_model = BalancedFairBayesModel()
basic_model.train(data)

basic_model.evaluate(data[0])


test_data_file = './data/adult.test'
test_data = loadCsv(test_data_file)

test_data = discretize_variable(test_data, 0)
test_data = discretize_variable(test_data, 2)
test_data = discretize_variable(test_data, 4)
test_data = discretize_variable(test_data, 10)
test_data = discretize_variable(test_data, 11)
test_data = discretize_variable(test_data, 12)

test_data = uniform_data(test_data)

print(basic_model.discrimination_measure(9, '>50K', test_data))
