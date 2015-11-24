from bayes_utils import *

class BinaryBayesModel(object):
    summaries = None
    global_summary = None
    total_len = 0

    def train(self, train_data):
        self.summaries = discrete_summarize_by_class(train_data)
        self.global_summary = discrete_summarize_total(train_data)
        self.total_len = len(train_data)
        return (self.summaries, self.global_summary, self.total_len)

    def evaluate(self,
                 input, summaries=None,
                 global_summary=None, total_len=None):
        # Evaluates the probabilies of input beloging to each set of
        # classes

        if not summaries:
            summaries = self.summaries

        if not global_summary:
            global_summary = self.global_summary

        if not total_len:
            total_len = self.total_len

        probabilities = {}
        for classValue, classSummaries in summaries.iteritems():
            summary = classSummaries[0]
            total = classSummaries[1]
            probabilities[classValue] = float(1)

            for i in range(len(summary)):
                x = input[i]
                p_i_class = \
                    float(summary[i].get(x, 0))/float(total) * \
                    (float(global_summary[0][i].get(x, 0))/float(total_len))
                probabilities[classValue] *= float(p_i_class)
        return probabilities

    def predict(self, input):
        probabilities = self.evaluate(input)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.iteritems():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel

    def getPredictions(self, testSet):
        predictions = []
        for i in range(len(testSet)):
            result = self.predict(testSet[i])
            predictions.append(result)

        return predictions

    def test(self, test_data):
        def _getAccuracy(testSet, predictions):
            correct = 0
            for i in range(len(test_data)):
                if test_data[i][-1] == predictions[i]:
                    correct += 1
            return (correct/float(len(test_data))) * 100.0

        predictions = self.getPredictions(test_data)
        accuracy = _getAccuracy(test_data, predictions)
        return (accuracy, predictions)

    def discrimination_measure(self, index, label, test_data):
        # A simteric disctimination measure
        predictions = self.getPredictions(test_data)

        test_global_summary = discrete_summarize_total(test_data)
        # entitnes in the lablel
        results = {}

        possible_values = list(
            set([sample[index] for sample in test_data]))
        for possible_value in possible_values:
            results[possible_value] = 0

        for i in range(0, len(predictions)):
            if predictions[i] == label:
                results[test_data[i][index]] += 1

        for key, value in test_global_summary[0][index].iteritems():
            results[key] = float(results[key])/value

        # warning, this only supports 2 classes for now:

        values = results.values()

        return max(values) - min(values)


class SplitedFairBayesModel(BinaryBayesModel):
    # This model splits the model into sub models individually, for each
    # value of a sensitive variable.

    sensitive_params_summaries = {}
    sensitve_param_indexes = []

    def __init__(self, sensitive_parameter_indexes):
        super(SplitedFairBayesModel,self).__init__()
        self.sensitve_param_indexes = sensitive_parameter_indexes

    def train(self, train_data):
        # WARNING, ONLY WORKS CURRENTY WITH ONE SENSITIVE PARAMETER.
        # SPLITING THE MODEL IN EXPONENCIAL NUMBER OF MODELS, AND WILL
        # BORK THE WHOLE THING

        # Lets train a model for each sensitive parameters
        for index in self.sensitve_param_indexes:
            self.sensitive_params_summaries[index] = {}
            # Get all unique values for each of the sensitive parameters
            # making a set, and getting back to list does the trick
            possible_values = list(
                set([sample[index] for sample in train_data]))

            for possible_value in possible_values:
                # TODO: Copy the list, pop and buld recursively this thing.
                data_set_partition = filter(
                    lambda x: x[index] == possible_value, train_data)

                self.sensitive_params_summaries[index][possible_value] = \
                    super(SplitedFairBayesModel, self).train(data_set_partition)

    def predict(self, input):
        # Decide what to model to use:
        # WARNING! THIS ALSO ONLY SUPPORTS ONE SENSITIVE VARIABLE YET !!
        sensitive_index = self.sensitve_param_indexes[0]
        sentivive_value = input[sensitive_index]

        (summaries, global_summary, total_len) = \
            self.sensitive_params_summaries[sensitive_index][sentivive_value]
        probabilities = \
            self.evaluate(input, summaries=summaries,
                          global_summary=global_summary, total_len=total_len)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.iteritems():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel


class BalancedBayesModel(BinaryBayesModel):
    # WARNING, THIS IS A SPECIFIC IMPLEMENTATION FOR THE dataset
    # FOLLOWING THE PAPER Calders10

    def discrimination_measure(
            self, index, discriminated_class, privileged_class,
            positive_label, test_data):
        # An asimteric disctimination measure
        predictions = self.getPredictions(test_data)

        test_global_summary = discrete_summarize_total(test_data)
        # entitnes in the lablel
        results = {}
        total = {}

        possible_values = list(
            set([sample[index] for sample in test_data]))
        for possible_value in possible_values:
            results[possible_value] = 0
            total[possible_value] = 0

        for i in range(0, len(predictions)):
            if predictions[i] == positive_label:
                results[test_data[i][index]] += 1
                total[test_data[i][index]] += 1

        for key, value in test_global_summary[0][index].iteritems():
            results[key] = float(results[key])/value

        # warning, this only supports 2 classes for now:
        # return the pair - (Discrimination score , Total positive labels)
        return (
            results[privileged_class] - results[discriminated_class],
            total[privileged_class] + total[discriminated_class])

    def balance_model(
            self, index, discriminated_class, privileged_class,
            positive_label, negative_label, train_data):
        balance_resutls = []

        total_positive_labels = len(
            filter(lambda x: x[index] == positive_label, train_data))

        (disc, assinged_labels) = \
            self.discrimination_measure(index, discriminated_class,
                                        privileged_class, positive_label,
                                        train_data)

        accuracy = self.test(train_data)[0]
        balance_resutls.append((disc, accuracy))
        # print "%s , %s " % (disc, accuracy)

        while disc > 0:
            if assinged_labels > total_positive_labels:
                self.summaries[positive_label][0][index][discriminated_class] = \
                    self.summaries[positive_label][0][index][discriminated_class] + \
                    0.01 * self.summaries[negative_label][0][index][privileged_class]

                self.summaries[negative_label][0][index][privileged_class] = \
                    self.summaries[positive_label][0][index][discriminated_class] - \
                    0.01 * self.summaries[negative_label][0][index][privileged_class]
            else:
                self.summaries[negative_label][0][index][privileged_class] = \
                    self.summaries[negative_label][0][index][privileged_class] + \
                    0.01 * self.summaries[positive_label][0][index][discriminated_class]

                self.summaries[positive_label][0][index][privileged_class] = \
                    self.summaries[negative_label][0][index][privileged_class] - \
                    0.01 * self.summaries[positive_label][0][index][discriminated_class]

            accuracy = self.test(train_data)[0]
            balance_resutls.append((disc, accuracy))
            # print "%s , %s " % (disc, accuracy)
            (disc, assinged_labels) = \
                self.discrimination_measure(index, discriminated_class,
                                            privileged_class, positive_label,
                                            train_data)

        return balance_resutls
