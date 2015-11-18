from bayes_utils import *


class BinaryBayesModel(object):
    summaries = None
    global_summary = None
    total_len = 0

    def train(self, train_data):
        self.summaries = discrete_summarize_by_class(train_data)
        self.global_summary = discrete_summarize_total(train_data)
        self.total_len = len(train_data)

    def evaluate(self, input):
        probabilities = {}
        for classValue, classSummaries in self.summaries.iteritems():
            summary = classSummaries[0]
            total = classSummaries[1]
            probabilities[classValue] = float(1)

            for i in range(len(summary)):
                x = input[i]
                p_i_class = \
                    float(summary[i].get(x, 0))/float(total) * \
                    (float(self.global_summary[0][i].get(x, 0))/float(self.total_len))
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

    def getPredictions(self,testSet):
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



class BalancedFairBayesModel(BinaryBayesModel):

    def discrimination_measure(self, index, label, test_data):
        predictions = self.getPredictions(test_data)
        test_global_summary = discrete_summarize_total(test_data)
        # entitnes in the lablel
        results = {}

        for key in self.global_summary[0][index].keys():
            results[key] = 0

        for i in range(0, len(predictions)):
            if predictions[i] == label:
                results[test_data[i][index]] += 1

        for key, value in test_global_summary[0][index].iteritems():
            results[key] = float(results[key])/value

        # warning, this only supports 2 classes for now:

        values = results.values()

        return max(values) - min(values)
