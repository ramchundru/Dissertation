from __future__ import print_function

import sys

from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.classification import SVMWithSGD, SVMModel

def parsePoint(line):
    """
    Parse a line of text into an MLlib LabeledPoint object.
    """
    values = [float(s) for s in line.split(',')]
    if values[0] == -1:   # Convert -1 labels to 0 for MLlib
        values[0] = 0
    return LabeledPoint(values[0], values[1:])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: logistic_regression <file> <iterations>", file=sys.stderr)
        sys.exit(-1)
    sc = SparkContext(appName="PythonLR")
    points = sc.textFile(sys.argv[1]).map(parsePoint)
    iterations = int(sys.argv[2])
    training, test = points.randomSplit([0.6, 0.4], seed=11)
    training.cache()
    import time
    start_time = time.time()
    model = SVMWithSGD.train(points, iterations)
    #print("Final weights: " + str(model.weights))
    #print("Final intercept: " + str(model.intercept))
    predictionAndLabels = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))
    accuracy = 1.0 * predictionAndLabels.filter(lambda (x, v): x == v).count() / float(test.count())
    metrics = MulticlassMetrics(predictionAndLabels)
    precision = metrics.precision(1.0)
    recall = metrics.recall(1.0)
    f1Score = metrics.fMeasure(1.0)
    print("Summary Stats")
    print("Precision = %s" % precision)
    print("Recall = %s" % recall)
    print("F1 Score = %s" % f1Score)
    print('model accuracy {}'.format(accuracy))
    print("--- %s seconds ---" % (time.time() - start_time))
    sc.stop()
