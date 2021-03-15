from __future__ import print_function

import sys

from pyspark import SparkContext
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics

def parsePoint(line):
    """
    Parse a line of text into an MLlib LabeledPoint object. When numpy processes data, it only understands binary values.
    """
    values = [float(s) for s in line.split(',')]
    if values[0] == -1:   # Convert -1 labels to 0 for MLlib
        values[0] = 0
    for i in range(1,len(values)):
        if values[i] < 0:
            values[i] = values[i] * -1
    return LabeledPoint(values[0], values[1:]) # [1:] returns all elements in the list except for the first one.


if __name__ == "__main__":

    #sc.stop()
    sc = SparkContext()
    #data = sc.textFile("hdfs://localhost:9000/1000_rows.csv").map(parsePoint)
    data = sc.textFile("file:///root/finaloutputs/2000000_rows.csv").map(parsePoint)
    #print (data.take(5))
    #iterations = 100
    training, test = data.randomSplit([0.6, 0.4]) #, seed=11)
    #training.cache()
    #print (training.count())
    #print (training.take(100))
    import time
    start_time = time.time()
    model = NaiveBayes.train(training) #, iterations=100)
    #print("Final weights: " + str(model.weights))
    #print("Final intercept: " + str(model.intercept))
    predictionAndLabels = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))
    #print (predictionAndLabels.take(400))
    #accuracy = 1.0 * predictionAndLabels.filter(lambda (x, v): x == v).count() / float(test.count())
    metrics = MulticlassMetrics(predictionAndLabels)
    precision = metrics.precision(1.0)
    recall = metrics.recall(1.0)
    f1Score = metrics.fMeasure(1.0)
    accuracy = metrics.accuracy
    print("Summary Stats")
    print("Accuracy = %s" % accuracy)
    print("Precision = %s" % precision)
    print("Recall = %s" % recall)
    print("F1 Score = %s" % f1Score)
    #print('model accuracy {}'.format(accuracy))
    print("--- %s seconds ---" % (time.time() - start_time))
    sc.stop()
