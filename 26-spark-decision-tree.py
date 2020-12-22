from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark import SparkConf, SparkContext
from numpy import array

conf = SparkConf().setMaster("local").setAppName("SparkDecisionTree")
sc = SparkContext(conf = conf)

def binary(YN):
    return 1 if YN == 'Y' else 0

def mapEducation(degree):
    switch = {
        'BS': 1,
        'MS': 2,
        'PhD': 3
    }

    return switch.get(degree, 0)

def createLabeledPoints(fields):
    yearsExperience = int(fields[0])
    employed = binary(fields[1])
    previousEmployers = int(fields[2])
    educationLevel = mapEducation(fields[3])
    topTier = binary(fields[4])
    interned = binary(fields[5])
    hired = binary(fields[6])

    return LabeledPoint(hired, array([yearsExperience, employed,
        previousEmployers, educationLevel, topTier, interned]))

# load csv file and filter out the header
rawData = sc.textFile("data/PastHires.csv")
header = rawData.first()
rawData = rawData.filter(lambda x: x != header)

# split each line into a list based on comma delimiters
csvData = rawData.map(lambda x: x.split(","))

# convert to LabeledPoints
trainingData = csvData.map(createLabeledPoints)

# create a test candidate
testCandidates = [array([10, 1, 3, 1, 0, 0])]
testData = sc.parallelize(testCandidates)

# Train our DecisionTree classifier using our data set
model = DecisionTree.trainClassifier(trainingData, numClasses=2,
                categoricalFeaturesInfo={1:2, 3:4, 4:2, 5: 2},
                impurity='gini', maxDepth=5, maxBins=32)

# predictions
predictions = model.predict(testData)
print('Hire prediction:')

results = predictions.collect()
for result in results:
    print(result)

# Print out decision tree
print('Learned classification tree model:')
print(model.toDebugString())

# Hire prediction:
# 1.0
# Learned classification tree model:
# DecisionTreeModel classifier of depth 4 with 9 nodes
  # If (feature 1 in {0.0})
   # If (feature 5 in {0.0})
    # If (feature 0 <= 0.5)
     # If (feature 3 in {1.0})
      # Predict: 0.0
     # Else (feature 3 not in {1.0})
      # Predict: 1.0
    # Else (feature 0 > 0.5)
     # Predict: 0.0
   # Else (feature 5 not in {0.0})
    # Predict: 1.0
  # Else (feature 1 not in {0.0})
   # Predict: 1.0
