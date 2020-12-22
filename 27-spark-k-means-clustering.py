from pyspark.mllib.clustering import KMeans
from numpy import array, random
from math import sqrt
from pyspark import SparkConf, SparkContext
from sklearn.preprocessing import scale

conf = SparkConf().setMaster("local").setAppName("SparkKMeans")
sc = SparkContext(conf = conf)

K = 5

def createClusteredData(N, k):
    random.seed(10)
    pointsPerCluster = float(N)/k
    X = []
    for i in range(k):
        incomeCentroid = random.uniform(20000.0, 200000.0)
        ageCentroid = random.uniform(20, 70)

        for i in range(int(pointsPerCluster)):
            X.append([random.normal(incomeCentroid, 10000.0), random.normal(ageCentroid, 20.0)])
    return array(X)

random.seed(0)

# Load and normalize data
data = sc.parallelize(scale(createClusteredData(100, K)))

# build the model
cluster = KMeans.train(data, K, maxIterations=10, initializationMode="random")
resultRDD = data.map(lambda point: cluster.predict(point)).cache()

counts = resultRDD.countByValue()
results = resultRDD.collect()

print("Counts by value: %s" % counts)
print(results)

## Evaluate

def error(point):
    center = cluster.centers[cluster.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within set sum of squared error = %s" % str(WSSE))

# Counts by value: defaultdict(<class 'int'>, {1: 32, 2: 37, 3: 12, 4: 18, 0: 1})
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 3, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1, 1, 3, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1]
# Within set sum of squared error = 61.15175329914654
