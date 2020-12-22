from __future__ import print_function
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

if __name__ == "__main__":
    # create spark session (note, the config section is only for windows
    # .config("spark.sql.warehouse.dir", file:///C:/temp")
    spark = SparkSession.builder.appName("LinearRegression").getOrCreate()

    # load up out data and convert it to format MLLib expects
    inputLines = spark.sparkContext.textFile("data/regression.txt")
    data = inputLines.map(lambda x: x.split(",")).map(lambda x: (float(x[0]), Vectors.dense(float(x[1]))))

    # convert this RDD to a DataFrame
    # Can be avoided when importing data from a real database or using structured streaming.
    colNames = ["label", "features"]
    df = data.toDF(colNames)

    # split our data into training data and testing data
    trainTest = df.randomSplit([0.5, 0.5])
    trainingDF = trainTest[0]
    testDF = trainTest[1]

    # create linear regression model
    lir = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # train the model
    model = lir.fit(trainingDF)

    # predictions
    fullPredictions = model.transform(testDF).cache()
    predictions = fullPredictions.select("prediction").rdd.map(lambda x: x[0])
    labels = fullPredictions.select("label").rdd.map(lambda x: x[0])
    predictionAndLabel = predictions.zip(labels).collect()

    for prediction in predictionAndLabel:
        print(prediction)

    # stop the session
    spark.stop()
