from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

conf = SparkConf().setMaster("local").setAppName("SparkTFIDF")
sc = SparkContext(conf = conf)

# Load documents one per line
rawData = sc.textFile("data/subset-small.tsv")
fields = rawData.map(lambda x: x.split("\t"))
documents = fields.map(lambda x: x[3].split(" "))
documentNames = fields.map(lambda x: x[1])

#  hash the words in each document to their term frequencies
hashingTF = HashingTF(100000)  #100k hash buckets to save memory
tf = hashingTF.transform(documents)

# Ath this point we have an RDD of sparse vectors representing each document,
# where each value maps to the term frequency of each unique hash value
# Lets comput the TF*IDF of each term in each document

tf.cache()
idf = IDF(minDocFreq=2).fit(tf)
tfidf = idf.transform(tf)

# Now we have an RDD of sparse vectors, where each value is the TFxIDF
# of each unique hash value for each document.

gettysburgTF = hashingTF.transform(["Gettysburg"])
gettysburgHashValue = int(gettysburgTF.indices[0])
gettysburgRelevance = tfidf.map(lambda x: x[gettysburgHashValue])
zippedResults = gettysburgRelevance.zip(documentNames)

print("Best document for Gettysburg is: ")
print(zippedResults.max())

# Best document for Gettysburg is:
# (11.04492083639066, 'Aircraft')
