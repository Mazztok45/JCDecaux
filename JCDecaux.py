
# coding: utf-8

# In[1]:


from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkConf
conf = SparkConf()
sc = SparkContext('local', conf)
spark = SparkSession(sc)


# In[2]:


import pyspark.sql.functions as F
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.clustering import KMeans


# In[3]:


spark.read.json('C:/Users/maxen/Downloads/Test_DS/Test_DS/Brisbane_CityBike.json').show(150,False)


# In[42]:


df = spark.read.json('C:/Users/maxen/Downloads/Test_DS/Test_DS/Brisbane_CityBike.json').select('address').withColumn('newadress', 
            F.regexp_replace(F.regexp_replace(F.regexp_replace(F.upper(F.col('address')), 'ST', ''), 'RD', ''), '/', ''))


# In[43]:


tokenizer = Tokenizer(inputCol="newadress", outputCol="words")
wordsData = tokenizer.transform(df)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
featurizedData = hashingTF.transform(wordsData)
# alternatively, CountVectorizer can also be used to get term frequency vectors

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)


# In[44]:


kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(rescaledData.select('features'))


# In[45]:


model.transform((rescaledData.select('features'))).groupby('prediction').count().show()


# In[47]:


model.transform((rescaledData.select('features'))).write.json('C:/Users/maxen/Downloads/result.json')

