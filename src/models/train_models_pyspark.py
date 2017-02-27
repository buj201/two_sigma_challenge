from pyspark import SparkContext,SQLContext
#For grid search optimization of hyperparameters
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator

#For featurizing categorical features and assembling feature vector
from pyspark.ml.feature import OneHotEncoder, VectorAssembler

#Candidate models
from pyspark.ml.regression import LinearRegression, GBTRegressor

if __name__ == "__main__":

    #Create SparkContext
    sc = SparkContext(appName="CitiBikeModel")

    sqlContext = SQLContext(sc)

    train = sqlContext.read.csv("s3://buj201-two-sigma-challenge/data/train.csv.gz", header=True, inferSchema=True)

    test = sqlContext.read.csv("s3://buj201-two-sigma-challenge/data/test.csv.gz", header=True, inferSchema=True)

    # Notes from API:
    #  - elasticNetParam: Elastic net mixing paramater- 0 is ridge, 1 is lasso
    #  - regParam: regularization hyperparameter

    categorical_features = ['Start Station ID',
                            'Gender',
                            'Day of Week',
                            'Hour',
                            'Month',
                            'Age Missing',
                            'User Type',
                            'WT01',
                            'WT08']

    for feature in categorical_features:
        encoder = OneHotEncoder(inputCol=feature, outputCol='{}_index'.format(feature))
        train = encoder.transform(train)
        test = encoder.transform(test)
        train = train.drop(feature)
        test = test.drop(feature)

    #Note target is first column in dataframe
    labelCol = 'Trip Duration'
    features = train.columns[1:]

    assembler = VectorAssembler(inputCols=features, outputCol="features")

    train = assembler.transform(train)
    test = assembler.transform(test)

    for feature in features:
        test = test.drop(feature)
        train = train.drop(feature)

    #Note continuous features already standardized locally.
    EN = LinearRegression(labelCol = labelCol,
                          featuresCol = 'features',
                          fitIntercept=True,
                          standardization=False)

    EN_paramGrid = ParamGridBuilder().addGrid(EN.regParam, [10,1,0.1, 0.01,0.001])\
                                     .addGrid(EN.elasticNetParam, [0.0, 0.5, 1.0])\
                                     .build()

    EN_tvs = TrainValidationSplit(estimator=EN,
                                  estimatorParamMaps=EN_paramGrid,
                                  evaluator=RegressionEvaluator(labelCol=labelCol),
                                  # 80% of the data will be used for training, 20% for validation.
                                  trainRatio=0.8)

    EN_model = EN_tvs.fit(train)

    EN_model.save("s3://buj201-two-sigma-challenge/EN_model")

    GBR = GBTRegressor(labelCol=labelCol, lossType="squared")¶

    GBR_paramGrid = ParamGridBuilder().addGrid(BGR.maxDepth, [2,4,6])\
                                      .addGrid(BGR.maxIter, [50,100,200])\
                                      .addGrid(BGR.stepSize, [0.01,0.1,0.3])\
                                      .build()

    GBR_tvs = TrainValidationSplit(estimator=GBR,
                                   estimatorParamMaps=GBR_paramGrid,
                                   evaluator=RegressionEvaluator(labelCol=labelCol),
                                   # 80% of the data will be used for training, 20% for validation.
                                   trainRatio=0.8)

    GBR_model = GBR_tvs.fit(train)

    GBR_model.save("s3://buj201-two-sigma-challenge/GBR_model")

    #Stop Spark
    sc.stop()
