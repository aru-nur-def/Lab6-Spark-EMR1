from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time

def run_lr_pipeline(data, use_categorical=True):
    # Categorical columns
    geo_indexer = StringIndexer(inputCol="Geography", outputCol="GeographyIndex", handleInvalid="keep")
    gender_indexer = StringIndexer(inputCol="Gender", outputCol="GenderIndex", handleInvalid="keep")

    encoder = OneHotEncoder(
        inputCols=["GeographyIndex", "GenderIndex"],
        outputCols=["GeographyVec", "GenderVec"]
    )

    # Numerical columns (as required in the lab)
    numeric_cols = [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "EstimatedSalary"
    ]

    # Features list with/without categorical ablation
    if use_categorical:
        input_cols = numeric_cols + ["GeographyVec", "GenderVec"]
        stages = [geo_indexer, gender_indexer, encoder]
    else:
        input_cols = numeric_cols
        stages = []

    assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

    lr = LogisticRegression(labelCol="Exited", featuresCol="scaledFeatures")

    stages += [assembler, scaler, lr]
    pipeline = Pipeline(stages=stages)

    start = time.time()
    model = pipeline.fit(data)
    predictions = model.transform(data)
    elapsed = time.time() - start

    evaluator = MulticlassClassificationEvaluator(
        labelCol="Exited",
        predictionCol="prediction",
        metricName="accuracy"
    )
    acc = evaluator.evaluate(predictions)
    return acc, elapsed, predictions

def run_rf_pipeline(data, use_categorical=True):
    geo_indexer = StringIndexer(inputCol="Geography", outputCol="GeographyIndex", handleInvalid="keep")
    gender_indexer = StringIndexer(inputCol="Gender", outputCol="GenderIndex", handleInvalid="keep")
    encoder = OneHotEncoder(
        inputCols=["GeographyIndex", "GenderIndex"],
        outputCols=["GeographyVec", "GenderVec"]
    )

    numeric_cols = [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "EstimatedSalary"
    ]

    if use_categorical:
        input_cols = numeric_cols + ["GeographyVec", "GenderVec"]
        stages = [geo_indexer, gender_indexer, encoder]
    else:
        input_cols = numeric_cols
        stages = []

    assembler = VectorAssembler(inputCols=input_cols, outputCol="features")

    # RF does not require scaling (optional), keep simple
    rf = RandomForestClassifier(labelCol="Exited", featuresCol="features", numTrees=50, maxDepth=8)

    stages += [assembler, rf]
    pipeline = Pipeline(stages=stages)

    start = time.time()
    model = pipeline.fit(data)
    predictions = model.transform(data)
    elapsed = time.time() - start

    evaluator = MulticlassClassificationEvaluator(
        labelCol="Exited",
        predictionCol="prediction",
        metricName="accuracy"
    )
    acc = evaluator.evaluate(predictions)
    return acc, elapsed, predictions


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CustomerChurnPipelineEMR").getOrCreate()

    data = spark.read.csv(
        "hdfs:///user/hadoop/churn_input/Churn_Modelling.csv",
        header=True,
        inferSchema=True
    )

    cols_needed = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "EstimatedSalary", "Exited"
    ]
    data = data.select(*cols_needed).na.drop()

    print("Rows:", data.count())
    data.groupBy("Exited").count().show()

 
    train, test = data.randomSplit([0.8, 0.2], seed=42)

    acc_lr_full, t_lr_full, pred_lr_full = run_lr_pipeline(train, use_categorical=True)
    pred_lr_full_test = run_lr_pipeline(test, use_categorical=True)[2]

    evaluator = MulticlassClassificationEvaluator(
        labelCol="Exited",
        predictionCol="prediction",
        metricName="accuracy"
    )
    acc_lr_full_test = evaluator.evaluate(pred_lr_full_test)

    print(f"[LR FULL] Train accuracy: {acc_lr_full:.4f}, Train time: {t_lr_full:.2f}s")
    print(f"[LR FULL] Test  accuracy: {acc_lr_full_test:.4f}")

    pred_lr_full_test.select("Exited", "prediction", "probability").show(10, truncate=False)

    acc_lr_num, t_lr_num, _ = run_lr_pipeline(train, use_categorical=False)
    pred_lr_num_test = run_lr_pipeline(test, use_categorical=False)[2]
    acc_lr_num_test = evaluator.evaluate(pred_lr_num_test)

    print(f"[LR NUMERIC ONLY] Train accuracy: {acc_lr_num:.4f}, Train time: {t_lr_num:.2f}s")
    print(f"[LR NUMERIC ONLY] Test  accuracy: {acc_lr_num_test:.4f}")

    acc_rf_full, t_rf_full, _ = run_rf_pipeline(train, use_categorical=True)
    pred_rf_full_test = run_rf_pipeline(test, use_categorical=True)[2]
    acc_rf_full_test = evaluator.evaluate(pred_rf_full_test)

    print(f"[RF FULL] Train accuracy: {acc_rf_full:.4f}, Train time: {t_rf_full:.2f}s")
    print(f"[RF FULL] Test  accuracy: {acc_rf_full_test:.4f}")

    spark.stop()
