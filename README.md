# Lab 6 – Spark ML Pipeline on Amazon EMR

## Overview
This project demonstrates the deployment and execution of a distributed machine learning pipeline
using Apache Spark on Amazon EMR. The task focuses on customer churn prediction and includes
cluster setup, data ingestion, feature engineering, model training, experimentation, and evaluation
in a cloud-based distributed environment.

## Technologies Used
- Amazon EMR 7.12.0
- Apache Spark 3.5.6
- Apache Hadoop 3.4.1
- PySpark ML
- YARN
- Python 3

## Cluster Configuration
- 1 Primary node, 2 Core nodes
- Instance type: m4.large
- Operating system: Amazon Linux 2023
- Installed applications: Hadoop, Spark

## Dataset
The Bank Customer Churn dataset (`Churn_Modelling.csv`) contains 10,000 records.
The target variable is `Exited`, which indicates whether a customer left the bank.
The dataset was uploaded to the EMR cluster and stored in HDFS for distributed processing.

## Spark ML Pipeline
The implemented Spark ML pipeline includes:
- Categorical feature encoding using StringIndexer and OneHotEncoder
- Feature assembly with VectorAssembler
- Feature scaling with StandardScaler
- Logistic Regression classifier
- Random Forest classifier

## Experiments
A feature ablation experiment was conducted by removing categorical features
(Geography and Gender) from the Logistic Regression pipeline.
This experiment was used to analyze the impact of categorical variables on model performance.

## Results
- Logistic Regression (full features): Test accuracy = 0.7887
- Logistic Regression (numeric only): Test accuracy = 0.7741
- Random Forest (full features): Test accuracy = 0.8459

Random Forest achieved the highest accuracy, demonstrating the benefit of non-linear models
for customer churn prediction.

## Execution
The Spark job was executed on YARN using the following command:

```bash
spark-submit --master yarn --deploy-mode client churn_pipeline.py
