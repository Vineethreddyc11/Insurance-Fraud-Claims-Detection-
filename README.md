# Insurance-Fraud-Claims-Detection-

- Built machine learning model for enabling loss control units to achieve high coverage with low false positive rates to detect whether claim is genuine, or fraudulent based on data from an auto insurance company that has 40+ features and deployed using clouderizer. 

- Performed data manipulation, data preparation, normalization, predictive modeling, and EDA and implemented classification algorithms such as SVM, KNN, Decision tree, Random Forest, Gradient Boost, Ada Boost, XgBoost, SGM, LGBM classifiers.


## Problem Definition

The goal of this project is to build a model that can detect auto insurance fraud. The challenge behind fraud detection in machine learning is that frauds are far less common as compared to legit insurance claims.

Insurance fraud detection is a challenging problem, given the variety of fraud patterns and relatively small ratio of known frauds in typical samples. While building detection models, the savings from loss prevention needs to be balanced with the cost of false alerts. Machine learning techniques allow for improving predictive accuracy, enabling loss control units to achieve higher coverage with low false positive rates.

Insurance frauds cover the range of improper activities which an individual may commit in order to achieve a favourable outcome from the insurance company. This could range from staging the incident, misrepresenting the situation including the relevant actors and the cause of incident and finally the extent of damage caused.

## Data Analysis

In this project, I used dataset which has the details of the insurance policy along with the customer details. It also has the details of the accident on the basis of which the claims have been made.

The given dataset contains 40 columns. The column names like policy number, policy bind date, policy annual premium, incident severity, incident location, auto model, etc.

The obvious con of this data set is the small sample size. However, there are still many companies who do not have big data sets. The ability to work with what is available is crucial for any company looking to transition into leveraging data science.


## Exploratory data analysis

- **Dependent variable:** Exploratory data analysis was conducted starting with the dependent variable, Fraud_reported. There were 247 frauds and 753 non-frauds. 24.7% of the data were frauds while 75.3% were non-fraudulent claims.


<img width="513" alt="Screen Shot 2022-11-28 at 12 15 21 PM" src="https://user-images.githubusercontent.com/68578215/204372428-984d2100-7e39-4025-b35d-2a095a94e0a2.png">


- **Correlations among variables:** Heatmap was plotted for variables with at least 0.3 Pearson’s correlation coefficient, including the DV. Month as customer and age had a correlation of 0.92. Probably because drivers buy auto insurance when they own a car and this time measure only increases with age. Apart from that, there don’t seem to be many correlations in the data. There don’t seem to be multicollinearity problems except maybe that all the claims are all correlated, and somehow total claims have accounted for them. However, the other claims provide some granularity that will not otherwise be captured by total claims. Thus, these variables were kept.

 - **Visualizing variables:** The value of fraud reported differs across hobbies of the customer. It seems like chess players and crossfitters have higher tendencies to fraud.

<img width="752" alt="Screen Shot 2022-11-28 at 12 15 29 PM" src="https://user-images.githubusercontent.com/68578215/204372423-4db56fd7-f0c8-4c00-b8e7-faaec06d2c07.png">


- Major incidents severity seems to have the highest fraud cases that exceeds non fraud cases.

<img width="546" alt="Screen Shot 2022-11-28 at 12 15 35 PM" src="https://user-images.githubusercontent.com/68578215/204372421-4a294e75-66e0-4016-adab-6da86e210d6f.png">

- The total_claim_amount is high in Saab and Subaru auto_make.

<img width="722" alt="Screen Shot 2022-11-28 at 12 15 41 PM" src="https://user-images.githubusercontent.com/68578215/204372417-360ce6ec-3c12-45c0-9fbb-ac6e15418d0d.png">

- The Injury_claim found highest in the Nissan.

<img width="770" alt="Screen Shot 2022-11-28 at 12 15 52 PM" src="https://user-images.githubusercontent.com/68578215/204372411-108df6e6-4374-464a-b1af-f085fc0e71c7.png">

## Pre-processing Pipeline

Data preprocessing is a predominant step in machine learning to yield highly accurate and insightful results. Greater the quality of data, the greater is the reliability of the produced results. Incomplete, noisy, and inconsistent data are the inherent nature of real-world datasets. Data preprocessing helps in increasing the quality of data by filling in missing incomplete data, smoothing noise, and resolving inconsistencies.

- **Incomplete data** can occur due to many reasons. Appropriate data may not be persisted due to a misunderstanding, or because of instrument defects and malfunctions.

- **Noisy data** can occur for a number of reasons (having incorrect feature values). The instruments used for the data collection might be faulty. Data entry may contain human or instrument errors. Data transmission errors might occur as well.


There are many stages involved in data preprocessing.

- **Data cleaning** attempts to impute missing values, removing outliers from the dataset.

- **Data integration** integrates data from a multitude of sources into a single data warehouse.

- **Data transformation** such as normalization, may be applied. For example, normalization may improve the accuracy and efficiency of mining algorithms involving distance measurement.

- **Data reduction** can reduce the data size by dropping out redundant features. Feature selection and feature extraction techniques can be used.

### Treating null values

Sometimes there are certain columns which contain the null value used to indicate missing or unknown values or maybe the value doesn’t exist. In our dataset the null values are present in columns collision_type, property_damage, police_report_available, and _c39 with 178, 360, 343 and 1000 number of null values.

There are different ways of replacing null values from the dataset, but wi usED fillna to replace the null values from data.

### Converting labels into numeric


In machine learning, we usually deal with datasets which contain multiple labels in one or more than one column. These labels can be in the form of words or numbers. To make the data understandable or in human readable form, the training data is often labelled in words.

In our data there are columns with categorical values. The columns like incident_severity, incident_state, incident_type, insured_hobbies, authorities_contacted, incident_city, police_report_available, auto_make, collision_type, auto_model, insured_occupation, insured_education_level, property_damage, insured_relationship, policy_state, insured_sex, fraud_reported. These columns have to be treated with one hot encoding or the label encoder. The target variable fraud_reported has to convert by using label encoder only.

**Label Encoder** refers to converting the labels into numeric form so as to convert it into the machine readable form. Machine learning algorithms can then decide in a better way on how those labels must be operated. It is an important preprocessing step for the structured dataset in supervised learning.

Label encoding in python can be imported from the Sklearn library. Sklearn provides a very efficient tool for encoding. Label encoders encode labels with a value between 0 and n_classes-1.

**Outliers** are data points that are distant from other similar points. They may be due to variability in the measurement or may indicate experimental errors. If possible, outliers should be excluded from the data set. However, detecting that anomalous instance might be very difficult, and is not always possible.

Methods to remove outliers:

- **Z-score** — Call scipy.stats.zscore() with the given data-frame as its argument to get a numpy array containing the z-score of each value in a dataframe. Call numpy.abs() with the previous result to convert each element in the dataframe to its absolute value. Use the syntax (array < 3).all(axis=1) with array as the previous result to create a boolean array.


- **Interquartile range** — The interquartile range can be used to detect the outliers present in the dataframe.

Calculate the interquartile range for the data by using scipy.stats.iqr module.

Multiply the interquartile range by 1.5.

Add 1.5 x interquartile range to the third quartile. Any number greater than this is a suspected outlier.

Subtract 1.5 x interquartile range from the first quartile. Any number lesser than this is a suspected outlier.

### Balancing imbalanced data

There are different algorithms present to balance the target variable. I used SMOTE() algorithm to make our data balance.

SMOTE algorithm works in 4 simple steps:

1. Choose a minority class as input vector.

2. Find its k-nearest neighbors.

3. Choose one of these neighbors and place a synthetic point anywhere on the line joining the point under consideration and its chosen neighbors.

4. Repeat the step until the data is balanced.

## Building machine learning models
For building machine learning models there are several models present inside the Sklearn module.

Sklearn provides two types of models i.e. regression and classification. Our dataset’s target variable is to predict whether fraud is reported or not. So for this kind of problem we use classification models.

But before fitting our dataset to its model first we have to separate the predictor variable and the target variable, then we pass this variable to the train_test_split method to create a random test and train subset.

I have selected 12 models:

- Support Vector Classifier
- Knn
- Decision Tree Classifier
- Random Forest Classifier
- Ada Boost Classifier
- Gradient Boosting Classifier
- Stochastic Gradient Boosting (SGB)
- XgBoost
- Cat Boost Classifier
- Extra Trees Classifier
- LGBM Classifier
- Voting Classifier
