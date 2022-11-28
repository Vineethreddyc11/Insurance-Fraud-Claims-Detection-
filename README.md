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

- Dependent variable: Exploratory data analysis was conducted starting with the dependent variable, Fraud_reported. There were 247 frauds and 753 non-frauds. 24.7% of the data were frauds while 75.3% were non-fraudulent claims.


<img width="513" alt="Screen Shot 2022-11-28 at 12 15 21 PM" src="https://user-images.githubusercontent.com/68578215/204372428-984d2100-7e39-4025-b35d-2a095a94e0a2.png">


- Correlations among variables: Heatmap was plotted for variables with at least 0.3 Pearson’s correlation coefficient, including the DV. Month as customer and age had a correlation of 0.92. Probably because drivers buy auto insurance when they own a car and this time measure only increases with age. Apart from that, there don’t seem to be many correlations in the data. There don’t seem to be multicollinearity problems except maybe that all the claims are all correlated, and somehow total claims have accounted for them. However, the other claims provide some granularity that will not otherwise be captured by total claims. Thus, these variables were kept.

 - Visualizing variables: The value of fraud reported differs across hobbies of the customer. It seems like chess players and crossfitters have higher tendencies to fraud.

<img width="752" alt="Screen Shot 2022-11-28 at 12 15 29 PM" src="https://user-images.githubusercontent.com/68578215/204372423-4db56fd7-f0c8-4c00-b8e7-faaec06d2c07.png">


- Major incidents severity seems to have the highest fraud cases that exceeds non fraud cases.

<img width="546" alt="Screen Shot 2022-11-28 at 12 15 35 PM" src="https://user-images.githubusercontent.com/68578215/204372421-4a294e75-66e0-4016-adab-6da86e210d6f.png">

- The total_claim_amount is high in Saab and Subaru auto_make.

<img width="722" alt="Screen Shot 2022-11-28 at 12 15 41 PM" src="https://user-images.githubusercontent.com/68578215/204372417-360ce6ec-3c12-45c0-9fbb-ac6e15418d0d.png">

- The Injury_claim found highest in the Nissan.

<img width="770" alt="Screen Shot 2022-11-28 at 12 15 52 PM" src="https://user-images.githubusercontent.com/68578215/204372411-108df6e6-4374-464a-b1af-f085fc0e71c7.png">


