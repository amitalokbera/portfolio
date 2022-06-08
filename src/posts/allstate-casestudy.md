---
title: Allstate Claims Severity - How severe is an insurance claim?
slug: "/blog/allstate-casestudy"
date: "2022-05-29"
description: Allstate Claims Severity - How severe is an insurance claim?
---

When you’ve been devastated by a serious car accident, your focus is on the things that matter the most: family, friends, and other loved ones. Pushing paper with your insurance agent is the last place you want your time or mental energy spent. This is why Allstate, a personal insurer in the United States, is continually seeking fresh ideas to improve its claims service for the over 16 million households they protect.

![](https://miro.medium.com/max/1400/1*NSyDEf1Yi3cNsp60GcsiNA.jpeg)

**Contents**

-   Introduction
-   Business Problem
-   ML Formulation of the business problem
-   Business Constraint
-   Dataset
-   Performance Metrics
-   EDA and Preprocessing
-   Encoding of the data
-   Modeling part
-   Kaggle Private LB submission
-   Future Section
-   Reference

**Introduction**

An insurance policy/plan is a contract between an individual (Policyholder) and an insurance company (Provider). Under the contract, you pay regular amounts of money (as premiums) to the insurer, and they pay you if the sum assured on unfortunate event arises, for example, untimely demise of the life insured, an accident, or damage to a house The Allstate Corporation is an American insurance company headquartered in Northfield Township, Illinois, near Northbrook since 1967. Allstate comes under the Top 5 insurance writers in America. In 2020, 54 percent of all people in the United States were covered by some type of life insurance, according to LIMRA’s 2020 Insurance Barometer Study

**Business Problem**

In the past few years, the U.S insurance industry has increased tremendously, so there is no doubt that many people also claim insurance when some unfortunate event takes place.

-   Identify the predicting cost of the claim, if the higher the claim is, the more severe problem is.

As a company, Allstate could identify if the claim is severe or not, If the claim is severe it could start working on it soon. So that the claim is released as soon as possible to the needed ones.

**ML Formulation of the business problem**

So given anonymized data of Allstate customers, which consist of 116 categorical data and 14 numerical data we want to predict the insurance claim amount.

**Business Constraint**

-   Minimize MAE (Mean Absolute Error)
-   No strict latency constraint
-   Minimize large errors while predicting the Target Value

**Dataset**

Source:  [https://www.kaggle.com/competitions/allstate-claims-severity/data](https://www.kaggle.com/competitions/allstate-claims-severity/data)

The dataset which has been provided to us is completely anonymized and no additional details have been provided to us, like what each column over here signifies to us. All the categorical columns are encoded in alphabetic format and all the numerical columns are normalized between 0 and 1

3 CSV files are provided to us —

_train.csv_

This file contains are the data that we will use to train our machine learning model. It consists of 188K rows with 132 columns. We have 116 categorical columns and 14 numerical columns and 1 target value, which consists of the actual insurance claim value.

_test.csv_

Same as train.csv files, only the major difference is that the target column is missing from here and we need to build an ML model which would predict the missing target column, which later is used to evaluate our model on Allstate Claim Severity Kaggle Competition.

_submission.csv_

Contains the submission format for the Kaggle Competition.

**Performance Metrics**

In this Kaggle Competition, submissions are evaluated on the Mean Absolute Error (MAE) between the predicted target value and the actual target value.

**EDA and Preprocessing**

We will be doing extensive EDA on the training dataset because we need to gain a lot of insight into how our data is and what factors affect our end results.

The dataset which is provided by Allstate doesn't have any Null values.

Looking at the ID column we can say that some of the points are removed from the dataset to create more anonymization. This random removal of the data points can be seen throughout the dataset

Output:

array([ 1,  2,  5, 10, 11, 13, 14, 20, 23, 24, 25, 33, 34, 41, 47, 48, 49, 51, 52, 55], dtype=int64)

Later I categorized all the categorical columns which have only two unique values i.e A and B. After plotting the pie chart on all the two-way categorical columns, we find out that value A has the major dominance in most of the columns.

![](https://miro.medium.com/max/1400/1*TRjckALbQ6GmATx6g44SQw.png)

piechart on column cat1

If we take a look at our target value i.e loss column in the train.csv file

![](https://miro.medium.com/max/1400/1*xzJUfFYhrV1VTS14ZOi0OQ.png)

histogram and pdf on the loss column

We can see that our loss column is very skewed, to overcome this problem we will apply various transformation techniques to our loss column

![](https://miro.medium.com/max/1400/1*8lW45cWJwIgomZL9u30ZOw.png)

min-max transformation on loss

![](https://miro.medium.com/max/1400/1*ea0FLWQAZJ2B30NA4vtGPA.png)

log transformation on loss

![](https://miro.medium.com/max/1400/1*Lg6yQQl2I86eDI3W-lzwlQ.png)

normalization + log transformation on loss

Here we can see that I applying some basic transformation functions to the loss column, the PDF of the loss column follows normal distribution now. This transformation will help our model to better learn the loss column

Applying percentile on the loss column

Output:

0 percentile - 0.0   
10 percentile - 0.5487345983053316   
20 percentile - 0.5763245026028123   
30 percentile - 0.5989953104238187   
40 percentile - 0.6191047741468068   
50 percentile - 0.6384448180750446   
60 percentile - 0.6584707400054535   
70 percentile - 0.6801497993895131   
80 percentile - 0.7051612270539311   
90 percentile - 0.7373587923394795   
100 percentile - 1.0

If we take a look over here we can see that 0–10th percentile took a huge jump from 0.0 to 0.54, but from over there we gradually keep on increasing but for 90–100th percentile we again see that it jumped a lot between 90th and 100th percentile

Zooming in the 90–100th percentile

Output:

90 percentile - 0.7373587923394795   
91 percentile - 0.7414995960777733   
92 percentile - 0.7459409784296457   
93 percentile - 0.7507778033597101   
94 percentile - 0.7564234976101883   
95 percentile - 0.7627779024742124   
96 percentile - 0.770004711840892   
97 percentile - 0.7783437970174782   
98 percentile - 0.7898148423985144   
99 percentile - 0.8071533201770725   
100 percentile - 1.0

Till the 99th percentile it was increasing gradually, but after the 99th percentile, it moved a lot

**NOTE**  — We are working on the log normalized loss variable, not on the original loss variable.

By looking at the percentile we can say that 90% of the claim values are under 0.73. Even till the 99th percentile, our loss value is under 0.85. Between 99th — 100th percentile, our loss variable jumped a lot

If we perform the same analysis on our original loss variable, it also follows the same pattern as above.

Checking for dominant variable covered percentage in a categorical column

We find out that there are 31 columns where the same value is repeated at least 99% of the time. While training our model some of the columns might not provide useful insight, so to reduce computation usage we could drop some of the columns.

While doing further analysis on the two-way categorical column, I have noticed that for column cat57, the mean value of loss value is a bit different for both the class i.e A and B. We could use this column to come up with simple and useful Feature engineer to improve the performance of our model

Output:

#########################   
In cat57 column - A, its mean value is 0.6389277650810136   
In cat57 column - B, its mean value is 0.766728360555299   
#########################

![](https://miro.medium.com/max/764/1*-76-4ZDUrLvCqe34A2Nhsg.png)

PDF on cat57 column

![](https://miro.medium.com/max/744/1*P_uNCpmaD3emUKQyxGpzcQ.png)

Boxplot on cat57 column

Now we are focusing on the k-way categorical column, where the categorical column can have more than 2 unique values.

Output:

In cat77 column, value D has covered 99.5672% of area   
In cat78 column, value B has covered 99.0484% of area

We can drop the cat77 and cat78 columns in the data preprocessing stage as the data in these two columns is very skewed.

Numerical Column correlation heatmap

![](https://miro.medium.com/max/1400/1*KAyTo4uzGSNeqMJafS18CQ.png)

Over here we can see that cont1 and cont9 are highly correlated, same with the cont11 and cont12. There are also many other highly correlated pairs that I have not mentioned over here, while going through the preprocessing stage we need to handle all these pairs also.

Checking skewness of numerical column

Output:

We noticed highed skewness for cont9 i.e 1.072428719811583

![](https://miro.medium.com/max/1400/1*bnjcp0Pe6Je7enKTIXdPiw.png)

hist and PDF for cont9 column

So at the data preprocessing step, we can drop the cont9 from our dataset or we can try to apply various feature transformations to this column.

**Encoding of the Data**

For all the numerical columns we have applied Box Cox Transformation.

For the categorical columns, we have encoded the data using Lexical encoding.

To reduce the training time and computational usage, we have dropped some of the categorical columns where the same value is repeated at least 99.9% of the time.

Output:

['cat15', 'cat22', 'cat55', 'cat56', 'cat62', 'cat63', 'cat64', 'cat68', 'cat70']

We have applied various types of transformation to the loss column to improve the performance of our model

Here is all the decoding function to revert the predicted value to the original loss range.

I have split the training data into two parts i.e train and val data in an 80:20 ratio. To make sure that both the train and val data follow the same loss column distribution, we have used the percentile data to bin the loss data and split it using stratifies binned data.

Checking the PDF of both train and val datasets.

![](https://miro.medium.com/max/1400/1*MQHeL3-VjRuVGWqK06KKIw.png)

**Modeling part**

Here is the list of models which I have tried

-   Linear Regression
-   DecisionTreeRegressor
-   LinearSVR
-   GBDT
-   XGboost
-   Catboost
-   LightGBM

After training all the models and testing the result on val dataset, we found that Gradient Boost Trees-based model gave us the best result. As a result, XGBoost, CatBoost, LightGBM, and GBDT gave us the best result.

So we focus more on these sets of models only.

After digging around a bit on the Kaggle Discussion forum, I found that there’s a Xgboost-based python library- XGBfir, that will use the XGboost model and give us the two-way interaction between the columns, based on which we can create new features.

XGBfir library will save the result in an Excel file. So based on this 2-way feature interaction we could create new features by performing simple operations such as addition, subtraction, or concatenation between two columns.

By default, the XGBoost library will evaluate our model on RMSE, but in this case, we are primarily dealing with the MAE score. So while training the XGBoost we could pass a custom metric to our XGBoost thus it will reduce our XGBoost model to make large prediction errors.

For hyper tuning parameter tuning of all the models, we have opted to go with Optuna. Optuna internally uses Bayesian optimization to find good parameters for our model. There’s also a big advantage of using Optuna compared to Sklearn’s GridSearchCV or RandomSearchCV. We could even run the trial search from the last point where it stopped, which is a huge plus point in the case of Optuna.

Both XGBoost and LightBGM internally support GPU boost up, so this helps us to train our model faster and efficiently.

Here are all the hyperparameter tuned values for all the models.

Rather than using a Mean Average score of all the models, we have opted to combine all the models using Sklearn’s StackingRegressor, where we are passing all the hyper tuned models as the base model to the StackingRegressor and for the second stage, we are using CatBoostRegressor as our final predictor.

**Kaggle Private LB submission**

Using this model on the Kaggle test dataset, resulting in a score of 1120.53 on Public Leaderboard.

Kaggle Leaderboard submission

I have also deployed this model on my local system using Flask API.

Video Demo of the deployment —

**Future Section**

We can improve this model furthermore by doing some aggressive Feature Engineering and also using some Deep Learning models at the base model of StackingRegressor. As most of the winning solutions of this Kaggle competition have used some sort of Deep Learning model.

Thank you for reading! 😄 And if you liked this blog, please hit the clap icon! 👏