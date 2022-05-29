---
title: Allstate Claims Severity - How severe is an insurance claim?
slug: "/blog/pix2pix-mnist"
date: "2022-05-29"
description: Allstate Claims Severity - How severe is an insurance claim?
---

When you’ve been devastated by a serious car accident, your focus is on the things that matter the most: family, friends, and other loved ones. Pushing paper with your insurance agent is the last place you want your time or mental energy spent. This is why Allstate, a personal insurer in the United States, is continually seeking fresh ideas to improve their claims service for the over 16 million households they protect.

**Content**

* Introduction

* Business problem

* ML formulation of the business problem

* Business constraints

* Performance Metrics

* Extensive EDA and pre-processing

* Encoding the categorical columns

* Modelling part : Comparison of models, Custom Ensembler classifier

* Predicting for final dataset provided for competition

* Kaggle score in the leaderboard

* Future Section

* References


**Introduction**

An insurance policy/plan is an contact between an individual (Policyholder) and an insurance company (Provider). Under the contract, you pay regular amounts of money (as premiums) to the insurer, and they pay you if the sum assured on unfortunate event arises, for example, untimely demise of the life insured, an accident, or damage to a house The Allstate Corporation is an American insurance company headquartered in Northfield Township, Illinois, near Northbrook since 1967. Allstate comes under Top 5 insurance writer in America. In 2020, 54 percent of all people in the United States were covered by some type of life insurance, according to LIMRA's 2020 Insurance Barometer Study



**Business problem**

In the past few years, U.S insurance industry have increased tremendously, so there is no doubt that many people also claim for insurance when some unfortunate event takes place. 

 - Identify the predicting cost of the claim, if higher the claim is, the more severe problem it is.

As a company, Allstate could identify if the claim is severe or not, If the claim is severe it could start working on it soon. So that the claim is released as soon as possible to the needed one's.




**ML formulation of the business problem** 

We have one CSV file, which contains all the data. Over here the dataset which Allstate has provided to us, have been gone through the data anonymization stage (Data anonymization is a type of information sanitization whose intent is privacy protection. It is the process of removing personally identifiable information from data sets) In the CSV File we have total 132 columns, out of which 1 is index column, 130 is training data and 1 is our target variable. Taking a deeper look at training dataset we found out that, there are total 116 categorical (discrete) valued column and 14 numerical (continuous) valued column. Here we are given the task to predict the Loss column which is our Target variable.

Type of Machine Learning Problem – Regression 

Source - https://www.kaggle.com/c/allstate-claims-severity




**Business constraints**

- Minimize MAE (Mean Absolute Error)

- No strict latency constraint

- Minimize large error while predicting the Target Value 



**Performance Metrics** 

In this Kaggle Competition, submissions are evaluated on the Mean Absolute Error (MAE) between the predicted target value and the actual target value.



**Extensive EDA and pre-processing**


