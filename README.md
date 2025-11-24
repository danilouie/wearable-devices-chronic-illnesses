# DSCI 510 Final Project

## Predicting Chronic Diseases from Personal Wearable Devices

This repository contains the code for Danielle Louie's final project for DSCI 510 at the University of Southern California.

---

## Table of Contents

1. [About](#about)
2. [Data Sources](#data-sources)
3. [Results](#results)
4. [Improvements](#improvements)
5. [Installation](#installation)
6. [Running Analysis](#running-analysis)

---

### About

The project aims to analyze how machine learning algorithms, specifically RandomForestClassifier, can help to flag potential chronic diseases based on health data collected by user-wearable devices. For this project, we focus on data collected by Apple Watches and Fitbits.

### Data sources

[Apple Watch and Fitbit Data](https://www.kaggle.com/datasets/aleespinosa/apple-watch-and-fitbit-data?utm_source=chatgpt.com&select=aw_fb_data.csv)
The `aw_fb_data.csv` file contains 20 features of data collected from either Apple Watches and Fitbits during a range of activities and has a total of 6264 instances. The data was used to as a baseline for physical indicators that a machine learning algorithm could use to predict disease.

[US Chronic Disease Indicators](https://healthdata.gov/CDC/U-S-Chronic-Disease-Indicators/dhcp-wb3k/about_data)
The `U.S._Chronic_Disease_Indicators.xlsx` file, obtained from using a RESTful web API, contains 34 public surveillance indicators and 309215 related to chronic diseases and their risk factors. The data was used to train a RandomForestClassifier in matching what types of chronic diseases may be linked to diseases that can be identified better through physical indicators. For this project, we will only be using 271694 instances that were stratified by `Sex`, `Race/Ethnicity`, and `Age`.

[Nutrition, Physical Activity, and Obesity Behavioral Risk Factors](https://catalog.data.gov/dataset/nutrition-physical-activity-and-obesity-behavioral-risk-factor-surveillance-system)
The data from `Nutrition_Physical_Activity_and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv` contains 33 public surveillance indicators and 106260 instances related to behavioral risk factors. The data was used to train a RandomForestClassifier in matching what types of chronic diseases may be linked to diseases that can be identified better through physical indicators. For this project, we will only be using 60720 instances that were stratified by `Sex`, `Race/Ethnicity`, and `Age`.

### Results

##### Data Observation

From our EDA analysis, we can see that due to the large number of data in all three data sources, we have a fairly even distribution of `Sex` and `Race/Ethnicity`. The main class imbalance that we accounted for was the `Age_Bin`, as our age bins were of different sizes and it appeared that younger people are more likely to wear an Apple Watch or Fitbit.

##### Result Analysis

From our results, we can observe that the RandomForestClassifier found that people who have a high BMI (and might be experience weight concerns) are more likely to have a chronic disease under the "Nutrition, Physical Activity, and Weight Status"; this seems to make sense as given that we were mainly provided with physical health data from wearable user devices, such as It was also predicted that only males would experience this problem, which likely can be bias in the machine learning algorithm caused by the availability of data that we are working with. 

### Improvements

### Installation

- _describe what API keys, user must set where (in .enve) to be able to run the project._
- _describe what special python packages you have used_

### Running analysis

_update these instructions_

From `src/` directory run:

`python main.py `

Results will appear in `results/` folder. All obtained will be stored in `data/`

to do:

- analyze.py
- main.py
- README.md
- descriptions in results.ipynb
- test.py
- requirements.txt
- slides

done:

- load.py
- process.py
- augment.py
