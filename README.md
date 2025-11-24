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

The project aims to analyze how machine learning algorithms, specifically RandomForestClassifier, can help to flag potential chronic diseases based on health data collected by user-wearable devices. For this project, we focus on data collected by Apple Watches and Fitbits. The project is conducted primarily with Python and Jupyter Notebook.

---

### Data sources

[Apple Watch and Fitbit Data](https://www.kaggle.com/datasets/aleespinosa/apple-watch-and-fitbit-data?utm_source=chatgpt.com&select=aw_fb_data.csv)

The `aw_fb_data.csv` file contains 20 features of data collected from either Apple Watches and Fitbits during a range of activities and has a total of 6264 instances. The data was used to as a baseline for physical indicators that a machine learning algorithm could use to predict disease.

[US Chronic Disease Indicators](https://healthdata.gov/CDC/U-S-Chronic-Disease-Indicators/dhcp-wb3k/about_data)

The `U.S._Chronic_Disease_Indicators.xlsx` file, obtained from using a RESTful web API, contains 34 public surveillance indicators and 309215 instances related to chronic diseases and their risk factors. The data was used to train a RandomForestClassifier in matching what types of chronic diseases may be linked to diseases that can be identified better through physical indicators. For this project, we will only be using 271694 instances that were stratified by `Sex`, `Race/Ethnicity`, and `Age`.

[Nutrition, Physical Activity, and Obesity Behavioral Risk Factors](https://catalog.data.gov/dataset/nutrition-physical-activity-and-obesity-behavioral-risk-factor-surveillance-system)

The data from `Nutrition_Physical_Activity_and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv` contains 33 public surveillance indicators and 106260 instances related to behavioral risk factors. The data was used to train a RandomForestClassifier in matching what types of chronic diseases may be linked to diseases that can be identified better through physical indicators. For this project, we will only be using 60720 instances that were stratified by `Sex`, `Race/Ethnicity`, and `Age`.

---

### Files
- `load.py`: Loads all the data and converts them into usable DataFrames.
- `process.py`: Cleans and engineers the features in each dataset. Creates DataFrames as needed.
- `augment.py`: Uses RandomForestClassifier to create a full DataFrame with predictions of diseases.
- `analyze.py`: Analyzes the final results and produces data visualizations.
- `tests.py`: Unit tests for checking if functions are working as expected.
- `results.ipynb`: A Jupyter Notebook that runs the project from start to finish.
- `main.py`: A Python script that runs the project from start to finish.

---

### Results

#### Data Observation

From our EDA analysis, we can see that due to the large number of data in all three data sources, we have a fairly even distribution of `Sex` and `Race/Ethnicity`. The main class imbalance that we accounted for was the `Age_Bin`, as our age bins were of different sizes and it appeared that younger people are more likely to wear an Apple Watch or Fitbit.

#### Result Analysis

Overall, the machine learning algorithm appeared to have biased, yet also yield insightful results. 

First, we can observe the bias from the final three bar plots created, specifically the last two: **Disease Distribution by Sex** and **Disease Distribution by Age Bin**. In Disease Distribution by Sex, we see that the RandomForestClassifier predicted that only men would experience a chronic disease related to *Nutrition, Physical Activity, and Weight Status* while only women would experience *Arthritis*. It is difficult to believe that in such a large sample size, a disease would be distributed so evenly by sex alone, especially since neither of these diseases are strongly impacted by sex only. However, it is important to note that women are more prone to arthritis, which could partially explain that perhaps, the RandomForestClassifier and the data we provided had more women than men with arthritis. 

Second, in Disease Distribution by Age Bin, we see a huge disparancy between the number of people who fall into the category of 18-44 compared to 45-64. Part of this age distribution can be due to the fact that 18-44 is a wider range of data compared to 45-64; however, this was the only reasonable binning of age due to the age grouping provided from the US Chronic Disease Indicators resource.

Despite these outcomes, we can still draw insight from this project. What was interesting was that the RandomForestClassifier predicted the highest count for *Nutrition, Physical Activity, and Weight Status*, followed by *Arthritis*, then *Asthma*. If we only considered the data that we were testing on, which was **physical health data**, this ordering makes sense. The first two chronic diseases reflect in physical conditions much more than the third. The first condition especially is more correlated to features that we were able to engineer, such as `BMI` and `heart_rate compared` to `target_heart_rate`, which the RandomForestClassifier may have learned when training on the Behavioral Risk Factor and Chronic Disease Indicator data that listed "Obesity / Weight Status" as a disease.

---

### Improvements

1. **More Joint Data**: Collecting datasets that directly link physical indicators from wearables to tracked chronic conditions could greatly strengthen the model.
2. **Alternative Algorithms**: Other algorithms besides Random Forest (e.g., boosting, neural networks) may yield different or more robust results given class imbalance and dataset structure.
3. **Feature Engineering**: Additional features or more nuanced aggregation might improve performance and interpretability.

---

### Installation

**Prerequisites:**
- Python 3.8 or higher
- pip or conda

**Steps:**
1. Clone this repository
    ```python
    git clone <https://github.com/danilouie/wearable-devices-chronic-illnesses.git>
    cd <your-repo-directory>
    ```

2. Setup environment variables
    - You can use `env.example` as a reference.

3. Install dependencies
    ```python
    pip install -r requirements.txt
    ```

4. Download Data 
    - Download data from the links in the [Data Sources](#data-sources) section. No credentials or API keys are neded. Datasets are either included or downloaded at runtime, using public API endpoints as referenced in code and comments.

*Note: Feel free to set up your own [venv](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).*

---

### Running analysis

From `src/` directory run:
- `python main.py`: Results will appear in `results/` folder. All obtained will be stored in `data/`.
- `results.ipynb`: Results are printed chronologically in the cells. Plots are shown as well.