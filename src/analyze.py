import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --- 1. CONDUCT EDA
def analyze_aw_fb_data(aw_fb_df, save_dir=None):
    """
    Plot 

    Args:
        aw_fb_df: A DataFrame with data collected by Apple Watches and Fitbits.

    Returns:
        One stacked bar plot visually portraying the distribution of three different types of data:
            1. Apple Watch vs Fitbit
            2. Age Bins
            3. Sex
    """
    try:
        grouped = (
            aw_fb_df.groupby(['Age_Bin', 'Device', 'Sex'])
            .size()
            .unstack('Sex', fill_value=0)
            .reset_index()
        )

        age_bins = grouped['Age_Bin'].unique()
        devices = grouped['Device'].unique()
        sexes = [col for col in grouped.columns if col not in ['Age_Bin', 'Device']]

        bar_width = 0.35
        gap = 0.25
        x = np.arange(len(age_bins)) * (len(devices) * bar_width + gap)

        fig, ax = plt.subplots(figsize=(12, 8))

        for idx, device in enumerate(devices):
            df_device = grouped[grouped['Device'] == device]
            positions = x + idx * bar_width

            bottom = np.zeros(len(age_bins))
            for sex in sexes:
                counts = df_device[sex].values
                ax.bar(
                    positions,
                    counts,
                    bar_width,
                    label=f"{device}, {sex}",
                    bottom=bottom
                )
                bottom += counts

        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(age_bins, rotation=45)
        ax.set_xlabel("Age Bin")
        ax.set_ylabel("Count")
        ax.set_title("Device Count by Age Bin and Device, Stacked by Sex", fontsize=18)
        ax.legend(title="Device/Sex", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "aw_fb_analysis.png"))
            plt.close()

        print("EDA for aw_fb plotted!")
        
    except Exception as e:
        print(f"Unable to visualize aw_fb_data: {e}")


def analyze_chronic_data(chronic_age_df, chronic_race_df, chronic_sex_df, save_dir=None):
    """
    Plot the distribution of each split chronic DataFrame.

    Args:
        chronic_age_df: A DataFrame with chronic data stratified by age.
        chronic_race_df: A DataFrame with chronic data stratified by race.
        chronic_sex_df: A DataFrame with chronic data stratified by sex.

    Returns:
        Three bar plots showing the distribution of each dataframe.
    """
    try:
        # Chronic age counts
        age_counts = chronic_age_df['age_bin'].value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        age_counts.plot(kind='bar')
        plt.title("Chronic Data: Age Bin Counts", fontsize=18)
        plt.xlabel("Age Bin")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "chronic_age_analysis.png"))
            plt.close()

        # Chronic race counts
        race_counts = chronic_race_df['Race/Ethnicity'].value_counts()
        plt.figure(figsize=(12, 7))
        race_counts.plot(kind='bar')
        plt.title("Chronic Data: Race/Ethnicity Counts", fontsize=18)
        plt.xlabel("Race/Ethnicity")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "chronic_race_analysis.png"))
            plt.close()
                
        # Chronic sex counts
        sex_counts = chronic_sex_df['Sex'].value_counts()
        plt.figure(figsize=(7, 5))
        sex_counts.plot(kind='bar')
        plt.title("Chronic Data: Sex Counts", fontsize=18)
        plt.xlabel("Sex")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "chronic_sex_analysis.png"))
            plt.close()

        print("Three bar plots created!")
    
    except Exception as e:
        print(f"Unable to visualize chronic data: {e}")


def analyze_nutri_data(nutri_sex_df, nutri_age_df, nutri_race_df, save_dir=None):
    """
    Plot the distribution of each split nutri DataFrame.

    Args:
        nutri_sex_df: A DataFrame with chronic data stratified by sex.
        nutri_age_df: A DataFrame with chronic data stratified by age.
        nutri_race_df: A DataFrame with chronic data stratified by race.

    Returns:
        Three bar plots showing the distribution of each dataframe.
    """
    try:
        # Nutri sex counts
        sex_counts = nutri_sex_df['Sex'].value_counts()
        plt.figure(figsize=(7, 5))
        sex_counts.plot(kind='bar')
        plt.title("Nutri Data: Sex Counts", fontsize=18)
        plt.xlabel("Sex")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "nutri_sex_analysis.png"))
            plt.close()

        # Nutri age counts
        age_counts = nutri_age_df['age_bin'].value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        age_counts.plot(kind='bar')
        plt.title("Nutri Data: Age Bin Counts", fontsize=18)
        plt.xlabel("Age Bin")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "nutri_age_analysis.png"))
            plt.close()

        # Nutri race counts
        race_counts = nutri_race_df['Race/Ethnicity'].value_counts()
        plt.figure(figsize=(12, 7))
        race_counts.plot(kind='bar')
        plt.title("Nutri Data: Race/Ethnicity Counts", fontsize=18)
        plt.xlabel("Race/Ethnicity")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "nutri_race_analysis.png"))
            plt.close()

        print("Three bar plots created!")
    
    except Exception as e:
        print(f"Unable to visualize nutri data: {e}")


# --- 2. ANALYZE RESULTS
def analyze_assigned_diseases(full_df) -> tuple:
    """
    Analyzes disease assignments and their relationship to Sex and Age_Bin.

    Args:
        full_df: DataFrame with 'Assigned_Disease', 'Sex', 'Age_Bin' columns.

    Returns:
        tuple: Contains three outputs:
            - disease_counts: Series with disease assignment counts.
            - disease_sex: DataFrame of disease by Sex.
            - disease_age: DataFrame of disease by Age_Bin.
    """
    try:
        full_df['Assigned_Disease'] = full_df['Assigned_Disease'].replace({'Nutrition, Physical Activity, and Weight Status': 'NPW'})

        print("Counting number of each disease...")
        disease_counts = full_df['Assigned_Disease'].value_counts(dropna=True)

        print("Analyzing disease by sex...")
        disease_sex = full_df.pivot_table(index='Assigned_Disease', columns='Sex', aggfunc='size', fill_value=0)
        
        print("Analyzing disease by age")
        disease_age = full_df.pivot_table(index='Assigned_Disease', columns='Age_Bin', aggfunc='size', fill_value=0)

        return disease_counts, disease_sex, disease_age
    
    except Exception as e:
        print(f"Unable to analyze disease assignments: {e}")


def plot_disease_results(disease_counts, disease_sex, disease_age, save_dir=None):
    """
    Plots bar charts for disease assignment analyses.

    Args:
        disease_counts: Series with disease assignment counts.
        disease_sex: DataFrame of disease by Sex.
        disease_age: DataFrame of disease by Age_Bin.

    Returns:
        Three bar plots visually portraying the relationships in each of the three inputs.
    """
    
    try:
        # Disease counts
        plt.figure(figsize=(10, 8))
        disease_counts.plot(kind='bar')
        plt.title("Disease Counts", fontsize=18)
        plt.xlabel("Disease")
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "disease_counts.png"))
            plt.close()
        
        # Disease by Sex
        plt.figure(figsize=(10, 8))
        disease_sex.plot(kind='bar', stacked=False)
        plt.title("Disease Distribution by Sex", fontsize=18)
        plt.xlabel("Disease")
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        plt.legend(title="Sex")
        plt.tight_layout()
        plt.show()

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "disease_by_sex.png"))
            plt.close()

        # Disease by Age Bin
        plt.figure(figsize=(10, 8))
        disease_age.plot(kind='bar', stacked=False)
        plt.title("Disease Distribution by Age Bin", fontsize=18)
        plt.xlabel("Disease")
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        plt.legend(title="Age Bin")
        plt.tight_layout()
        plt.show()

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "disease_by_age.png"))
            plt.close()
    
    except Exception as e:
        print(f"Unable to create visualizations: {e}")

def analyze_dem_info(full_df, save_dir=None):
    """
    Plots boxplots for BMI, Sex, and Age Bin analyses.


    Args:
        full_df: A cleaned DataFrame with predicted diseases.

    Returns: 
        Boxplot showing BMI distribution by Sex and Age Bin.
    """
    
    try: 
        plt.figure(figsize=(8,6))
        sns.boxplot(x='Sex', y='BMI', hue='Age_Bin', data=full_df)
        plt.title('BMI Distribution by Sex and Age Bin', fontsize=18)
        plt.show()

        if save_dir is not None:
                plt.savefig(os.path.join(save_dir, "disease_by_age.png"))
                plt.close()
    
    except Exception as e:
        print(f"Unable to create visualizations: {e}")
