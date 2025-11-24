import pandas as pd


# --- 1. CLEANS Apple Watch and Fitbit DATA
def process_aw_fb_data(aw_fb_df) -> pd.DataFrame:
    """
    Preprocesses data collected from aw_fb_data.csv.

    Args:
        aw_fb_df: The DataFrame created after running get_csv from load.py.

    Returns:
        pd.DataFrame: A DataFrame with cleaned and engineered features.
    """

    try: 
        # Creating dataframe of equal length
        print(f"Cleaning aw_fb_data...")
        aw_fb_cleaned = pd.DataFrame(index=aw_fb_df.index)

        # Adding engineered / cleaned features
        aw_fb_cleaned['Device'] = aw_fb_df['device'].map({'apple watch': 'Apple Watch', 'fitbit': 'Fitbit'})
        aw_fb_cleaned['Activity'] = aw_fb_df['activity']
        aw_fb_cleaned['Sex'] = aw_fb_df['gender'].map({0: 'Female', 1: 'Male'})
        aw_fb_cleaned['Age'] = aw_fb_df['age']
        aw_fb_cleaned['Age_Bin'] = aw_fb_df['age'].apply(lambda x: '18-44' if 18 <= x <= 44 else ('45-64' if 45 <= x <= 64 else ('65+' if x >= 65 else 'other')))
        aw_fb_cleaned['Height_cm'] = aw_fb_df['height']
        aw_fb_cleaned['Weight_kg'] = aw_fb_df['weight']
        aw_fb_cleaned['BMI'] = aw_fb_df['weight'] / ((aw_fb_df['height'] / 100) ** 2)
        aw_fb_cleaned['heart_rate'] = aw_fb_df['hear_rate']
        aw_fb_cleaned['sd_norm_heart'] = aw_fb_df['sd_norm_heart']
        aw_fb_cleaned['resting_heart'] = aw_fb_df['resting_heart']
        aw_fb_cleaned['intensity_karvonen'] = aw_fb_df['intensity_karvonen']
        aw_fb_cleaned['target_heart_rate'] = aw_fb_cleaned['resting_heart'] + (aw_fb_cleaned['heart_rate'] - aw_fb_cleaned['resting_heart']) * aw_fb_cleaned['intensity_karvonen']
        aw_fb_cleaned['Disease'] = aw_fb_cleaned.apply(lambda row: 1 if (row['heart_rate'] > row['target_heart_rate'] + 2 * row['sd_norm_heart']) and 
                                                                        (row['heart_rate'] > row['target_heart_rate'] - 2 * row['sd_norm_heart'])
                                                                        else 0, axis=1)
        aw_fb_cleaned['Possible Obesity'] = aw_fb_cleaned.apply(lambda row: 1 if (18.5 <= row['BMI'] > 18.5 <= 24.9)
                                                                        else 0, axis=1)                                       
        print("Data successfully cleaned.")
        return aw_fb_cleaned
    
    except Exception as e:
        print(f"Could not clean aw_fb_data: {e}")


# --- 2. CLEANS Nutrition Physical Activity and Obesity - Behavioral Risk Factor Surveillance System DATA
def process_nutri_data(nutri_df) -> tuple:
    """
    Preprocesses data collected from Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv.

    Args:
        nutri_df: The DataFrame created data after running get_csv from load.py.

    Returns:
        tuple: A tuple containing three DataFrames with cleaned and engineered features.
            - nutri_age_df: A DataFrame stratified by age.
            - nutri_race_df: A DataFrame stratified by race.
            - nutri_sex_df: A DataFrame stratified by sex.
    """

    try:
        print("Cleaning nutri_data...")
        keep_columns = ['YearStart', 'YearEnd', 'LocationDesc', 'Topic', 'Sample_Size', 'StratificationCategory1', 'Stratification1']
        nutri_df_cleaned = nutri_df[keep_columns]

        print("Creating nutri_sex_df...")
        nutri_sex_df = nutri_df_cleaned[nutri_df_cleaned['StratificationCategory1'] == 'Sex']
        nutri_sex_df = nutri_sex_df.drop(columns='StratificationCategory1').rename(columns={'Stratification1': 'Sex'})
       
        print("Creating nutri_age_df...")
        
        # Function to group ages into corresponding age bins
        def map_nutri_age(age_label):
            if age_label in ['18 - 24', '25 - 34', '35 - 44']:
                return '18-44'
            elif age_label in ['45 - 54', '55 - 64']:
                return '45-64'
            elif age_label == '65 or older':
                return '65+'
            else:
                return None

        nutri_age_df = nutri_df_cleaned[nutri_df_cleaned['StratificationCategory1'] == 'Age (years)']
        nutri_age_df = nutri_age_df.drop(columns='StratificationCategory1').rename(columns={'Stratification1': 'age_bin'})
        nutri_age_df['age_bin'] = nutri_age_df['age_bin'].apply(map_nutri_age)

        print("Creating nutri_race_df...")
        nutri_race_df = nutri_df_cleaned[nutri_df_cleaned['StratificationCategory1'] == 'Race/Ethnicity']
        nutri_race_df = nutri_race_df.drop(columns='StratificationCategory1').rename(columns={'Stratification1': 'Race/Ethnicity'})

        print("Data successfully cleaned and split.")
        return nutri_sex_df, nutri_age_df, nutri_race_df
    
    except Exception as e:
        print(f"Could not clean nutri_data: {e}")


# --- 3. CLEANS U.S. Chronic Disease Indicators DATA (REST API)
def process_chronic_data(chronic_df_raw) -> tuple:
    """
    Preprocesses data collected from US Chronic Disease Indicators.

    Args:
        chronic_df: The DataFrame created data after running get_chronic_data from load.py.

    Returns:
        tuple: A tuple containing three DataFrames with cleaned and engineered features.
            - chronic_age_df: A DataFrame stratified by age.
            - chronic_race_df: A DataFrame stratified by race.
            - chronic_sex_df: A DataFrame stratified by sex.
    """

    try:
        print("Cleaning chronic_data...")
        keep_columns = ['YearStart', 'YearEnd', 'LocationDesc', 'Topic', 'StratificationCategory1', 'Stratification1' ]
        chronic_df_cleaned = chronic_df_raw[keep_columns]

        print("Creating chronic_age_df...")
        valid_ages = ['18-44', '45-64', '>=65']
        chronic_age_df = chronic_df_cleaned[chronic_df_cleaned['StratificationCategory1'] == 'Age']
        chronic_age_df = chronic_age_df.rename(columns={'Stratification1': 'age_bin'})
        chronic_age_df['age_bin'] = chronic_age_df['age_bin'].str.replace('Age ', '', regex=False)
        chronic_age_df = chronic_age_df[chronic_age_df['age_bin'].isin(valid_ages)]
        chronic_age_df['age_bin'] = chronic_age_df['age_bin'].replace({'>=65': '65+'})
        chronic_age_df = chronic_age_df.drop(columns=['StratificationCategory1'])

        print("Creating chronic_race_df...") 
        chronic_race_df = chronic_df_cleaned[chronic_df_cleaned['StratificationCategory1'] == 'Race/Ethnicity']
        chronic_race_df = chronic_race_df.drop(columns=['StratificationCategory1']).rename(columns={'Stratification1': 'Race/Ethnicity'})

        print("Creating chronic_sex_df...")
        chronic_sex_df = chronic_df_cleaned[chronic_df_cleaned['StratificationCategory1'] == 'Sex']
        chronic_sex_df = chronic_sex_df.drop(columns=['StratificationCategory1']).rename(columns={'Stratification1': 'Sex'})

        print("Data successfully cleaned and split.")
        return chronic_age_df, chronic_race_df, chronic_sex_df
    
    except Exception as e:
        print(f"Could not clean chronic_data: {e}")
