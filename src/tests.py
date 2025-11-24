import unittest
import pandas as pd
from load import get_csv, get_chronic_data
from process import process_aw_fb_data, process_chronic_data, process_nutri_data
from augment import predict_sex_age_nutri, assign_disease


# Test if data is loaded properly
class TestCSVLoading(unittest.TestCase):
    def test_get_csv_not_empty(self):
        filepath = "../data/aw_fb_data.csv"
        df = get_csv(filepath)
        self.assertIsNotNone(df, "get_csv returned None.")
        self.assertGreater(len(df), 0, "CSV loaded by get_csv is empty.")

    def test_get_chronic_data_not_empty(self):
        url = "https://data.cdc.gov/api/views/hksd-2xuw/rows.csv?accessType=DOWNLOAD"
        df = get_chronic_data(url)
        self.assertIsNotNone(df, "get_chronic_data returned None.")
        self.assertGreater(len(df), 0, "CSV loaded by get_chronic_data is empty.")


# Test if data is processed properly
class TestProcessing(unittest.TestCase):
    def test_process_aw_fb_data(self):
        # Expected columns in the processed output
        expected_columns = [
                'Device', 'Activity', 'Sex', 'Age', 'Age_Bin', 'Height_cm', 'Weight_kg', 'BMI', 'heart_rate', 'sd_norm_heart', 'resting_heart',
                'intensity_karvonen', 'target_heart_rate', 'Disease', 'Possible Obesity']
        
        test_df = pd.DataFrame({
            'device': ['Apple Watch'],
            'activity': ['Lying'],
            'gender': ['Female'],
            'age': [30],
            'height': [170],
            'weight': [70],
            'hear_rate': [80],
            'sd_norm_heart': [5],
            'resting_heart': [70],
            'intensity_karvonen': [60]
        })

        processed = process_aw_fb_data(test_df)
        self.assertIsNotNone(processed, "Processing aw_fb returned None.")
        self.assertGreater(len(processed), 0, "Processed aw_fb data is empty.")
        for col in expected_columns:
            self.assertIn(col, processed.columns, f"Processed data missing column: {col}")
    
    def test_process_chronic_data(self):
        # Expected columns in the processed output
        expected_age_columns = ['YearStart', 'YearEnd', 'LocationDesc', 'Topic', 'age_bin']
        expected_race_columns = ['YearStart', 'YearEnd', 'LocationDesc', 'Topic', 'Race/Ethnicity']
        expected_sex_columns = ['YearStart', 'YearEnd', 'LocationDesc', 'Topic', 'Sex']

        test_df = pd.DataFrame([
            {
                'YearStart': 2019,
                'YearEnd': 2019,
                'LocationDesc': 'California',
                'Topic': 'Health Status',
                'StratificationCategory1': 'Age',
                'Stratification1': '18-44'
            },
            {
                'YearStart': 2014,
                'YearEnd': 2019,
                'LocationDesc': 'Colorado',
                'Topic': 'Health Status',
                'StratificationCategory1': 'Race/Ethnicity',
                'Stratification1': 'Asian or Pacific Islander'
            },
            {
                'YearStart': 2019,
                'YearEnd': 2019,
                'LocationDesc': 'Texas',
                'Topic': 'Immunization',
                'StratificationCategory1': 'Sex',
                'Stratification1': 'Male'
            }
        ])

        processed_age, processed_race, processed_sex = process_chronic_data(test_df)

        for df, label, expected_columns in zip(
            [processed_age, processed_race, processed_sex],
            ["age", "race", "sex"],
            [expected_age_columns, expected_race_columns, expected_sex_columns]
        ):
            self.assertIsNotNone(df, f"Processing chronic data (split: {label}) returned None.")
            self.assertGreater(len(df), 0, f"Processed chronic data (split: {label}) is empty.")
            for col in expected_columns:
                self.assertIn(col, df.columns, f"Processed chronic data (split: {label}) missing column: {col}")
    
    def test_process_nutri_data(self):
        # Expected columns in the processed output
        expected_age_columns = ['YearStart', 'YearEnd', 'LocationDesc', 'Topic', 'Sample_Size', 'age_bin']
        expected_race_columns = ['YearStart', 'YearEnd', 'LocationDesc', 'Topic', 'Sample_Size', 'Race/Ethnicity']
        expected_sex_columns = ['YearStart', 'YearEnd', 'LocationDesc', 'Topic', 'Sample_Size', 'Sex']

        test_df = pd.DataFrame([
            {
                'YearStart': 2019,
                'YearEnd': 2019,
                'LocationDesc': 'California',
                'Topic': 'Health Status',
                'Sample_Size': 200, 
                'StratificationCategory1': 'Age (years)',
                'Stratification1': '25-34'
            },
            {
                'YearStart': 2014,
                'YearEnd': 2019,
                'LocationDesc': 'Colorado',
                'Topic': 'Health Status',
                'Sample_Size': 400, 
                'StratificationCategory1': 'Race/Ethnicity',
                'Stratification1': 'Asian or Pacific Islander'
            },
            {
                'YearStart': 2019,
                'YearEnd': 2019,
                'LocationDesc': 'Texas',
                'Topic': 'Immunization',
                'Sample_Size': 5000, 
                'StratificationCategory1': 'Sex',
                'Stratification1': 'Male'
            }
        ])

        processed_sex, processed_age, processed_race = process_nutri_data(test_df)

        for df, label, expected_columns in zip(
            [processed_age, processed_race, processed_sex],
            ["age", "race", "sex"],
            [expected_age_columns, expected_race_columns, expected_sex_columns]
        ):
            self.assertIsNotNone(df, f"Processing nutri data (split: {label}) returned None.")
            self.assertGreater(len(df), 0, f"Processed nutri data (split: {label}) is empty.")
            for col in expected_columns:
                self.assertIn(col, df.columns, f"Processed nutri data (split: {label}) missing column: {col}")


# Test if data is augmented properly
class TestAugmentationAndAnalysis(unittest.TestCase):
    def test_predict_sex_age_nutri(self):
        # Expected columns in the processed output
        expected_age_columns = ['YearStart', 'YearEnd', 'LocationDesc', 'Topic', 'Sample_Size', 'age_bin']
        expected_race_columns = ['YearStart', 'YearEnd', 'LocationDesc', 'Topic', 'Sample_Size', 'Race/Ethnicity']
        expected_sex_columns = ['YearStart', 'YearEnd', 'LocationDesc', 'Topic', 'Sample_Size', 'Sex']
        
        nutri_sex_df = pd.DataFrame([
            [2015, 2016, "LocationA", "Health Status", 200, "Female"],
            [2015, 2016, "LocationB", "Immunization", 459872, "Male"]
        ], columns=expected_sex_columns)
        
        nutri_age_df = pd.DataFrame([
            [2015, 2016, "LocationA", "Alcohol", 50, "18-44"],
            [2015, 2016, "LocationA", "Cancer", 1029, "45-64"]
        ], columns=expected_age_columns)
        
        nutri_race_df = pd.DataFrame([
            [2015, 2016, "LocationA", "Health Status", 23848, "Hispanic"],
            [2015, 2016, "LocationB", "Asthma", 1094, "Asian"]
        ], columns=expected_race_columns)
        
        result = predict_sex_age_nutri(nutri_sex_df, nutri_age_df, nutri_race_df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Sex", result.columns)
        self.assertIn("Age_Bin", result.columns)
        self.assertGreater(len(result), 0)

    def test_assign_disease(self):

        second_disease_df = pd.DataFrame({
            'Sex': ['Female', 'Male', 'Female'],
            'Age_Bin': ['18-44', '18-44', '45-64'],
            'Obesity_Binary': [1, 1, 1],
            'Topic': ['Heart', 'Obesity', 'Hypertension']
        })

        aw_fb_df = pd.DataFrame({
            'Sex': ['Female', 'Male', 'Female', 'Female'],
            'Age_Bin': ['18-44', '18-44', '45-64', '18-44'],
            'Disease': [1, 0, 1, 1],
            'Possible Obesity': [1, 1, 0, 1]
        })

        result = assign_disease(second_disease_df, aw_fb_df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('Possible_Disease', result.columns)
        self.assertIn('Assigned_Disease', result.columns)

        assigned = result.loc[result['Assigned_Disease'].notnull()]
        self.assertGreaterEqual(len(assigned), 1, "No disease assignments were made when some should have been.")

        for idx, row in result.iterrows():
            if row['Disease'] == 1 and row['Possible Obesity'] == 1:
                self.assertIsNotNone(row['Assigned_Disease'], "Assignment should exist when both conditions are satisfied.")
            else:
                self.assertIsNone(row['Assigned_Disease'], "Assignment should not exist when conditions are not satisfied.")


if __name__ == "__main__":
    unittest.main()
