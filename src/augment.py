import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def predict_sex_age_nutri(nutri_sex_df, nutri_age_df, nutri_race_df) -> pd.DataFrame:
    """
    Use RandomForestClassifier and one-hot-encoding to assign Sex and Age Bin to nutri_race_df.
    
    Args:
        nutri_race_df: A DataFrame from nutri_df stratified by race.
        nutri_sex_df: A DataFrame from nutri_df stratified by sex.
        nutri_age_df: A DataFrame from nutri_df stratified by age.

    Returns:
        pd.DataFrame: nutri_race_df with assigned Sex and Age Bin columns.
    """
    try:
        print("Using RandomForestClassifier and one-hot-encoding for nutri_df...")

        feature_cols = ['YearStart', 'YearEnd', 'LocationDesc', 'Topic']  

        # Assign Sex
        X_sex = pd.get_dummies(nutri_sex_df[feature_cols])
        y_sex = nutri_sex_df['Sex']

        le_sex = LabelEncoder()
        y_sex_le = le_sex.fit_transform(y_sex)

        print("Running classifier for Sex...")
        rfc_sex_clf = RandomForestClassifier(random_state=42, class_weight='balanced')
        rfc_sex_clf.fit(X_sex, y_sex_le)

        X_race = pd.get_dummies(nutri_race_df[feature_cols])
        X_race_sex = X_race.reindex(columns=X_sex.columns, fill_value=0)

        print("Assigning Sex...")
        sex_preds = rfc_sex_clf.predict(X_race_sex)
        nutri_race_df['Sex'] = le_sex.inverse_transform(sex_preds)

        # Assign Age Bin
        X_age = pd.get_dummies(nutri_age_df[feature_cols])
        y_age = nutri_age_df['age_bin']

        le_age = LabelEncoder()
        y_age_le = le_age.fit_transform(y_age)

        print("Training classifier for Age Bin...")
        rfc_age_clf = RandomForestClassifier(random_state=42, class_weight='balanced')
        rfc_age_clf.fit(X_age, y_age_le)

        X_race_age = X_race.reindex(columns=X_age.columns, fill_value=0)

        print("Assigning Age Bin...")
        age_preds = rfc_age_clf.predict(X_race_age)
        nutri_race_df['Age_Bin'] = le_age.inverse_transform(age_preds)

        print("Sex and Age Bin successfully assigned to nutri_df!")
        return nutri_race_df

    except Exception as e:
        print(f"Sex and Age Bin could not be assigned to nutri_df: {e}")


def predict_sex_age_chronic(chronic_sex_df, chronic_age_df, chronic_race_df) -> pd.DataFrame:
    """
    Use RandomForestClassifier and one-hot-encoding to assign Age Bin to chronic_race_df.

    Args:
        chronic_age_df: A DataFrame from chronic_df stratified by age.
        chronice_race_df: A DataFrame from chronic_df stratified by rage.

    Returns:
        pd.DataFrame: chronic_race_df with assigned Sex and Age Bin column.
    """

    try:
        print("Using RandomForestClassifier and one-hot-encoding for chronic_df...")

        feature_cols = ['YearStart', 'YearEnd', 'LocationDesc', 'Topic']  

        # Assign Sex
        X_sex = pd.get_dummies(chronic_sex_df[feature_cols])
        y_sex = chronic_sex_df['Sex']

        le_sex = LabelEncoder()
        y_sex_le = le_sex.fit_transform(y_sex)

        print("Running classifier for Sex...")
        rfc_sex_clf = RandomForestClassifier(random_state=42)
        rfc_sex_clf.fit(X_sex, y_sex_le)

        X_race = pd.get_dummies(chronic_race_df[feature_cols])
        X_race_sex = X_race.reindex(columns=X_sex.columns, fill_value=0)

        print("Assigning Sex...")
        sex_preds = rfc_sex_clf.predict(X_race_sex)
        chronic_race_df['Sex'] = le_sex.inverse_transform(sex_preds)

        # Assign Age Bin
        X_age = pd.get_dummies(chronic_age_df[feature_cols])
        y_age = chronic_age_df['age_bin']

        le_age = LabelEncoder()
        y_age_le = le_age.fit_transform(y_age)

        print("Training classifier for Age Bin...")
        rfc_age_clf = RandomForestClassifier(random_state=42, class_weight='balanced')
        rfc_age_clf.fit(X_age, y_age_le)

        X_race_age = X_race.reindex(columns=X_age.columns, fill_value=0)

        print("Assigning Age Bin...")
        age_preds = rfc_age_clf.predict(X_race_age)
        chronic_race_df['Age_Bin'] = le_age.inverse_transform(age_preds)

        print("Sex and Age Bin successfully assigned to chronic_df!")
        return chronic_race_df

    except Exception as e:
        print(f"Age Bin could not be assigned to chronic_df: {e}")


def predict_obesity(nutri_combined, chronic_combined) -> pd.DataFrame:
    """"
    Use RandomForestClassifier and one-hot-encoding to predict a secondary disease for chronic_combined based on nutri_combined.
    Args:
        nutri_combined: An engineered DataFrame with added Sex and Age_Bin columns.
        chronic_combined: An engineered DataFrame with added Sex and Age_Bin columns.

    Returns:
        pd.DataFrame: A cleaned DataFrame that has Obesity / Weight Status predicted as a secondary disease.
    """

    try:
        feature_cols = ['LocationDesc', 'Race/Ethnicity', 'Sex', 'Age_Bin']
        
        print(f"Assigning binary values for presence of obesity / weight problems...")
        nutri_combined['Obesity_Binary'] = (nutri_combined['Topic'] == 'Obesity / Weight Status').astype(int)
        
        X_nutri = pd.get_dummies(nutri_combined[feature_cols])
        y_nutri = nutri_combined['Obesity_Binary']

        print("Training classifier for Obesity_Binary...")
        rfc_obesity_clf = RandomForestClassifier(random_state=42, class_weight='balanced')
        rfc_obesity_clf.fit(X_nutri, y_nutri)

        print("Assigning Obesity_Binary...")
        X_chronic = pd.get_dummies(chronic_combined[feature_cols])
        X_chronic = X_chronic.reindex(columns=X_nutri.columns, fill_value=0)
        chronic_combined['Obesity_Binary'] = rfc_obesity_clf.predict(X_chronic)

        print("Obesity_Binary successfully assigned!")
        return chronic_combined
    
    except Exception as e:
        print(f"Obesity / Weight Status could not be predicted: {e}")


def assign_disease(second_disease_df, aw_fb_df) -> pd.DataFrame:
    """
    Use RandomForestClassifier and one-hot-encoding.
    Assign diseases to the aw_fb_df based on 2 conditions:
        1. aw_fb_df['Disease'] == 1
        2. second_disease_df['Obesity_Binary'] == 1

    Args:
        second_disease_df: A full DataFrame indicating whether a person may also be experience obesity / weight problems.
        aw_fb_df: A cleaned DataFrame of Apple Watch and FitBit data.

    Returns:
        pd.DataFrame: A combined DataFrame assigning the types of diseases a person may be suffering from.
    """

    try:
        feature_cols = ['Sex', 'Age_Bin']
        train_df = second_disease_df[second_disease_df['Obesity_Binary'] == 1].copy()

        X_train = pd.get_dummies(train_df[feature_cols])
        y_train = train_df['Topic'] 

        le_topic = LabelEncoder()
        y_train_le = le_topic.fit_transform(y_train)

        print("Training classifier for predicting disease...")
        rfc_disease_clf = RandomForestClassifier(random_state=42, class_weight='balanced')
        rfc_disease_clf.fit(X_train, y_train_le)

        X_awfb = pd.get_dummies(aw_fb_df[feature_cols])
        X_awfb = X_awfb.reindex(columns=X_train.columns, fill_value=0)
        aw_fb_df['Possible_Disease'] = le_topic.inverse_transform(rfc_disease_clf.predict(X_awfb))
        
        def assign_topic(row):
            if row['Disease'] == 1 and row['Possible Obesity'] == 1:
                return row['Possible_Disease']
            return None

        print("Assigning whether disease should exist or not...")
        aw_fb_df['Assigned_Disease'] = aw_fb_df.apply(assign_topic, axis=1)
    
        print(f"Successfully assigned disease to aw_fb_df!")
        return aw_fb_df
    
    except Exception as e:
        print(f"Disease could not be assigned to aw_fb_df: {e}")