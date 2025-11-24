import os
from load import get_csv, get_chronic_data
from process import process_aw_fb_data, process_chronic_data, process_nutri_data
from augment import predict_sex_age_nutri, predict_sex_age_chronic, predict_obesity, assign_disease
from analyze import analyze_aw_fb_data, analyze_chronic_data, analyze_nutri_data, analyze_assigned_diseases, plot_results

if __name__ == "__main__":

    # Creating Directories
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 1. Load data ---
    print("Loading data...")
    aw_fb_df = get_csv('../data/aw_fb_data.csv')
    nutri_df = get_csv('../data/Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv')
    chronic_df = get_chronic_data(url = 'https://data.cdc.gov/api/views/hksd-2xuw/rows.csv?accessType=DOWNLOAD')
    
    aw_fb_df.to_csv(os.path.join(DATA_DIR, 'aw_fb_data_loaded.csv'), index=False)
    nutri_df.to_csv(os.path.join(DATA_DIR, 'nutri_data_loaded.csv'), index=False)
    chronic_df.to_csv(os.path.join(DATA_DIR, 'chronic_data_loaded.csv'), index=False)

    # --- 2. Process data ---
    print("Processing data...")
    aw_fb_cleaned = process_aw_fb_data(aw_fb_df)
    nutri_sex_df, nutri_age_df, nutri_race_df = process_nutri_data(nutri_df)
    chronic_age_df, chronic_race_df, chronic_sex_df = process_chronic_data(chronic_df)
    
    # --- 3. Conduct EDA ---
    print("Conducting EDA...")
    analyze_aw_fb_data(aw_fb_cleaned, save_dir=RESULTS_DIR)
    analyze_nutri_data(nutri_sex_df, nutri_age_df, nutri_race_df, save_dir=RESULTS_DIR)
    analyze_chronic_data(chronic_age_df, chronic_race_df, chronic_sex_df, save_dir=RESULTS_DIR)
    
    # --- 4. Augment/Engineer features ---
    print("Engineering features and using RandomForestClassifer...")
    nutri_combined = predict_sex_age_nutri(nutri_sex_df, nutri_age_df, nutri_race_df)
    chronic_combined = predict_sex_age_chronic(chronic_sex_df, chronic_age_df, chronic_race_df)
    
    nutri_combined.to_csv(os.path.join(RESULTS_DIR, 'nutri_combined.csv'), index=False)
    chronic_combined.to_csv(os.path.join(RESULTS_DIR, 'chronic_combined.csv'), index=False)

    # --- 5. Predict obesity and assign secondary diseases
    print("Predicting Chronic Disease")
    second_disease_df = predict_obesity(nutri_combined, chronic_combined)
    full_df = assign_disease(second_disease_df, aw_fb_cleaned)

    full_df.to_csv(os.path.join(RESULTS_DIR, 'final_results.csv'), index=False)

    # --- 6. Analyze and plot results ---
    print("Plotting results...")
    disease_counts, disease_sex, disease_age = analyze_assigned_diseases(full_df)
    plot_results(disease_counts, disease_sex, disease_age, save_dir=RESULTS_DIR)

    print("\n--- Data collection and plotting complete. Check the `data` and 'results' directory. ---")