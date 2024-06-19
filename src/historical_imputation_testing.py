"""
Notes for running the script:
1. Edit the path in CONFIG_PATH (lien 16) to point to the historical imputation config file
2. To run the script: pytest historical_imputation_testing.py
"""

import pytest
import yaml
import pandas as pd
from historical_imputation_functions import (
    construct_incomes_donor_pool,
    historical_imputation_income
)
from donor_imputation_functions import data_preprocessing_store_donor_info
from pathlib import Path

CONFIG_PATH = '/workspaces/debug/src/historical_imp_config.yaml'

# Sample data fixture
@pytest.fixture
def sample_data():
    data = {
        "userid": [1, 2, 3],
        "Age_Interview_prev_wave": [30.0, 35.0, 28.0],
        "Q127_prev_wave": [1.0, 2.0, 3.0],
        "Q128_1_prev_wave": [100.0, 250.0, 350.0],
        "GO_Q130a_1_prev_wave": ["A", "B", "C"],
        "Q140a_1_1_prev_wave": [50.0, 60.0, 70.0],
        "Age_Interview": [32.0, 36.0, 29.0],
        "Q2001": [500.0, 600.0, 700.0],
        "Q127": [1.5, 2.5, 3.5],
        "Q128_1": [150.0, 250.0, 350.0],
        "GO_Q130a_1": ["D", "E", "F"],
        "Q135_1": [1.6, 2.6, 3.6],
        "Q140a_1_1": [55.0, 65.0, 75.0],
        "Job_change": [0, 1, 0],
    }
    return pd.DataFrame(data)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Test case 1: Handling missing donor information
def test_missing_donor_information(sample_data):
    """
    Test case for handling missing donor information in construct_incomes_donor_pool.
    """
    config = load_config(CONFIG_PATH)
    var_to_impute_list = list(config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"].keys())
    
    for var_current_wave in var_to_impute_list:
        var_previous_wave = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["var_previous_wave"]
        var_number_jobs_current_wave = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["var_number_jobs_current_wave"]
        var_number_jobs_previous_wave = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["var_number_jobs_previous_wave"]
        var_ssic_current_wave = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["var_ssic_current_wave"]
        values_to_impute = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["values_to_impute"]

        # Scenario 1: One record should remain after filtering
        df1 = sample_data.copy()
        df1.loc[0, var_current_wave] = -1  # Simulate missing donor information
        imputation_group = construct_incomes_donor_pool(
            df1, var_current_wave, var_previous_wave, var_number_jobs_current_wave,
            var_number_jobs_previous_wave, var_ssic_current_wave, values_to_impute)
        assert len(imputation_group) == 2

        # Scenario 2: Two records should remain after filtering
        df2 = sample_data.copy()
        df2.loc[0, var_previous_wave] = -1  # Simulate missing donor information
        imputation_group = construct_incomes_donor_pool(
            df2, var_current_wave, var_previous_wave, var_number_jobs_current_wave,
            var_number_jobs_previous_wave, var_ssic_current_wave, values_to_impute)
        assert len(imputation_group) == 2

        # Scenario 3: Ensure correct filtering
        imputation_group = construct_incomes_donor_pool(
            sample_data, var_current_wave, var_previous_wave, var_number_jobs_current_wave,
            var_number_jobs_previous_wave, var_ssic_current_wave, values_to_impute)
        assert len(imputation_group) == 2

# Test case 2: No available donors
def test_no_available_donors(sample_data):
    """
    Test case for handling scenarios where no donors are available.
    """
    config = load_config(CONFIG_PATH)
    var_to_impute_list = list(config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"].keys())

    for var_current_wave in var_to_impute_list:
        var_previous_wave = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["var_previous_wave"]
        var_number_jobs_current_wave = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["var_number_jobs_current_wave"]
        var_number_jobs_previous_wave = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["var_number_jobs_previous_wave"]
        var_ssic_current_wave = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["var_ssic_current_wave"]
        values_to_impute = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["values_to_impute"]

        df = sample_data.copy()
        df.loc[:, var_current_wave] = -1  # Simulate no available donors
        imputation_group = construct_incomes_donor_pool(
            df, var_current_wave, var_previous_wave, var_number_jobs_current_wave,
            var_number_jobs_previous_wave, var_ssic_current_wave, values_to_impute)
        assert imputation_group.empty

# Test case 3: Correct historical imputation
def test_correct_historical_imputation(sample_data):
    """
    Test case for ensuring correct historical imputation of income variables.
    """
    config = load_config(CONFIG_PATH)
    var_to_impute_list = list(config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"].keys())

    for var_current_wave in var_to_impute_list:
        var_previous_wave = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["var_previous_wave"]
        var_number_jobs_current_wave = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["var_number_jobs_current_wave"]
        var_number_jobs_previous_wave = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["var_number_jobs_previous_wave"]
        var_ssic_current_wave = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["var_ssic_current_wave"]
        values_to_impute = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["values_to_impute"]
        imputation_class_vars = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["imputation_class_vars"]

        imputation_group = construct_incomes_donor_pool(
            sample_data.copy(), var_current_wave, var_previous_wave, var_number_jobs_current_wave,
            var_number_jobs_previous_wave, var_ssic_current_wave, values_to_impute)

        df_data_hist_imputation_ready = data_preprocessing_store_donor_info(
            sample_data.copy(), var_current_wave, imputation_class_vars)

        imputed_data = historical_imputation_income(
            df_data_hist_imputation_ready, var_current_wave, var_previous_wave,
            var_number_jobs_current_wave, var_number_jobs_previous_wave,
            var_ssic_current_wave, imputation_group, values_to_impute)

        assert not imputed_data[var_current_wave].isna().any()

# Test case 4: Edge case with all missing values
def test_all_missing_values(sample_data):
    """
    Test case for handling edge cases where all values are missing.
    """
    config = load_config(CONFIG_PATH)
    var_to_impute_list = list(config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"].keys())

    for var_current_wave in var_to_impute_list:
        var_previous_wave = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["var_previous_wave"]
        var_number_jobs_current_wave = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["var_number_jobs_current_wave"]
        var_number_jobs_previous_wave = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["var_number_jobs_previous_wave"]
        var_ssic_current_wave = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["var_ssic_current_wave"]
        values_to_impute = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["values_to_impute"]
        imputation_class_vars = config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"][var_current_wave]["imputation_class_vars"]

        df = sample_data.copy()
        df.loc[:, var_current_wave] = float('nan')  # Simulate all missing values
        imputation_group = construct_incomes_donor_pool(
            df, var_current_wave, var_previous_wave, var_number_jobs_current_wave,
            var_number_jobs_previous_wave, var_ssic_current_wave, values_to_impute)

        df_data_hist_imputation_ready = data_preprocessing_store_donor_info(
            df, var_current_wave, imputation_class_vars)

        imputed_data = historical_imputation_income(
            df_data_hist_imputation_ready, var_current_wave, var_previous_wave,
            var_number_jobs_current_wave, var_number_jobs_previous_wave,
            var_ssic_current_wave, imputation_group, values_to_impute)

        assert imputed_data[var_current_wave].isna().sum() == len(df)