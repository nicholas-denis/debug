# test_historical_imputation.py

import pytest
import yaml
import pandas as pd
from historical_imputation_functions import (
    construct_incomes_donor_pool,
    historical_imputation_income
)
from donor_imputation_functions import data_preprocessing_store_donor_info
from pathlib import Path




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

def load_conifg(config_path):

    with open('/workspaces/debug/src/historical_imp_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

    

CONFIG_PATH = '/workspaces/debug/src/historical_imp_config.yaml'

# Test case for handling missing donor information
"""
Conditions being tested:
imputation_group = (
        data[
            (
                ~data[var_current_wave].isin(missing_values_to_impute)
            )  # no missing values in current wave
            & (
                ~data[var_previous_wave].isin(missing_values_to_impute)
            )  # no missing values in previous wave
            & (
                data[var_number_jobs_current_wave]
                == data[var_number_jobs_previous_wave]
            )  # number of jobs equal in previous and current waves
        ]
        .groupby([var_ssic_current_wave])[[var_current_wave, var_previous_wave]]
        .agg(["count", "mean"])
        .reset_index()
    )

"""
def test_missing_donor_information(sample_data):
    """
    Test case for handling missing donor information in construct_incomes_donor_pool.
    """

    config = load_conifg(CONFIG_PATH)
    var_to_impute_list = list(
            config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"].keys()
        )
    
    for var_current_wave in var_to_impute_list:
        var_previous_wave = config["IMPUTATION"]["HISTORICAL_IMPUTATION"][
                            "INCOME_VARS"
                        ][var_current_wave]["var_previous_wave"]
        var_number_jobs_current_wave = config["IMPUTATION"][
            "HISTORICAL_IMPUTATION"
        ]["INCOME_VARS"][var_current_wave]["var_number_jobs_current_wave"]
        var_number_jobs_previous_wave = config["IMPUTATION"][
            "HISTORICAL_IMPUTATION"
        ]["INCOME_VARS"][var_current_wave]["var_number_jobs_previous_wave"]
        var_ssic_current_wave = config["IMPUTATION"][
            "HISTORICAL_IMPUTATION"
        ]["INCOME_VARS"][var_current_wave]["var_ssic_current_wave"]
        imputation_class_vars = config["IMPUTATION"][
            "HISTORICAL_IMPUTATION"
        ]["INCOME_VARS"][var_current_wave]["imputation_class_vars"]
        values_to_impute = config["IMPUTATION"]["HISTORICAL_IMPUTATION"][
            "INCOME_VARS"
        ][var_current_wave]["values_to_impute"]

        
        df1 = sample_data.copy()
        # Replace the value in val_current_wave to -1
        # Only one record should remain
        df1.loc[0, var_current_wave] = -1  # Simulate missing donor information
        
        
        imputation_group = construct_incomes_donor_pool(
                        df1,
                        var_current_wave,
                        var_previous_wave,
                        var_number_jobs_current_wave,
                        var_number_jobs_previous_wave,
                        var_ssic_current_wave,
                        values_to_impute,
                    )
    
        assert len(imputation_group)==2


        df2 = sample_data.copy()
        # Replace the value in val_current_wave to -1
        # Two records should remain after filtering
        df2.loc[0, var_previous_wave] = -1  # Simulate missing donor information
        
        
        imputation_group = construct_incomes_donor_pool(
                        df2,
                        var_current_wave,
                        var_previous_wave,
                        var_number_jobs_current_wave,
                        var_number_jobs_previous_wave,
                        var_ssic_current_wave,
                        values_to_impute,
                    )
    
        assert len(imputation_group)==2

        df3 = sample_data.copy()
        # Replace the value in val_current_wave to -1
        # Only one record should remain
        df3.loc[0, var_previous_wave] = -1  # Simulate missing donor information
        
        
        imputation_group = construct_incomes_donor_pool(
                        df2,
                        var_current_wave,
                        var_previous_wave,
                        var_number_jobs_current_wave,
                        var_number_jobs_previous_wave,
                        var_ssic_current_wave,
                        values_to_impute,
                    )
    
        assert len(imputation_group)==2

        imputation_group = construct_incomes_donor_pool(
                        sample_data,
                        var_current_wave,
                        var_previous_wave,
                        var_number_jobs_current_wave,
                        var_number_jobs_previous_wave,
                        var_ssic_current_wave,
                        values_to_impute,
                    )
    
        assert len(imputation_group)==2





        '''

        with pytest.raises(ValueError):
            a = construct_incomes_donor_pool(
                df,
                'Q140a_1_1',
                config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"]['Q140a_1_1']['var_previous_wave'],
                config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"]['Q140a_1_1']['var_number_jobs_current_wave'],
                config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"]['Q140a_1_1']['var_number_jobs_previous_wave'],
                config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"]['Q140a_1_1']['var_ssic_current_wave'],
                config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"]['Q140a_1_1']['values_to_impute']
        )
        print(a)
        '''
'''
# Test case for handling empty donor pool
def test_empty_donor_pool(sample_data):
    """
    Test case for handling empty donor pool in construct_incomes_donor_pool.
    """
    df = sample_data.copy()
    df_empty = df.iloc[0:0]  # Create an empty dataframe
    with pytest.raises(ValueError):
        construct_incomes_donor_pool(
            df_empty, "Q128_1", "Q128_1_prev_wave", "Q2001", "Q127", "Q127_prev_wave", [100, 200, 300]
        )

# Test case for checking imputation flag consistency
def test_imputation_flag_consistency(sample_data):
    """
    Test case for checking imputation flag consistency in historical_imputation_income.
    """
    df = sample_data.copy()
    imputed_df = historical_imputation_income(
        df, "Q128_1", "Q128_1_prev_wave", "Q2001", "Q127", "Q127_prev_wave", {}, [100, 200, 300]
    )
    assert imputed_df["Q128_1_imputation_flag"].isin([0, 1]).all()  # Assert imputation flags are binary

# Test case for checking specific imputation types
def test_imputation_types(sample_data):
    """
    Test case for checking specific imputation types in historical_imputation_income.
    """
    df = sample_data.copy()
    imputed_df = historical_imputation_income(
        df, "Q128_1", "Q128_1_prev_wave", "Q2001", "Q127", "Q127_prev_wave", {}, [100, 200, 300]
    )
    
    # Check if imputed values correspond to expected types (e.g., mean, median, etc.)
    assert imputed_df["Q128_1_imputation_type"].isin(["mean", "median"]).all()

# Test case for handling categorical imputation
def test_categorical_imputation(sample_data):
    """
    Test case for handling categorical imputation in historical_imputation_income.
    """
    df = sample_data.copy()
    df["Q128_1_prev_wave"] = df["Q128_1_prev_wave"].astype(str)  # Convert to string to simulate categorical data
    imputed_df = historical_imputation_income(
        df, "Q128_1", "Q128_1_prev_wave", "Q2001", "Q127", "Q127_prev_wave", {}, ["A", "B", "C"]
    )
    
    # Assert that imputed values match one of the categorical values
    assert imputed_df["Q128_1_imputed_value"].isin(["A", "B", "C"]).all()

# Test case for handling NaN values in input
def test_nan_values_in_input(sample_data):
    """
    Test case for handling NaN values in input in historical_imputation_income.
    """
    df = sample_data.copy()
    df.loc[0, "Q2001"] = None  # Simulate NaN value in input
    with pytest.raises(ValueError):
        historical_imputation_income(
            df, "Q128_1", "Q128_1_prev_wave", "Q2001", "Q127", "Q127_prev_wave", {}, [100, 200, 300]
        )

# Test case for checking output columns existence
def test_output_columns_existence(sample_data):
    """
    Test case for checking output columns existence in historical_imputation_income.
    """
    df = sample_data.copy()
    imputed_df = historical_imputation_income(
        df, "Q128_1", "Q128_1_prev_wave", "Q2001", "Q127", "Q127_prev_wave", {}, [100, 200, 300]
    )
    
    expected_columns = [
        "userid",
        "Q2001",
        "Q127",
        "Q128_1_prev_wave",
        "Q128_1",
        "Q128_1_post_imputation",
        "Q128_1_imputed_value",
        "Q128_1_imputation_flag",
        "Q128_1_imputation_type",
        "avg_previous_wave",
        "avg_current_wave",
        "ratio"
    ]

    assert all(col in imputed_df.columns for col in expected_columns)

# Test case for checking imputed values in a specific range
def test_imputed_values_range(sample_data):
    """
    Test case for checking imputed values range in historical_imputation_income.
    """
    df = sample_data.copy()
    imputed_df = historical_imputation_income(
        df, "Q128_1", "Q128_1_prev_wave", "Q2001", "Q127", "Q127_prev_wave", {}, [100, 200, 300]
    )
    
    assert imputed_df["Q128_1_imputed_value"].between(100, 300).all()

# Test case for checking imputation function performance with large dataset
@pytest.mark.benchmark(group="large_dataset")
def test_large_dataset_performance(benchmark, sample_data):
    """
    Benchmark test case for checking performance with a large dataset in historical_imputation_income.
    """
    df_large = pd.concat([sample_data] * 1000, ignore_index=True)  # Create a large dataset
    benchmark(historical_imputation_income, df_large, "Q128_1", "Q128_1_prev_wave", "Q2001", "Q127", "Q127_prev_wave", {}, [100, 200, 300])

# Test case for handling different types of imputation values (float, int, str)
def test_different_imputation_value_types(sample_data):
    """
    Test case for handling different types of imputation values in historical_imputation_income.
    """
    df = sample_data.copy()
    imputed_df = historical_imputation_income(
        df, "Q128_1", "Q128_1_prev_wave", "Q2001", "Q127", "Q127_prev_wave", {}, [100.0, 200.0, 300.0]
    )
    
    assert imputed_df["Q128_1_imputed_value"].dtype == float  # Assert imputed value type consistency

# Test case for handling different scenarios of missing data in donor pool
def test_missing_data_in_donor_pool(sample_data):
    """
    Test case for handling different scenarios of missing data in donor pool in historical_imputation_income.
    """
    df = sample_data.copy()
    df.loc[1, "Q128_1_prev_wave"] = None  # Simulate missing donor data for a specific row
    imputed_df = historical_imputation_income(
        df, "Q128_1", "Q128_1_prev_wave", "Q2001", "Q127", "Q127_prev_wave", {}, [100, 200, 300]
    )
    
    # Assert that imputed values are handled correctly despite missing donor data
    assert imputed_df["Q128_1_imputation_flag"].isin([0, 1]).all()
'''