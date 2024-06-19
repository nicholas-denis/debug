"""
Notes for running the script:
1. Edit the path in IRIS_DATA_PATH (line 18) to point to the location of the iris dataset.
2. To run the script execute: 
    cd src
    pytest donor_imputation_testing.py
"""

import pytest
import pandas as pd
import numpy as np
import random
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import BallTree
from donor_imputation_functions import nearest_neighbour_imputation
from donor_imputation_functions import data_preprocessing_store_donor_info
from main import non_monetary_bins, non_monetary_bins_to_enumerated_val

IRIS_DATA_PATH = '/workspaces/debug/data/test_output_donor.xlsx'

# Define fixtures for test data
@pytest.fixture
def numerical_data():
    """Fixture for numerical data with missing values."""
    return pd.DataFrame({
        'userid': [1, 2, 3, 4, 5],
        'var_to_impute': [-1, 2, 3, -1, 5],
        'imputation_class_1': [1, 1, 2, 2, 3],
        'imputation_class_2': [5, 4, 3, 2, 1]
    })

@pytest.fixture
def iris_data():
    """Fixture for Iris dataset."""
    data = pd.read_excel(IRIS_DATA_PATH)
    return data

@pytest.fixture
def iris_data_imputed():
    """Fixture for imputed Iris dataset."""
    xls = pd.ExcelFile(IRIS_DATA_PATH)
    data = pd.read_excel(xls, 'sepal width (cm)')
    return data

# Test cases

"""
Test Case 1
-----------------------------------------------------------------
Objective: Test if all missing values in var_to_impute are imputed

Expected Outcome:
All missing values in var_to_impute are imputed, 
asserting that result['var_to_impute_post_imputation'] has no NaN values.
"""

def test_all_missing_values_imputed(numerical_data):
    """Test case for imputing missing values."""
    print("\nTest Case 1: Test if all the missing values in var_to_impute are imputed")
    
    var_to_impute_list = ['var_to_impute']
    imputation_class_vars = ['imputation_class_1', 'imputation_class_2']

    for var_to_impute in var_to_impute_list:
        # Preprocess data to store donor information
        df_data_donor_preprocessed = data_preprocessing_store_donor_info(
            numerical_data, var_to_impute, imputation_class_vars
        )
        # Perform nearest neighbor imputation
        result = nearest_neighbour_imputation(
            data=df_data_donor_preprocessed,
            var_to_impute=var_to_impute,
            imputation_class_vars=imputation_class_vars,
            missing_values_to_impute=[-1],
            bucket_imputation=False,
            max_donor_use=5,
            userid_col_name='userid'
        )
    
    # Assert that there are no missing values after imputation
    assert not result['var_to_impute_post_imputation'].isna().any()

"""
Test Case 2
-----------------------------------------------------------------
Objective: Test if the maximum donor use limit is respected

Expected Outcome:
No single donor is used more than the set value for the maximum donor use for imputing missing values, 
asserting that the maximum donor count does not exceed max_donor_use.
"""

def test_max_donor_use(numerical_data):
    """Test case to check maximum donor usage limit."""
    print("\nTest Case 2: Test if no single donor is used more than the max_donor_use for imputing missing values")

    var_to_impute_list = ['var_to_impute']
    imputation_class_vars = ['imputation_class_1', 'imputation_class_2']
    max_donor_use = 5

    for var_to_impute in var_to_impute_list:
        # Preprocess data to store donor information
        df_data_donor_preprocessed = data_preprocessing_store_donor_info(
            numerical_data, var_to_impute, imputation_class_vars
        )
        # Perform nearest neighbor imputation
        result = nearest_neighbour_imputation(
            data=df_data_donor_preprocessed,
            var_to_impute=var_to_impute,
            imputation_class_vars=imputation_class_vars,
            missing_values_to_impute=[-1],
            bucket_imputation=False,
            max_donor_use=max_donor_use,
            userid_col_name='userid'
        )

    # Check the maximum donor count used for any donor
    donor_counts = result['var_to_impute_donor_count'].value_counts()
    assert donor_counts.max() <= max_donor_use

"""
Test Case 3
-----------------------------------------------------------------
Objective: Test if all the missing values in var_to_impute are imputed when categorical variables are present

Expected Outcome:
All missing values in var_to_impute are imputed, 
asserting that result['sepal width (cm)_post_imputation'] has no NaN values.
"""

def test_categorical_class_vars(iris_data):
    """Test case for imputing missing values with categorical variables."""
    print("\nTest Case 3: Test if all the missing values in var_to_impute are imputed when data contains categorical value")
    
    var_to_impute_list = ['sepal width (cm)']
    imputation_class_vars = ['species', 'sepal length (cm)']
    max_donor_use = 5

    for var_to_impute in var_to_impute_list:
        # Preprocess data to store donor information
        df_data_donor_preprocessed = data_preprocessing_store_donor_info(
            iris_data, var_to_impute, imputation_class_vars
        )
        # Perform nearest neighbor imputation
        result = nearest_neighbour_imputation(
            data=df_data_donor_preprocessed,
            var_to_impute=var_to_impute,
            imputation_class_vars=imputation_class_vars,
            missing_values_to_impute=[-1],
            bucket_imputation=False,
            max_donor_use=max_donor_use,
            userid_col_name='userid'
        )
    
    # Assert that there are no missing values after imputation
    assert not result['sepal width (cm)_post_imputation'].isna().any()

"""
Test Case 4
-----------------------------------------------------------------
Objective: Evaluate the accuracy of the imputation process by comparing the imputed values 
with the original values in the presence of manually introduced missing values.

Expected Outcome:
The mean absolute error between the imputed values and the original values 
should be less than the defined threshold, confirming the accuracy of the imputation method.
"""

def test_accuracy(iris_data, iris_data_imputed):
    """Test case to evaluate imputation accuracy."""
    print("\nTest Case 4: Evaluate the accuracy of the imputation process")

    # Subset data for testing
    iris_data_imputed = iris_data_imputed.head(10)
    iris_data = iris_data.head(10)

    # Create a copy of imputed data
    iris_data_imputed_copy = iris_data_imputed.copy()
    iris_data_imputed_copy['sepal width (cm)'] = iris_data_imputed_copy['sepal width (cm)_post_imputation']

    # Align columns with original data
    iris_data_imputed_copy = iris_data_imputed_copy[iris_data.columns]

    # Number of values to change
    n_changes = 3

    # Randomly select indices to change in column 'sepal width (cm)'
    np.random.seed(0)
    indices_to_change = np.random.choice(iris_data_imputed_copy.index, size=n_changes, replace=False)

    # Change selected values to -1
    iris_data_imputed_copy.loc[indices_to_change, 'sepal width (cm)'] = -1

    var_to_impute_list = ['sepal width (cm)']
    imputation_class_vars = ['species', 'sepal length (cm)']
    max_donor_use = 5
    threshold = 5

    for var_to_impute in var_to_impute_list:
        # Preprocess data to store donor information
        df_data_donor_preprocessed = data_preprocessing_store_donor_info(
            iris_data_imputed_copy, var_to_impute, imputation_class_vars
        )
        # Perform nearest neighbor imputation
        result = nearest_neighbour_imputation(
            data=df_data_donor_preprocessed,
            var_to_impute=var_to_impute,
            imputation_class_vars=imputation_class_vars,
            missing_values_to_impute=[-1],
            bucket_imputation=False,
            max_donor_use=max_donor_use,
            userid_col_name='userid'
        )

    # Calculate absolute error between imputed and original values
    imputed_values = result['sepal width (cm)_post_imputation']
    original_values = iris_data_imputed['sepal width (cm)_post_imputation']
    absolute_error = np.mean([abs(ov - iv) for ov, iv in zip(original_values, imputed_values)])

    # Assert that mean absolute error is within threshold
    assert absolute_error < threshold, f"Mean absolute error is too high: {absolute_error}"