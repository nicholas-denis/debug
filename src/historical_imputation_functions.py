import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# dummy data to test out donor imputation with
from sklearn.datasets import load_iris
import random

# for nearest neighbour imputation
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.preprocessing import MinMaxScaler

# for type hints
import typing
from typing import List, Set, Dict, Tuple, Union, Optional

# for warning messages
import warnings


########## Historical Imputation Utility Functions ##########


def ratio_imputation(
    avg_current_wave: float, avg_previous_wave: float, val_past_wave: float
) -> float:
    """Function for applying ratio imputation.

    Parameters
    ----------

    avg_current_wave : float
        Average value in the current wave.

    avg_previous_wave : float
        Average value in the previous wave.

    val_past_wave : float
        Value in the previous wave.

    Returns
    -------

    (avg_current_wave / avg_previous_wave) * val_past_wave : float
        Imputed value using ratio imputation.

    """
    return (avg_current_wave / avg_previous_wave) * val_past_wave


def construct_house_loan_donor_pool_one_house(
    data: pd.DataFrame,
    var_housing_loan_current_wave: str,
    var_housing_loan_previous_wave: str,
    var_number_properties_current_wave: str,
    var_number_properties_previous_wave: str,
    var_hdb_indicator_current_wave: str,
    var_hdb_indicator_previous_wave: str,
    missing_values_to_impute: List[int],
) -> pd.DataFrame:
    """Function for constructing donor pool for historical imputation of house loan values, one house only.

    Pandas DataFrame output of this function feedsin into the historical_imputation_housing function.
    These

    Parameters
    ----------

    data : pd.Dataframe
        Input data for housing loan historical imputation.

    var_housing_loan_current_wave : str
        Name of housing loan variable in current wave to be imputed.

    var_housing_loan_previous_wave : str
        Name of housing loan variable in previous wave.

    var_number_properties_current_wave : str
        Name of number of properties variable in current wave.

    var_number_properties_previous_wave : str
        Name of number of properties variable in previous wave.

    var_hdb_indicator_current_wave : str
        Name of HDB housing indicator variable in current wave.

    var_hdb_indicator_previous_wave : str
        Name of HDB housing indicator variable in previous wave.

    missing_values_to_impute : list[int]
        List of missing values to impute for.

    Returns
    -------

    house_loan_donor_pool_one_house : pd.DataFrame
        Pandas DataFrame that has average housing loan and number of observations
        per imputation class for the historical imputation of housing value in the case
        that respondent only has 1 house.
    """

    house_loan_donor_pool_one_house = (
        data[
            (~data[var_housing_loan_current_wave].isin(missing_values_to_impute))
            & (~data[var_housing_loan_previous_wave].isin(missing_values_to_impute))
            & (data[var_number_properties_current_wave] == 1)
            & (data[var_number_properties_previous_wave] == 1)
        ]
        .groupby([var_hdb_indicator_previous_wave])[
            [var_housing_loan_current_wave, var_housing_loan_previous_wave]
        ]
        .agg(["count", "mean"])
        .reset_index()
    )

    house_loan_donor_pool_one_house.columns = [
        "_".join(col).rstrip("_") for col in house_loan_donor_pool_one_house.columns
    ]

    return house_loan_donor_pool_one_house


def construct_house_loan_donor_pool_multiple_houses(
    data: pd.DataFrame,
    var_housing_loan_current_wave: str,
    var_housing_loan_previous_wave: str,
    var_number_properties_current_wave: str,
    var_number_properties_previous_wave: str,
    var_hdb_indicator_current_wave: str,
    var_hdb_indicator_previous_wave: str,
    missing_values_to_impute: List[int],
) -> pd.DataFrame:
    """Function for constructing donor pool for historical imputation of house loan values for multiple homeowners.

    Pandas DataFrame output of this function feedsin into the historical_imputation_housing function.

    Parameters
    ----------

    data : pd.Dataframe
        Input data for housing loan historical imputation.

    var_housing_loan_current_wave : str
        Name of housing loan variable in current wave to be imputed.

    var_housing_loan_previous_wave : str
        Name of housing loan variable in previous wave.

    var_number_properties_current_wave : str
        Name of number of properties variable in current wave.

    var_number_properties_previous_wave : str
        Name of number of properties variable in previous wave.

    var_hdb_indicator_current_wave : str
        Name of HDB housing indicator variable in current wave.

    var_hdb_indicator_previous_wave : str
        Name of HDB housing indicator variable in previous wave.

    missing_values_to_impute : list[int]
        List of missing values to impute for.

    Returns
    -------

    house_loan_donor_pool_multiple : pd.DataFrame
        Pandas DataFrame that has housing loan values of respondents who meet the criteria for being
        in the donor pool for imputing housing loan values for individuals with multiple houses.
    """

    house_loan_donor_pool_multiple = data[
        (~data[var_housing_loan_current_wave].isin(missing_values_to_impute))
        & (~data[var_housing_loan_previous_wave].isin(missing_values_to_impute))
        & (~data[var_number_properties_current_wave].isin(missing_values_to_impute))
        & (data[var_number_properties_current_wave] > 1)
        & (
            data[var_number_properties_previous_wave]
            >= data[var_number_properties_current_wave]
        )
    ][[var_housing_loan_current_wave, var_housing_loan_previous_wave]]

    return house_loan_donor_pool_multiple


def construct_incomes_donor_pool(
    data: pd.DataFrame,
    var_current_wave: str,
    var_previous_wave: str,
    var_number_jobs_current_wave: str,
    var_number_jobs_previous_wave: str,
    var_ssic_current_wave: str,
    missing_values_to_impute: List[int],
) -> pd.DataFrame:
    """Function for constructing donor pool for historical imputation of house loan values, one house only.

    Pandas DataFrame output of this function feedsin into the historical_imputation_housing function.
    These

    Parameters
    ----------

    data : pd.Dataframe
        Input data for housing loan historical imputation.

    var_current_wave : str
        Name of variable to be imputed in the current wave.

    var_previous_wave : str
        Name of variable to be imputed in the previous wave.

    var_number_jobs_current_wave : str
        Name of number of jobs variable in current wave.

    var_number_jobs_previous_wave : str
        Name of number of jobs variable in current wave.

    var_ssic_current_wave : str
        Name of SSIC variable in current wave.

    missing_values_to_impute : list[int]
        List of missing values to impute for.

    Returns
    -------

    imputation_group : pd.DataFrame
        Pandas DataFrame that has average value of variable to be imputed and number of observations
        per imputation class for the historical imputation of income / earnings related variables.
    """

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

    imputation_group.columns = [
        "_".join(col).rstrip("_") for col in imputation_group.columns
    ]

    return imputation_group


########## Historical Imputation Functions ##########

##### Housing Loan Imputation #####


def housing_historical_imputation_by_row(
    row: pd.DataFrame,
    var_current_wave: str,
    var_previous_wave: str,
    var_number_properties: str,
    var_hdb_indicator_current_wave: str,
    var_hdb_indicator_previous_wave: str,
    imputation_group_one_house: pd.DataFrame,
    imputation_group_multi_house: pd.DataFrame,
    missing_values_to_impute: list[int],
    min_records_req: int,
) -> pd.DataFrame:
    """Historical imputation function for housing loan value, applied to a single row.

    Row-wise function that will be applied to the entire column in the
    historical_imputation_housing function.

    Parameters
    ----------

    row : pd.Dataframe
        A single row of a Pandas DataFrame.

    var_current_wave : str
        Name of variable in current wave to be imputed.

    var_previous_wave : str
        Name of variable in previous wave to be used for ratio imputation.

    var_number_properties : str
        Name of number of properties variable in current wave.

    var_hdb_indicator_current_wave : str
        Name of HDB housing indicator variable in current wave.

    var_hdb_indicator_previous_wave : str
        Name of HDB housing indicator variable in previous wave.

    imputation_group_one_house : pd.DataFrame
        Pandas DataFrame containing the imputation groups for respondents with 1 house.
        Columns of this DataFrame are: HDB housing indicator, which defines the
        imputation classes, count of observations per imputation class, and
        mean housing loan values in the current and previous waves.

    imputation_group_multi_house : pd.DataFrame
        Pandas DataFrame containing the imputation groups for respondents with multiple houses.
        Columns of this DataFrame are: mean housing loan values in the current and previous waves.

    missing_values_to_impute : list[int]
        List of missing values to impute for.

    min_records_req : int
        Minimum number of non-missing housing loan values for previous and current waves in
        imputation class.

    Returns
    -------

    row_post_imputation : pd.DataFrame
        Pandas DataFrame row of output data, with imputation applied to a row of the
        housing loan value variable.
    """

    # make copy of row to apply imputation
    row_post_imputation = row.copy()

    # get name of post imputation column
    post_imputation_col_name = f"{var_current_wave}_post_imputation"
    imputed_val_col_name = f"{var_current_wave}_imputed_value"

    # get names of counts and mean / average columns
    record_count_current_wave = f"{var_current_wave}_count"
    record_count_previous_wave = f"{var_previous_wave}_count"
    avg_current_wave = f"{var_current_wave}_mean"
    avg_previous_wave = f"{var_previous_wave}_mean"

    # only impute if:
    # var_current_wave is missing for row
    # var_previous_wave was not imputed
    # same housing for previous and current waves
    if pd.isna(row_post_imputation[post_imputation_col_name]) & (
        ~pd.isna(row_post_imputation[var_previous_wave])
    ):

        # if there is only 1 property
        if row_post_imputation[var_number_properties] == 1:

            # only impute if there are sufficient records based on imputation parameters
            if (
                imputation_group_one_house[
                    imputation_group_one_house[var_hdb_indicator_previous_wave]
                    == row_post_imputation[var_hdb_indicator_current_wave]
                ][[record_count_current_wave, record_count_previous_wave]].values
                > min_records_req
            ).all():

                # average housing loan of current wave
                avg_val_current_wave = imputation_group_one_house[
                    imputation_group_one_house[var_hdb_indicator_previous_wave]
                    == row_post_imputation[var_hdb_indicator_current_wave]
                ][avg_current_wave].values[0]

                # average housing loan of previous wave
                avg_val_previous_wave = imputation_group_one_house[
                    imputation_group_one_house[var_hdb_indicator_previous_wave]
                    == row_post_imputation[var_hdb_indicator_current_wave]
                ][avg_previous_wave].values[0]

                # reported value in previous wave
                reported_val_previous_wave = row_post_imputation[var_previous_wave]

                # impute using historical ratio imputation
                imputed_val = ratio_imputation(
                    avg_val_current_wave,
                    avg_val_previous_wave,
                    reported_val_previous_wave,
                )
                row_post_imputation[post_imputation_col_name] = imputed_val
                row_post_imputation[imputed_val_col_name] = imputed_val

                # change imputation flag to 1
                imputation_flag_col_name = f"{var_current_wave}_imputation_flag"
                row_post_imputation[imputation_flag_col_name] = 1

                # set imputation type to Historical
                imputation_type_col_name = f"{var_current_wave}_imputation_type"
                row_post_imputation[imputation_type_col_name] = "Historical"

        # if there are multiple properties
        elif row_post_imputation[var_number_properties] > 1:

            # only impute if there are sufficient records based on imputation parameters
            if imputation_group_multi_house.shape[0] > min_records_req:

                # average housing loan of current wave
                avg_val_current_wave = np.mean(
                    imputation_group_multi_house[var_current_wave]
                )

                # average housing loan of previous wave
                avg_val_previous_wave = np.mean(
                    imputation_group_multi_house[var_previous_wave]
                )

                # reported value in previous wave
                reported_val_previous_wave = row_post_imputation[var_previous_wave]

                # impute using historical ratio imputation
                imputed_val = ratio_imputation(
                    avg_val_current_wave,
                    avg_val_previous_wave,
                    reported_val_previous_wave,
                )
                row_post_imputation[post_imputation_col_name] = imputed_val
                row_post_imputation[imputed_val_col_name] = imputed_val

                # change imputation flag to 1
                imputation_flag_col_name = f"{var_current_wave}_imputation_flag"
                row_post_imputation[imputation_flag_col_name] = 1

                # set imputation type to Historical
                imputation_type_col_name = f"{var_current_wave}_imputation_type"
                row_post_imputation[imputation_type_col_name] = "Historical"

    return row_post_imputation


def historical_imputation_housing(
    data: pd.DataFrame,
    var_current_wave: str,
    var_previous_wave: str,
    var_number_properties: str,
    var_hdb_indicator_current_wave: str,
    var_hdb_indicator_previous_wave: str,
    imputation_group_one_house: pd.DataFrame,
    imputation_group_multi_house: pd.DataFrame,
    missing_values_to_impute: list[int],
    min_records_req: int = 30,
) -> pd.DataFrame:
    """Historical imputation function for housing loan value for entire data set.

    Wrapper function to housing_historical_imputation_by_row function.
    Applies historical imputation to entire housing loan value column.

    Parameters
    ----------

    data : pd.Dataframe
        Pandas DataFrame of input data.

    var_current_wave : str
        Name of variable in current wave to be imputed.

    var_previous_wave : str
        Name of variable in previous wave to be used for ratio imputation.

    var_number_properties : str
        Name of number of properties variable in current wave.

    var_hdb_indicator_current_wave : str
        Name of HDB housing indicator variable in current wave.

    var_hdb_indicator_previous_wave : str
        Name of HDB housing indicator variable in previous wave.

    imputation_group_one_house : pd.DataFrame
        Pandas DataFrame containing the imputation groups for respondents with 1 house.
        Columns of this DataFrame are: HDB housing indicator, which defines the
        imputation classes, count of observations per imputation class, and
        mean housing loan values in the current and previous waves.

    imputation_group_multi_house : pd.DataFrame
        Pandas DataFrame containing the imputation groups for respondents with multiple houses.
        Columns of this DataFrame are: mean housing loan values in the current and previous waves.

    missing_values_to_impute : list[int]
        List of missing values to impute for.

    min_records_req : int (default = 30)
        Minimum number of non-missing housing loan values for previous and current waves in
        imputation class.

    Returns
    -------

    data_post_imputation : pd.DataFrame
        Pandas DataFrame of output data, with imputations applied to housing loan value.
    """

    # make copy of data to do imputations on
    data_post_imputation = data.copy()

    # get name of post imputation column
    post_imputation_col_name = f"{var_current_wave}_post_imputation"

    # replace missing values in post-imputation variable and its previous wave's variable columns with NA
    data_post_imputation[post_imputation_col_name] = data_post_imputation[
        post_imputation_col_name
    ].replace(missing_values_to_impute, pd.NA)
    data_post_imputation[var_previous_wave] = data_post_imputation[
        var_previous_wave
    ].replace(missing_values_to_impute, pd.NA)

    # Do historical imputation for housing variables
    data_post_imputation = data_post_imputation.apply(
        lambda row: housing_historical_imputation_by_row(
            row,
            var_current_wave,
            var_previous_wave,
            var_number_properties,
            var_hdb_indicator_current_wave,
            var_hdb_indicator_previous_wave,
            imputation_group_one_house,
            imputation_group_multi_house,
            missing_values_to_impute,
            min_records_req,
        ),
        axis=1,
    )

    return data_post_imputation


##### Work Income, Bonuses, Earnings Imputation #####


def income_historical_imputation_by_row(
    row: pd.DataFrame,
    var_current_wave: str,
    var_previous_wave: str,
    num_jobs_current_wave: str,
    num_jobs_previous_wave: str,
    ssic: str,
    imputation_group: pd.DataFrame,
    missing_values_to_impute: list[int],
    min_records_req: int,
) -> pd.DataFrame:
    """Historical imputation function for earnings-related variables, applied to a single row.

    Row-wise function that will be applied to the entire column in the
    historical_imputation_income function.

    Parameters
    ----------

    row : pd.Dataframe
        A single row of a Pandas DataFrame.

    var_current_wave : str
        Name of variable in current wave to be imputed.

    var_previous_wave : str
        Name of variable in previous wave to be used for ratio imputation.

    num_jobs_current_wave : str
        Number of jobs / businesses held by respondent in the current wave.

    num_jobs_previous_wave : str
        Number of jobs / businesses held by respondent in the previous wave.

    ssic : str
        SSIC of respondent in current wave.

    imputation_group : pd.DataFrame
        Pandas DataFrame containing the imputation groups, which are defined by
        the job's / business' Singapore Standard Industrial Classification (SSIC).
        Columns of this DataFrame are: SSIC, which defines the imputation classes,
        count of observations per imputation class, and mean earnings values
        in the current and previous waves.

    missing_values_to_impute : list[int]
        List of missing values to impute for.

    min_records_req : int
        Minimum number of non-missing earnings value for previous and current waves in
        imputation class.

    Returns
    -------

    row_post_imputation : pd.DataFrame
        data_post_imputation : pd.DataFrame
        Pandas DataFrame of output data, with imputations applied to housing loan value.
    """

    # make copy of row to apply imputation
    row_post_imputation = row.copy()

    # get names of post imputation columns
    post_imputation_col_name = f"{var_current_wave}_post_imputation"
    imputed_val_col_name = f"{var_current_wave}_imputed_value"

    # get names of counts and mean / average columns
    record_count_current_wave = f"{var_current_wave}_count"
    record_count_previous_wave = f"{var_previous_wave}_count"
    avg_current_wave = f"{var_current_wave}_mean"
    avg_previous_wave = f"{var_previous_wave}_mean"

    # only impute if:
    # var_current_wave is missing for row
    # var_previous_wave was not imputed
    # same number of jobs in previous and current waves
    if (
        pd.isna(row_post_imputation[post_imputation_col_name])
        & (~pd.isna(row_post_imputation[var_previous_wave]))
        & (
            row_post_imputation[num_jobs_current_wave]
            == row_post_imputation[num_jobs_previous_wave]
        )
    ):

        # only impute if there are sufficient records based on imputation parameters
        if (
            imputation_group[imputation_group[ssic] == row_post_imputation[ssic]][
                [record_count_current_wave, record_count_previous_wave]
            ].values
            > min_records_req
        ).all():

            # average value of current wave's imputation group
            avg_val_current_wave = imputation_group[
                imputation_group[ssic] == row_post_imputation[ssic]
            ][avg_current_wave]

            # average value of previous wave's imputation group
            avg_val_previous_wave = imputation_group[
                imputation_group[ssic] == row_post_imputation[ssic]
            ][avg_previous_wave]

            # reported value in previous wave
            reported_val_previous_wave = row_post_imputation[var_previous_wave]

            # impute using historical ratio imputation
            imputed_val = ratio_imputation(
                avg_val_current_wave, avg_val_previous_wave, reported_val_previous_wave
            ).values[0]
            row_post_imputation[post_imputation_col_name] = imputed_val
            row_post_imputation[imputed_val_col_name] = imputed_val

            # change imputation flag to 1
            # check with clients: do we want to name flag variable as f'{var_current_wave}_c'?
            imputation_flag_col_name = f"{var_current_wave}_imputation_flag"
            row_post_imputation[imputation_flag_col_name] = 1

            # set imputation type to Historical
            imputation_type_col_name = f"{var_current_wave}_imputation_type"
            row_post_imputation[imputation_type_col_name] = "Historical"

    return row_post_imputation


def historical_imputation_income(
    data: pd.DataFrame,
    var_current_wave: str,
    var_previous_wave: str,
    num_jobs_current_wave: str,
    num_jobs_previous_wave: str,
    ssic: str,
    imputation_group: pd.DataFrame,
    missing_values_to_impute: list[int],
    min_records_req: int = 30,
):
    """Historical imputation function for earnings-related variable for entire data set.

    Wrapper function to income_historical_imputation_by_row function.
    Applies historical imputation to entire earnings-related variable column.

    Parameters
    ----------

    data : pd.Dataframe
        Pandas DataFrame of input data.

    var_current_wave : str
        Name of variable in current wave to be imputed.

    var_previous_wave : str
        Name of variable in previous wave to be used for ratio imputation.

    num_jobs_current_wave : str
        Number of jobs / businesses held by respondent in the current wave.

    num_jobs_previous_wave : str
        Number of jobs / businesses held by respondent in the previous wave.

    ssic : str
        SSIC of respondent in current wave.

    imputation_group : pd.DataFrame
        Pandas DataFrame containing the imputation groups, which are defined by
        the job's / business' Singapore Standard Industrial Classification (SSIC).
        Columns of this DataFrame are: SSIC, which defines the imputation classes,
        count of observations per imputation class, and mean earnings values
        in the current and previous waves.

    missing_values_to_impute : list[int]
        List of missing values to impute for.

    min_records_req : int (default = 30)
        Minimum number of non-missing earnings values for previous and current waves in
        imputation class.

    Returns
    -------

    data_post_imputation : pd.DataFrame
        Pandas DataFrame of output data, with imputations applied to earnings-related variable.
    """

    # TODO: check parameters for errors outside of function

    # make copy of data to do imputations on
    data_post_imputation = data.copy()

    # get name of post imputation column
    post_imputation_col_name = f"{var_current_wave}_post_imputation"

    # replace missing values in post-imputation variable and its previous wave's variable columns with NA
    data_post_imputation[post_imputation_col_name] = data_post_imputation[
        post_imputation_col_name
    ].replace(missing_values_to_impute, pd.NA)
    data_post_imputation[var_previous_wave] = data_post_imputation[
        var_previous_wave
    ].replace(missing_values_to_impute, pd.NA)

    # Do historical imputation for income variables
    data_post_imputation = data_post_imputation.apply(
        lambda row: income_historical_imputation_by_row(
            row,
            var_current_wave,
            var_previous_wave,
            num_jobs_current_wave,
            num_jobs_previous_wave,
            ssic,
            imputation_group,
            missing_values_to_impute,
            min_records_req,
        ),
        axis=1,
    )

    return data_post_imputation
