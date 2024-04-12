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


########## Data Formatting Functions ##########


def data_preprocessing_store_donor_info(
    data: pd.DataFrame,
    var_to_impute: str,
    imputation_class_vars: List[str],
    knn_distance: bool = True,
) -> pd.DataFrame:
    """Function that pre-processes data set to include variables pertaining to imputation.

    Variables added include: duplicate column of the variable to impute with suffix '_post_imputation',
    imputation flag, imputation type (historical, KNN), imputed value, donor user ID,
    donor use count, donor distance, and donor imputation class details.

    Donor-specific variables only apply to KNN imputation.

    Default values for these variables are:
        - imputation flag: 0
        - imputation type: '' (blank string)
        - imputed value: NA
        - donor user ID: -1
        - donor use count: 0
        - donor distance: NA
        - donor imputation class details: NA
    These variables are prefixed by the name of the variable to be imputed.

    Parameters
    ----------

    data : pd.DataFrame
        Pandas DataFrame of input data.

    var_to_impute : str
        Name of variable to impute.

    imputation_class_vars : list of str
        Names of imputation class variables.

    knn_distance : boolean (default = True)
        Distance of donor.

    Returns
    -------

    data_prepped : pd.DataFrame
        Pandas DataFrame of input data with additional columns pertaining to imputation details.
    """

    data_prepped = data.copy()

    # Create duplicate of variable to impute
    var_to_impute_post_imputation_name = f"{var_to_impute}_post_imputation"
    data_prepped[var_to_impute_post_imputation_name] = data_prepped[var_to_impute]

    # Create imputation flag for variable to impute.
    imputation_flag_col_name = f"{var_to_impute}_imputation_flag"
    data_prepped[imputation_flag_col_name] = 0

    # Create column that denotes type of imputation (KNN, historical).
    imputation_type_col_name = f"{var_to_impute}_imputation_type"
    data_prepped[imputation_type_col_name] = ""

    # Create column that contains donated, imputed value.
    # If imputation not required or no donor found, then imputed_value is NaN.
    imputed_val_col_name = f"{var_to_impute}_imputed_value"
    data_prepped[imputed_val_col_name] = pd.NA

    # Create column that stores userid of donor, if applicable.
    # If imputation not required or no donor found, then userid = -1.
    donor_userid_col_name = f"{var_to_impute}_donor_userid"
    data_prepped[donor_userid_col_name] = -1

    # Create donor count variable.
    donor_count_col_name = f"{var_to_impute}_donor_count"
    data_prepped[donor_count_col_name] = 0

    # Create columns for donor characteristics: values of imputation class variables.
    for var in imputation_class_vars:
        donor_imputation_class_var = f"{var_to_impute}_donor_{var}"
        data_prepped[donor_imputation_class_var] = pd.NA

    # Create column for donor distance.
    # If donor not obtained via KNN, then distance will default to NaN.
    donor_dist_col_name = f"{var_to_impute}_donor_distance"
    data_prepped[donor_dist_col_name] = pd.NA

    return data_prepped


########## Bucketing and Un-Bucketing Functions ##########


def continuous_to_range(
    data: pd.DataFrame,
    column: str,
    range_interval_index: pd.IntervalIndex,
    range_label_dict: Dict,
) -> pd.Series:
    """Function that converts continuous values to bucketed ranges.

    Parameters
    ----------

    data : pd.DataFrame
        Pandas DataFrame of input data.

    column : str
        Name of column to convert continuous values to bucketed ranges.

    range_interval_index : pd.IntervalIndex
        Pandas Interval Index type that defines the buckets.
        Note that the left bound is exclusive, right bound is inclusive.

    range_label_dict : dict
        Dictionary with Pandas Interval Indices as keys and corresponding integer labels as values.

    Returns
    -------

    continuous_to_bucketed_series: pd.Series
        Pandas Series of column with bucketed ranges.
    """

    # returns a pandas Series that converts continuous values to buckets

    continuous_to_bucketed_series = pd.cut(
        data[column],
        range_interval_index,
        labels=range(1, len(range_interval_index) + 1),
    )
    continuous_to_bucketed_series = continuous_to_bucketed_series.map(range_label_dict)
    return continuous_to_bucketed_series


def range_to_random_continuous(
    data: pd.DataFrame, column: str, range_to_continuous_dict: Dict
) -> pd.Series:
    """Function that converts bucketed ranges to a randomly selected value within the range.

    Parameters
    ----------

    data : pd.DataFrame
        Pandas DataFrame of input data.

    column : str
        Name of column to convert continuous values to bucketed ranges.

    range_to_continuous_dict : dict
        Dictionary with integer labels of buckets as keys and tuples of range .

    Returns
    -------

    bucketed_to_continuous_series: pd.Series
        Pandas Series of column with numbers randomly selected from bucketed range.
    """

    bucketed_to_continuous_series = pd.Series(pd.NA, index=data.index)

    for row_num, val in enumerate(data[column]):
        if not pd.isna(val):
            bucketed_to_continuous_series[row_num] = random.randint(
                *range_to_continuous_dict[val]
            )  # .astype(float)
        else:
            bucketed_to_continuous_series[row_num] = pd.NA

    return bucketed_to_continuous_series


# The following list of tuples defines the buckets for monetary columns ex. Purchase_Val_Amt, Value_Hse_Loan.
# Note that the lower bound is exclusive and the upper bound is inclusive.
# The lower bound of the first bucket is to allow for zeros.
monetary_buckets = [
    (-0.001, 20),
    (20, 40),
    (40, 60),
    (60, 80),
    (80, 100),
    (100, 200),
    (200, 300),
    (300, 400),
    (400, 500),
    (500, 600),
    (600, 700),
    (700, 800),
    (800, 900),
    (900, 1000),
    (1000, 1500),
    (1500, 2000),
    (2000, 2500),
    (2500, 3000),
    (3000, 3500),
    (3500, 4000),
    (4000, 4500),
    (4500, 5000),
    (5000, 5500),
    (5500, 6000),
    (6000, 6500),
    (6500, 7000),
    (7000, 7500),
    (7500, 8000),
    (8000, 8500),
    (8500, 9000),
    (9000, 9500),
    (9500, 10000),
    (10000, 15000),
    (15000, 20000),
    (20000, 25000),
    (25000, 30000),
    (30000, 35000),
    (35000, 40000),
    (40000, 45000),
    (45000, 50000),
    (50000, 55000),
    (55000, 60000),
    (60000, 65000),
    (65000, 70000),
    (70000, 75000),
    (75000, 80000),
    (80000, 85000),
    (85000, 90000),
    (90000, 95000),
    (95000, 100000),
    (100000, 125000),
    (125000, 150000),
    (150000, 175000),
    (175000, 200000),
    (200000, 225000),
    (225000, 250000),
    (250000, 275000),
    (275000, 300000),
    (300000, 325000),
    (325000, 350000),
    (350000, 375000),
    (375000, 400000),
    (400000, 425000),
    (425000, 450000),
    (450000, 475000),
    (475000, 500000),
    (500000, 550000),
    (550000, 600000),
    (600000, 650000),
    (650000, 700000),
    (700000, 750000),
    (750000, 800000),
    (800000, 850000),
    (850000, 900000),
    (900000, 950000),
    (950000, 1000000),
    (1000000, 1250000),
    (1250000, 1500000),
    (1500000, 1750000),
    (1750000, 2000000),
    (2000000, 2500000),
    (2500000, 3000000),
    (3000000, 4000000),
    (4000000, 5000000),
    (5000000, 6000000),
    (6000000, 7000000),
    (7000000, 8000000),
    (8000000, 9000000),
    (9000000, 10000000),
    (10000000, 11000000),
    (11000000, 12000000),
    (12000000, 13000000),
    (13000000, 14000000),
    (14000000, 15000000),
    (15000000, 16000000),
    (16000000, 17000000),
    (17000000, 18000000),
    (18000000, 19000000),
    (19000000, 20000000),
    (20000000, 40000000),
    (40000000, 60000000),
    (60000000, 80000000),
    (80000000, 100000000),
    (100000000, 200000000),
    (200000000, 300000000),
    (300000000, 400000000),
    (400000000, 500000000),
    (500000000, 600000000),
    (600000000, 700000000),
    (700000000, 800000000),
    (800000000, 900000000),
    (900000000, 1000000000),
]

# The following are buckets for variables that are strictly positive values.
# Note that buckets' first bin excludes 0
non_monetary_buckets = monetary_buckets.copy()
non_monetary_buckets[0] = (0, 20)

# Convert buckets to Pandas IntervalIndex
monetary_bins = pd.IntervalIndex.from_tuples(monetary_buckets)
non_monetary_bins = pd.IntervalIndex.from_tuples(non_monetary_buckets)

# Have dictionaries that convert Pandas IntervalIndex to integer labels
monetary_bins_to_enumerated_val = dict(
    zip(monetary_bins, range(1, len(monetary_bins) + 1))
)
non_monetary_bins_to_enumerated_val = dict(
    zip(non_monetary_bins, range(1, len(non_monetary_bins) + 1))
)

# Have dictionary that converts integer labels back to tuple ranges
enumerated_val_to_range = dict(
    zip(range(1, len(non_monetary_buckets) + 1), non_monetary_buckets)
)


########## Nearest Neighbour Imputation ##########


def nearest_neighbour_imputation_categorical(
    data: pd.DataFrame,
    var_to_impute: str,
    imputation_class_vars: List[str],
    categorical_imputation_class_vars: List[str],
    missing_values_to_impute: List[int],
    bucket_imputation: bool,
    bins: Optional[pd.IntervalIndex] = None,
    bins_to_labels: Optional[Dict] = None,
    multiple_possible_class_vars: Optional[List[str]] = None,
    max_donor_use: int = 5,
    userid_col_name: str = "userid",
) -> pd.DataFrame:
    """1-Nearest Neighbour imputation function for categorical imputation class variables.

    Parameters
    ----------

    data : pd.Dataframe
        Pandas DataFrame of input data.

    var_to_impute : str
        Name of variable to be imputed.

    imputation_class_vars : list of strings
        Names of all variables used to define the imputation class.

    categorical_imputation_class_vars : list of strings
        Names of categorical variables used to define the imputation class.

    missing_values_to_impute : list of integers
        List of missing values to impute for.

    bucket_imputation : bool
        Boolean indicator for whether to generate an additional, bucketed version of the imputed column.

    bins : pd.IntervalIndex, optional
        Only required if bucket_imputation == True.
        Pandas Interval Index type that defines the buckets.
        Note that the left bound is exclusive, right bound is inclusive.

    bins_to_labels : dict, optional
        Only required if bucket_imputation == True.
        Dictionary with Pandas Interval Indices as keys and corresponding integer labels as values.

    multiple_possible_class_vars : list of strings, optional
        Names of imputation class variables that can be used in place of the other
        ex. either work income or business earnings can be used as an imputation class variable.
        Note that the function only accomodates for ONE PAIR of interchangeable
        imputation class variables.

    max_donor_use : int, default = 5
        Maximum number of times a donor can be used.

    userid_col_name : str, default = 'userid'
        Name of user ID variable.

    Returns
    -------

    data_post_imputation : pd.DataFrame
        Pandas DataFrame of output data, with imputations applied to variable to be imputed.
    """

    # Make copy of data to implement imputations.
    data_post_imputation = data.copy()

    # get imputation details and donor details' variable names
    imputation_flag_col_name = f"{var_to_impute}_imputation_flag"
    imputation_type_col_name = f"{var_to_impute}_imputation_type"
    post_imputation_col_name = f"{var_to_impute}_post_imputation"
    imputed_val_col_name = f"{var_to_impute}_imputed_value"
    donor_userid_col_name = f"{var_to_impute}_donor_userid"
    donor_count_col_name = f"{var_to_impute}_donor_count"
    donor_characteristics_col_names = [
        f"{var_to_impute}_donor_{var}" for var in imputation_class_vars
    ]
    donor_distance_col_name = f"{var_to_impute}_donor_distance"

    # initialize donor use tracking dictionary
    donor_use_track_dict = {k: 0 for k in data_post_imputation[userid_col_name]}

    # Get list of imputation class variables that are numeric.
    numeric_imputation_class_vars = list(
        set(imputation_class_vars).difference(categorical_imputation_class_vars)
    )

    # Slice data by categorical imputation class variables.
    data_post_imputation_by_slices = {
        slice: sub_df.reset_index(names="original_index")
        for slice, sub_df in data_post_imputation.groupby(
            categorical_imputation_class_vars
        )
    }

    # get column numbers for indexing use later
    data_columns = data_post_imputation_by_slices[
        [k for k in data_post_imputation_by_slices.keys()][0]
    ].columns
    imputation_flag_col_idx = data_columns.get_loc(imputation_flag_col_name)
    imputation_type_col_idx = data_columns.get_loc(imputation_type_col_name)
    post_imputation_col_idx = data_columns.get_loc(post_imputation_col_name)
    imputed_val_col_idx = data_columns.get_loc(imputed_val_col_name)
    donor_userid_col_idx = data_columns.get_loc(donor_userid_col_name)
    donor_count_col_idx = data_columns.get_loc(donor_count_col_name)
    donor_characteristics_col_idxs = [
        data_columns.get_loc(var) for var in donor_characteristics_col_names
    ]
    donor_distance_col_idx = data_columns.get_loc(donor_distance_col_name)

    # initialize min-max scaler
    minmax_scaler = MinMaxScaler()

    # Slice data by categorical imputation class variables.
    data_post_imputation_by_slices = {
        slice: sub_df.reset_index(names="original_index")
        for slice, sub_df in data_post_imputation.groupby(
            categorical_imputation_class_vars
        )
    }

    # Get min-max scaled imputation class variables per slice of
    # categorical imputation class variables.
    imputation_class_vars_transformed_by_slices = {
        slice: minmax_scaler.fit_transform(sub_df[numeric_imputation_class_vars])
        for slice, sub_df in data_post_imputation_by_slices.items()
    }

    # Get BallTree per slice of categorical imputation class variables.
    neighbour_distances_indices_by_slices = {
        slice: BallTree(transformed_sub_df).query(
            transformed_sub_df,
            k=transformed_sub_df.shape[0],
            return_distance=True,
            dualtree=True,
        )
        for slice, transformed_sub_df in imputation_class_vars_transformed_by_slices.items()
    }

    for slice, subset_data in data_post_imputation_by_slices.items():

        for tree_index_num, data_index_num in enumerate(subset_data.index):

            # Search for neighbours if row has missing value
            if pd.notna(subset_data[post_imputation_col_name][data_index_num]) and (
                subset_data[post_imputation_col_name][data_index_num]
                in missing_values_to_impute
            ):

                neighbour_rank = 0
                continue_neighbour_search = True

                while continue_neighbour_search:

                    # Determine which category of the categorical imputation class variable
                    # the observation belongs to in order to select correct BallTree to work with.

                    if (
                        tuple(
                            subset_data.iloc[data_index_num][
                                categorical_imputation_class_vars
                            ]
                        )
                        == slice
                    ):
                        neighbour_distances_indices = (
                            neighbour_distances_indices_by_slices[slice]
                        )

                        donor_idx = neighbour_distances_indices[1][
                            tree_index_num, neighbour_rank
                        ]
                        donor_dist = neighbour_distances_indices[0][
                            tree_index_num, neighbour_rank
                        ]
                        donor_userid = subset_data[userid_col_name][donor_idx]

                        # first check: donor use does not exceed max_donor_use.
                        if donor_use_track_dict[donor_userid] > max_donor_use:
                            neighbour_rank += 1

                        # second check: if closest donor is itself, then continue neighbour search
                        elif donor_idx == data_index_num:
                            neighbour_rank += 1

                        # third check: if donor has missing value, then continue neighbour search
                        elif (
                            data_post_imputation_by_slices[slice][
                                post_imputation_col_name
                            ][donor_idx]
                            in missing_values_to_impute
                        ):
                            neighbour_rank += 1

                        # else, donor is eligible; end neighbour search
                        else:
                            continue_neighbour_search = False

                            # impute value into post-imputation variable
                            data_post_imputation_by_slices[slice].iloc[
                                data_index_num, post_imputation_col_idx
                            ] = data_post_imputation_by_slices[slice][
                                post_imputation_col_name
                            ][
                                donor_idx
                            ]

                            # indicate imputation
                            data_post_imputation_by_slices[slice].iloc[
                                data_index_num, imputation_flag_col_idx
                            ] = 1

                            # indicate type of imputation
                            data_post_imputation_by_slices[slice].iloc[
                                data_index_num, imputation_type_col_idx
                            ] = "KNN"

                            # save imputed value
                            data_post_imputation_by_slices[slice].iloc[
                                data_index_num, imputed_val_col_idx
                            ] = data_post_imputation_by_slices[slice][
                                post_imputation_col_name
                            ][
                                donor_idx
                            ]

                            # save donor's user id
                            data_post_imputation_by_slices[slice].iloc[
                                data_index_num, donor_userid_col_idx
                            ] = donor_userid

                            # increment donor's use count in donor use tracking dictionary
                            donor_use_track_dict[donor_userid] += 1

                            # save donor characteristics
                            for n, var in enumerate(imputation_class_vars):
                                data_post_imputation_by_slices[slice].iloc[
                                    data_index_num, donor_characteristics_col_idxs[n]
                                ] = data_post_imputation_by_slices[slice][var][
                                    donor_idx
                                ]

                            # save donor distance
                            data_post_imputation_by_slices[slice].iloc[
                                data_index_num, donor_distance_col_idx
                            ] = donor_dist

                            # # save sub-dataframe in dictionary tracking data post-imputation
                            # data_post_imputation_by_slices[slice] = subset_data

    for slice, sub_data in data_post_imputation_by_slices.items():
        # reset index for subsets post-imputation
        sub_data = sub_data.set_index("original_index")
        sub_data = sub_data.rename_axis(index=None)

        # update full data set to include post-imputation information
        data_post_imputation.update(sub_data)

    # populate donor use count column
    for row_num, user_id in enumerate(data_post_imputation[donor_userid_col_name]):
        if user_id != -1:
            data_post_imputation.iloc[row_num, (donor_count_col_idx - 1)] = (
                donor_use_track_dict[user_id]
            )

    return data_post_imputation


def nearest_neighbour_imputation(
    data: pd.DataFrame,
    var_to_impute: str,
    imputation_class_vars: List[str],
    missing_values_to_impute: List[int],
    bucket_imputation: bool,
    bins: Optional[pd.IntervalIndex] = None,
    bins_to_labels: Optional[Dict] = None,
    multiple_possible_class_vars: Optional[List[str]] = None,
    max_donor_use: int = 5,
    userid_col_name: str = "userid",
) -> pd.DataFrame:
    """1-Nearest Neighbour imputation function.

    Parameters
    ----------

    data : pd.Dataframe
        Pandas DataFrame of input data.

    var_to_impute : str
        Name of variable to be imputed.

    imputation_class_vars : list of strings
        Names of variables used to define the imputation class.

    missing_values_to_impute : list of integers
        List of missing values to impute for.

    bucket_imputation : bool
        Boolean indicator for whether to generate an additional, bucketed version of the imputed column.

    bins : pd.IntervalIndex, optional
        Only required if bucket_imputation == True.
        Pandas Interval Index type that defines the buckets.
        Note that the left bound is exclusive, right bound is inclusive.

    bins_to_labels : dict, optional
        Only required if bucket_imputation == True.
        Dictionary with Pandas Interval Indices as keys and corresponding integer labels as values.

    multiple_possible_class_vars : list of strings, optional
        Names of imputation class variables that can be used in place of the other
        ex. either work income or business earnings can be used as an imputation class variable.
        Note that the function only accomodates for ONE PAIR of interchangeable
        imputation class variables.

    max_donor_use : int, default = 5
        Maximum number of times a donor can be used.

    userid_col_name : str, default = 'userid'
        Name of user ID variable.

    Returns
    -------

    data_post_imputation : pd.DataFrame
        Pandas DataFrame of output data, with imputations applied to variable to be imputed.
    """
    print("Inside nearest_neighbour_imputation ")
    # make a copy to apply imputation
    data_post_imputation = data.copy()

    # get imputation details and donor details' variable names
    imputation_flag_col_name = f"{var_to_impute}_imputation_flag"
    imputation_type_col_name = f"{var_to_impute}_imputation_type"
    post_imputation_col_name = f"{var_to_impute}_post_imputation"
    imputed_val_col_name = f"{var_to_impute}_imputed_value"
    donor_userid_col_name = f"{var_to_impute}_donor_userid"
    donor_count_col_name = f"{var_to_impute}_donor_count"
    donor_characteristics_col_names = [
        f"{var_to_impute}_donor_{var}" for var in imputation_class_vars
    ]
    donor_distance_col_name = f"{var_to_impute}_donor_distance"

    print("HERE")
    # initialize donor use tracking dictionary
    donor_use_track_dict = {k: 0 for k in data_post_imputation[userid_col_name]}

    # create copy of post-imputation variable column and imputation class variables with missing values replaced with NA
    imputation_class_vars_w_na = [f"{var}_w_na" for var in imputation_class_vars]
    post_imputation_col_name_w_na = f"{post_imputation_col_name}_w_na"
    data_post_imputation[imputation_class_vars_w_na] = data_post_imputation[
        imputation_class_vars
    ].replace(missing_values_to_impute, pd.NA)
    data_post_imputation[post_imputation_col_name_w_na] = data_post_imputation[
        post_imputation_col_name
    ].replace(missing_values_to_impute, pd.NA)

    if multiple_possible_class_vars is not None:
        multiple_possible_class_vars_w_na = [
            f"{var}_w_na" for var in multiple_possible_class_vars
        ]
        data_post_imputation[multiple_possible_class_vars_w_na] = data_post_imputation[
            multiple_possible_class_vars
        ].replace(missing_values_to_impute, pd.NA)

    # initialize min-max scaler
    minmax_scaler = MinMaxScaler()

    
    # subset data to remove blanks and potential donors with missing imputation class variables
    knn_impute_subset = data_post_imputation[
        (data_post_imputation[imputation_class_vars_w_na].notna().all(axis=1))
    ].copy()
    print("knn_impute_subset.shape: ", knn_impute_subset.shape)
    print("knn_impute_subset.head(): ", knn_impute_subset.head(3))

    # Need to determine if any imputation class variables are strings.
    # If so, then call KNN imputation function with categorical variable.
    categorical_imputation_class_vars = list(
        knn_impute_subset[imputation_class_vars].columns[
            knn_impute_subset[imputation_class_vars].dtypes == "object"
        ]
    )
    print("categorical_imputation_class_vars: ", categorical_imputation_class_vars)
    print("about to go into block...")
    if categorical_imputation_class_vars:
        print("inside if block")
        data_post_imputation = nearest_neighbour_imputation_categorical(
            data=knn_impute_subset,
            var_to_impute=var_to_impute,
            imputation_class_vars=imputation_class_vars,
            categorical_imputation_class_vars=categorical_imputation_class_vars,
            missing_values_to_impute=missing_values_to_impute,
            bucket_imputation=bucket_imputation,
            bins=bins,
            bins_to_labels=bins_to_labels,
            multiple_possible_class_vars=multiple_possible_class_vars,
            max_donor_use=max_donor_use,
            userid_col_name=userid_col_name,
        )

    elif multiple_possible_class_vars is None:
        print("Inside elif block")
        # subset data to remove blanks and potential donors with missing imputation class variables
        knn_impute_subset = data_post_imputation[
            (data_post_imputation[imputation_class_vars_w_na].notna().all(axis=1))
        ].copy()

        # reset index of subset, but retain the old index to facilitate updating the full imput data set later
        knn_impute_subset = knn_impute_subset.reset_index(names="original_index")

        # get column numbers for indexing use later
        data_columns = knn_impute_subset.columns
        imputation_flag_col_idx = data_columns.get_loc(imputation_flag_col_name)
        imputation_type_col_idx = data_columns.get_loc(imputation_type_col_name)
        post_imputation_col_idx = data_columns.get_loc(post_imputation_col_name)
        imputed_val_col_idx = data_columns.get_loc(imputed_val_col_name)
        donor_userid_col_idx = data_columns.get_loc(donor_userid_col_name)
        donor_count_col_idx = data_columns.get_loc(donor_count_col_name)
        donor_characteristics_col_idxs = [
            data_columns.get_loc(var) for var in donor_characteristics_col_names
        ]
        donor_distance_col_idx = data_columns.get_loc(donor_distance_col_name)

        # TODO: FIx
        print("About to test: knn_impute_subset[imputation_class_vars_w_na]", knn_impute_subset[imputation_class_vars_w_na])
        print(type(knn_impute_subset[imputation_class_vars_w_na]))
        print(knn_impute_subset[imputation_class_vars_w_na].shape)


        # Need to min-max scale imputation class variables before determining nearest neighbours.
        imputation_class_vars_transformed = minmax_scaler.fit_transform(
            knn_impute_subset[imputation_class_vars_w_na]
        )

        # Use scikit-learn's BallTree function to get nearest neighbours and their indices.
        tree = BallTree(imputation_class_vars_transformed)
        neighbour_distances_indices = tree.query(
            imputation_class_vars_transformed,
            k=knn_impute_subset.shape[0],
            return_distance=True,
            dualtree=True,
        )

        for tree_index_num, data_index_num in enumerate(knn_impute_subset.index):

            # tree_index_num tracks the rows of the BallTree / distance + donor index array
            # data_index_num tracks the row of knn_impute_subset, the subset of the data
            # that does not contain missingness in imputation class variables

            # Search for neighbours if row has missing value
            # First condition disregards blanks.
            # Second condition filters for rows that contain a missing value as defined by missing_values_to_impute.
            # condition formerly pd.isna(data_post_imputation[post_imputation_col_name][data_index_num])
            if pd.notna(
                knn_impute_subset[post_imputation_col_name][data_index_num]
            ) and (
                knn_impute_subset[post_imputation_col_name][data_index_num]
                in missing_values_to_impute
            ):

                neighbour_rank = 0
                continue_neighbour_search = True

                while continue_neighbour_search:

                    donor_idx = neighbour_distances_indices[1][
                        tree_index_num, neighbour_rank
                    ]
                    donor_dist = neighbour_distances_indices[0][
                        tree_index_num, neighbour_rank
                    ]
                    donor_userid = knn_impute_subset[userid_col_name][donor_idx]

                    # first check: donor use does not exceed max_donor_use.
                    if donor_use_track_dict[donor_userid] > max_donor_use:
                        neighbour_rank += 1

                    # second check: if closest donor is itself, then continue neighbour search
                    elif donor_idx == data_index_num:
                        neighbour_rank += 1

                    # third check: if donor has missing value, then continue neighbour search
                    elif (
                        knn_impute_subset[post_imputation_col_name][donor_idx]
                        in missing_values_to_impute
                    ):
                        neighbour_rank += 1

                    # else, donor is eligible; end neighbour search
                    else:
                        continue_neighbour_search = False

                        # impute value into post-imputation variable
                        knn_impute_subset.iloc[
                            data_index_num, post_imputation_col_idx
                        ] = knn_impute_subset[post_imputation_col_name][donor_idx]

                        # indicate imputation
                        knn_impute_subset.iloc[
                            data_index_num, imputation_flag_col_idx
                        ] = 1

                        # indicate type of imputation
                        knn_impute_subset.iloc[
                            data_index_num, imputation_type_col_idx
                        ] = "KNN"

                        # save imputed value
                        knn_impute_subset.iloc[data_index_num, imputed_val_col_idx] = (
                            knn_impute_subset[post_imputation_col_name][donor_idx]
                        )

                        # save donor's user id
                        knn_impute_subset.iloc[data_index_num, donor_userid_col_idx] = (
                            donor_userid
                        )

                        # increment donor's use count in donor use tracking dictionary
                        donor_use_track_dict[donor_userid] += 1

                        # save donor characteristics
                        for n, var in enumerate(imputation_class_vars):
                            knn_impute_subset.iloc[
                                data_index_num, donor_characteristics_col_idxs[n]
                            ] = knn_impute_subset[var][donor_idx]

                        # save donor distance
                        knn_impute_subset.iloc[
                            data_index_num, donor_distance_col_idx
                        ] = donor_dist

                        # populate donor use count column
                        for row_num, user_id in enumerate(
                            knn_impute_subset[donor_userid_col_name]
                        ):
                            if user_id != -1:
                                knn_impute_subset.iloc[row_num, donor_count_col_idx] = (
                                    donor_use_track_dict[user_id]
                                )

                        # bucket imputed column if bucket_imputation == True
                        if bucket_imputation:
                            bucketed_post_imputation_col_name = (
                                f"{var_to_impute}_post_imputation_bucketed"
                            )
                            knn_impute_subset[bucketed_post_imputation_col_name] = (
                                continuous_to_range(
                                    knn_impute_subset,
                                    post_imputation_col_name,
                                    bins,
                                    bins_to_labels,
                                )
                            )

                        # reset index for subset that underwent imputation
                        knn_impute_subset = knn_impute_subset.set_index(
                            "original_index"
                        )
                        knn_impute_subset = knn_impute_subset.rename_axis(index=None)

                        # update full data set to include post-imputation information
                        data_post_imputation.update(knn_impute_subset)

    # if there are imputation class variables that can be used in lieu of the other,
    # then separate distance matrices need to be computed per variable in multiple_possible_class_vars
    else:
        print("ELSE")
        # this list stores tuples of distances and neighbour indices per multiple_possible_class_vars variable
        list_neighbour_distances_indices = []

        for var in multiple_possible_class_vars_w_na:

            # subset data to remove blanks and potential donors with missing imputation class variables
            knn_impute_subset = data_post_imputation[
                (
                    data_post_imputation[imputation_class_vars_w_na + var]
                    .notna()
                    .all(axis=1)
                )
            ]

            # Need to min-max scale imputation class variables before determining nearest neighbours.
            imputation_class_vars_transformed = minmax_scaler.fit_transform(
                knn_impute_subset[imputation_class_vars_w_na + var]
            )

            # Use scikit-learn's BallTree function to get nearest neighbours and their indices.
            tree = BallTree(imputation_class_vars_transformed)
            list_neighbour_distances_indices.append(
                tree.query(
                    imputation_class_vars_transformed,
                    k=knn_impute_subset.shape[0],
                    return_distance=True,
                    dualtree=True,
                )
            )

        # re-subset the data to retain rows where none of imputation_class_vars are blank / missing
        # and at least one of multiple_possible_class_vars_w_na is populated
        knn_impute_subset = data_post_imputation[
            (data_post_imputation[imputation_class_vars_w_na].notna().all(axis=1))
        ].copy()

        knn_impute_subset = knn_impute_subset[
            (knn_impute_subset[multiple_possible_class_vars_w_na].notna().any(axis=1))
        ]

        # reset index of subset, but retain the old index to facilitate updating the full imput data set later
        knn_impute_subset = knn_impute_subset.reset_index(names="original_index")

        # get column numbers for indexing use later
        data_columns = knn_impute_subset.columns
        imputation_flag_col_idx = data_columns.get_loc(imputation_flag_col_name)
        imputation_type_col_idx = data_columns.get_loc(imputation_type_col_name)
        post_imputation_col_idx = data_columns.get_loc(post_imputation_col_name)
        imputed_val_col_idx = data_columns.get_loc(imputed_val_col_name)
        donor_userid_col_idx = data_columns.get_loc(donor_userid_col_name)
        donor_count_col_idx = data_columns.get_loc(donor_count_col_name)
        donor_characteristics_col_idxs = [
            data_columns.get_loc(var) for var in donor_characteristics_col_names
        ]
        donor_distance_col_idx = data_columns.get_loc(donor_distance_col_name)

        for tree_index_num, data_index_num in enumerate(knn_impute_subset.index):

            # Search for neighbours if row has missing value
            # condition formerly pd.isna(data_post_imputation[post_imputation_col_name][data_index_num])
            if pd.notna(
                knn_impute_subset[post_imputation_col_name][data_index_num]
            ) and (
                knn_impute_subset[post_imputation_col_name][data_index_num]
                in missing_values_to_impute
            ):

                neighbour_rank = 0
                continue_neighbour_search = True

                while continue_neighbour_search:

                    # Determine which multiple_possible_class_vars variable to use
                    # by finding the first variable with non-missing value
                    tree_to_use = 0
                    continue_class_vars_search = True

                    while continue_class_vars_search:
                        # This case only occurs if all variables in multiple_possible_class_vars are missing in a given row.
                        # This should not happen as a base assumption is that at least one of the imputation class variables
                        # for recipients have no missingness.
                        if tree_to_use >= len(multiple_possible_class_vars):
                            continue_class_vars_search = False
                            userid_missing_class_vars = knn_impute_subset[
                                data_index_num
                            ][userid_col_name]
                            warnings.warn(
                                f"All variables in multiple_possible_class_vars are missing for user id {userid_missing_class_vars}."
                            )

                        elif pd.isna(
                            knn_impute_subset[multiple_possible_class_vars[tree_to_use]]
                        ):
                            tree_to_use += 1

                        # first variable in multiple_possible_class_vars that is not missing
                        else:
                            continue_class_vars_search = False

                    neighbour_distances_indices = list_neighbour_distances_indices[
                        tree_to_use
                    ]

                    donor_idx = neighbour_distances_indices[1][
                        tree_index_num, neighbour_rank
                    ]
                    donor_dist = neighbour_distances_indices[0][
                        tree_index_num, neighbour_rank
                    ]
                    donor_userid = knn_impute_subset[userid_col_name][donor_idx]

                    # first check: donor use does not exceed max_donor_use.
                    if donor_use_track_dict[donor_userid] > max_donor_use:
                        neighbour_rank += 1

                    # second check: if closest donor is itself, then continue neighbour search
                    elif donor_idx == data_index_num:
                        neighbour_rank += 1

                    # third check: if donor has missing value, then continue neighbour search
                    elif (
                        knn_impute_subset[post_imputation_col_name][donor_idx]
                        in missing_values_to_impute
                    ):
                        neighbour_rank += 1

                    # else, donor is eligible; end neighbour search
                    else:
                        continue_neighbour_search = False

                        # impute value into post-imputation variable
                        knn_impute_subset.iloc[
                            data_index_num, post_imputation_col_idx
                        ] = knn_impute_subset[post_imputation_col_name][donor_idx]

                        # indicate imputation
                        knn_impute_subset.iloc[
                            data_index_num, imputation_flag_col_idx
                        ] = 1

                        # indicate type of imputation
                        knn_impute_subset.iloc[
                            data_index_num, imputation_type_col_idx
                        ] = "KNN"

                        # save imputed value
                        knn_impute_subset.iloc[data_index_num, imputed_val_col_idx] = (
                            knn_impute_subset[post_imputation_col_name][donor_idx]
                        )

                        # save donor's user id
                        knn_impute_subset.iloc[data_index_num, donor_userid_col_idx] = (
                            donor_userid
                        )

                        # increment donor's use count in donor use tracking dictionary
                        donor_use_track_dict[donor_userid] += 1

                        # save donor characteristics
                        for n, var in enumerate(imputation_class_vars):
                            knn_impute_subset.iloc[
                                data_index_num, donor_characteristics_col_idxs[n]
                            ] = knn_impute_subset[var][donor_idx]

                        # save donor distance
                        knn_impute_subset.iloc[
                            data_index_num, donor_distance_col_idx
                        ] = donor_dist

                        # populate donor use count column
                        for row_num, user_id in enumerate(
                            knn_impute_subset[donor_userid_col_name]
                        ):
                            if user_id != -1:
                                knn_impute_subset.iloc[row_num, donor_count_col_idx] = (
                                    donor_use_track_dict[user_id]
                                )

                        # bucket imputed column if bucket_imputation == True
                        if bucket_imputation:
                            bucketed_post_imputation_col_name = (
                                f"{var_to_impute}_post_imputation_bucketed"
                            )
                            knn_impute_subset[bucketed_post_imputation_col_name] = (
                                continuous_to_range(
                                    knn_impute_subset,
                                    post_imputation_col_name,
                                    bins,
                                    bins_to_labels,
                                )
                            )

                        # reset index for subset that underwent imputation
                        knn_impute_subset = knn_impute_subset.set_index(
                            "original_index"
                        )
                        knn_impute_subset = knn_impute_subset.rename_axis(index=None)

                        # update full data set to include post-imputation information
                        data_post_imputation.update(knn_impute_subset)

    return data_post_imputation


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
