import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from tqdm.auto import tqdm
import yaml


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

from donor_imputation_functions import (
    data_preprocessing_store_donor_info,
    continuous_to_range,
    range_to_random_continuous,
    monetary_buckets,
    nearest_neighbour_imputation_categorical,
    nearest_neighbour_imputation,
)
from historical_imputation_functions import (
    ratio_imputation,
    construct_house_loan_donor_pool_one_house,
    construct_house_loan_donor_pool_multiple_houses,
    construct_incomes_donor_pool,
    housing_historical_imputation_by_row,
    historical_imputation_housing,
    income_historical_imputation_by_row,
    historical_imputation_income,
)


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

# read in pipeline configuration
with open("./config.yaml", "r") as stream:
    config = yaml.safe_load(stream)

# Format paths
for path_key in config["PATHS"].keys():
    config["PATHS"][path_key] = Path(config["PATHS"][path_key])


def main(config=config):

    if config["IMPUTATION"]["DONOR_IMPUTATION"]:

        df_data_donor = pd.read_excel(config["PATHS"]["data_donor_imp"])
        df_data_donor["userid"] = range(0, len(df_data_donor))

        var_to_impute_list = list(config["IMPUTATION"]["DONOR_IMPUTATION"].keys())

        print(
            f"\n\nVariables to be imputed with donor imputation : {', '.join(var_to_impute_list)}"
        )
        with tqdm(
            total=len(var_to_impute_list), desc="Processing donor imputation"
        ) as progress_bar:
            with pd.ExcelWriter(
                config["PATHS"]["output_dir"] / config["PATHS"]["output_file_donor"]
            ) as writer:

                df_data_donor.to_excel(writer, sheet_name="Original Data", index=False)

                for var_to_impute in var_to_impute_list:

                    imputation_class_vars = config["IMPUTATION"]["DONOR_IMPUTATION"][
                        var_to_impute
                    ]
                    df_data_donor_preprocessed = data_preprocessing_store_donor_info(
                        df_data_donor, var_to_impute, imputation_class_vars
                    )

                    # apply nearest neighbour imputation
                    df_data_donor_post_knn = nearest_neighbour_imputation(
                        df_data_donor_preprocessed,
                        var_to_impute,
                        imputation_class_vars,
                        [-1],
                        True,
                        non_monetary_bins,
                        non_monetary_bins_to_enumerated_val,
                    )

                    df_data_donor_post_knn.to_excel(
                        writer, sheet_name=var_to_impute, index=False
                    )
                    progress_bar.update(1)


if __name__ == "__main__":
    main(config=config)
