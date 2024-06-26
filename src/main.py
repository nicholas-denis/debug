import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from tqdm.auto import tqdm
import yaml
import argparse


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


def main(config):

    donor_imputation = config.get("IMPUTATION", {}).get("DONOR_IMPUTATION", None)
    if donor_imputation:
    #if config["IMPUTATION"]["DONOR_IMPUTATION"]:
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



    # Using get() to safely access nested dictionary keys
    housing_vars = config.get("IMPUTATION", {}).get("HISTORICAL_IMPUTATION", {}).get("HOUSING_VARS", None)
    if housing_vars:
    #if config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["HOUSING_VARS"]:

        df_data_hist = pd.read_excel(config["PATHS"]["data_hist_imp"])

        var_to_impute_list = list(
            config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["HOUSING_VARS"].keys()
        )

        print(
            f"\n\nHousing variables to be imputed with historical imputation : {', '.join(var_to_impute_list)}"
        )
        with tqdm(
            total=len(var_to_impute_list), desc="Processing housing imputation"
        ) as progress_bar:

            with pd.ExcelWriter(
                config["PATHS"]["output_dir"]
                / config["PATHS"]["output_file_hist_housing"]
            ) as writer:

                df_data_hist.to_excel(writer, sheet_name="Original Data", index=False)

                for var_current_wave in var_to_impute_list:
                    var_previous_wave = config["IMPUTATION"]["HISTORICAL_IMPUTATION"][
                        "HOUSING_VARS"
                    ][var_current_wave]["var_housing_loan_previous_wave"]
                    var_number_properties_current_wave = config["IMPUTATION"][
                        "HISTORICAL_IMPUTATION"
                    ]["HOUSING_VARS"][var_current_wave][
                        "var_number_properties_current_wave"
                    ]
                    var_number_properties_previous_wave = config["IMPUTATION"][
                        "HISTORICAL_IMPUTATION"
                    ]["HOUSING_VARS"][var_current_wave][
                        "var_number_properties_previous_wave"
                    ]
                    var_hdb_indicator_current_wave = config["IMPUTATION"][
                        "HISTORICAL_IMPUTATION"
                    ]["HOUSING_VARS"][var_current_wave][
                        "var_hdb_indicator_current_wave"
                    ]
                    var_hdb_indicator_previous_wave = config["IMPUTATION"][
                        "HISTORICAL_IMPUTATION"
                    ]["HOUSING_VARS"][var_current_wave][
                        "var_hdb_indicator_previous_wave"
                    ]
                    imputation_class_vars = config["IMPUTATION"][
                        "HISTORICAL_IMPUTATION"
                    ]["HOUSING_VARS"][var_current_wave]["imputation_class_vars"]
                    values_to_impute = config["IMPUTATION"]["HISTORICAL_IMPUTATION"][
                        "HOUSING_VARS"
                    ][var_current_wave]["values_to_impute"]

                    house_loan_donor_pool_one_house = (
                        construct_house_loan_donor_pool_one_house(
                            df_data_hist.copy(),
                            var_current_wave,
                            var_previous_wave,
                            var_number_properties_current_wave,
                            var_number_properties_previous_wave,
                            var_hdb_indicator_current_wave,
                            var_hdb_indicator_previous_wave,
                            values_to_impute,
                        )
                    )

                    house_loan_donor_pool_multiple = (
                        construct_house_loan_donor_pool_multiple_houses(
                            df_data_hist.copy(),
                            var_current_wave,
                            var_previous_wave,
                            var_number_properties_current_wave,
                            var_number_properties_previous_wave,
                            var_hdb_indicator_current_wave,
                            var_hdb_indicator_previous_wave,
                            values_to_impute,
                        )
                    )

                    df_data_hist_imputation_ready = data_preprocessing_store_donor_info(
                        df_data_hist.copy(), var_current_wave, imputation_class_vars
                    )

                    df_data_hist_post_imputation = historical_imputation_housing(
                        df_data_hist_imputation_ready,
                        var_current_wave,
                        var_previous_wave,
                        var_number_properties_current_wave,
                        var_hdb_indicator_current_wave,
                        var_hdb_indicator_previous_wave,
                        house_loan_donor_pool_one_house,
                        house_loan_donor_pool_multiple,
                        values_to_impute,
                    )

                    df_data_hist_post_imputation[
                        df_data_hist_post_imputation[
                            f"{var_current_wave}_imputation_flag"
                        ]
                        == 1
                    ][
                        [
                            "userid",
                            "Total_Purchase_Price",
                            "Total_Purchase_Price_prev_wave",
                            var_hdb_indicator_current_wave,
                            var_hdb_indicator_previous_wave,
                            var_number_properties_current_wave,
                            var_number_properties_previous_wave,
                            var_current_wave,
                            var_previous_wave,
                            f"{var_current_wave}_post_imputation",
                            f"{var_current_wave}_imputation_flag",
                            f"{var_current_wave}_imputation_type",
                            f"{var_current_wave}_imputed_value",
                            "avg_previous_wave",
                            "avg_current_wave",
                            "ratio"
                        ]
                    ].to_excel(
                        writer, sheet_name=var_current_wave, index=False
                    )
                    progress_bar.update(1)

    income_vars = config.get("IMPUTATION", {}).get("HISTORICAL_IMPUTATION", {}).get("INCOME_VARS", None)
    if income_vars:
    #if config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"]:

        df_data_hist = pd.read_excel(config["PATHS"]["data_hist_imp"])

        var_to_impute_list = list(
            config["IMPUTATION"]["HISTORICAL_IMPUTATION"]["INCOME_VARS"].keys()
        )

        print(
            f"\n\nIncome variables to be imputed with historical imputation : {', '.join(var_to_impute_list)}"
        )
        with tqdm(
            total=len(var_to_impute_list), desc="Processing income imputation"
        ) as progress_bar:

            with pd.ExcelWriter(
                config["PATHS"]["output_dir"]
                / config["PATHS"]["output_file_hist_income"]
            ) as writer:

                df_data_hist.to_excel(writer, sheet_name="Original Data", index=False)

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

                    imputation_group = construct_incomes_donor_pool(
                        df_data_hist.copy(),
                        var_current_wave,
                        var_previous_wave,
                        var_number_jobs_current_wave,
                        var_number_jobs_previous_wave,
                        var_ssic_current_wave,
                        values_to_impute,
                    )

                    df_data_hist_imputation_ready = data_preprocessing_store_donor_info(
                        df_data_hist.copy(), var_current_wave, imputation_class_vars
                    )

                    df_data_hist_post_imputation = historical_imputation_income(
                        df_data_hist_imputation_ready,
                        var_current_wave,
                        var_previous_wave,
                        var_number_jobs_current_wave,
                        var_number_jobs_previous_wave,
                        var_ssic_current_wave,
                        imputation_group,
                        values_to_impute,
                    )

                    df_data_hist_post_imputation[
                        df_data_hist_post_imputation[
                            f"{var_current_wave}_imputation_flag"
                        ]
                        == 1
                    ][
                        [
                            "userid",
                            var_number_jobs_current_wave,
                            var_number_jobs_previous_wave,
                            var_ssic_current_wave,
                            f"{var_ssic_current_wave}_prev_wave",
                            var_current_wave,
                            var_previous_wave,
                            f"{var_current_wave}_post_imputation",
                            f"{var_current_wave}_imputed_value",
                            f"{var_current_wave}_imputation_flag",
                            f"{var_current_wave}_imputation_type",
                            'avg_previous_wave', 
                            'avg_current_wave', 
                            'ratio'
                        ]
                    ].to_excel(
                        writer, sheet_name=var_current_wave, index=False
                    )
                    progress_bar.update(1)

def parse_arguments():

    # YAML_FILENAME = "./config.yaml"
    #YAML_FILENAME = "/workspaces/debug/Donor imputation_bank savings_forsc/bank_donor_config.yaml"
    #YAML_FILENAME = "/workspaces/debug/Historical_Imputation_WorkIncome/config.yaml"

    parser = argparse.ArgumentParser(description="Process YAML configuration file path")
    parser.add_argument("yaml_file", nargs="?", type=str, default="config.yaml", help="Path to the YAML configuration file")
    args = parser.parse_args()
    return args

if __name__ == "__main__":


    args = parse_arguments()
    yaml_file_path = args.yaml_file

    # read in pipeline configuration
    with open(yaml_file_path, "r") as stream:
        config = yaml.safe_load(stream)
    
    for path_key in config["PATHS"].keys():
        config["PATHS"][path_key] = Path(config["PATHS"][path_key])

    main(config=config)
