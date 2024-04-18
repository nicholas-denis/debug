# Impudon Python Redesign

This repository hosts the code for redesigning and refactoring an imputation system called Impudon. The original implementation of Impudon was in SAS, but this repository focuses on implementing the donor imputation and historical imputation features in Python.

## Overview

This is an imputation system used to impute missing values in a dataset coming from The Central Provident Fund Board administers the Retirement and Health Study (RHS), a longitudinal study since 2014 – data is collected in biennial increments called *Waves*. 

This redesign project aims to migrate Impudon from its original SAS implementation to Python, providing a more flexible and accessible solution for imputation tasks. The repository includes code, documentation, and examples to demonstrate the functionality and usage of the redesigned Impudon system.

## Features

The Impudon Python Redesign repository focuses on implementing the following features:

1. **Donor Imputation**: This feature allows for imputation using donor values from similar records in the dataset. It helps fill in missing values by leveraging the information from other non-missing records.

2. **Historical Imputation**: Historical imputation utilizes temporal information to fill in missing values based on the historical patterns within the dataset. This feature is particularly useful when dealing with longitudinal data.

## Repository Structure

The repository is organized as follows:

```
├── src/
│   ├── config.yaml
│   ├── donor_imputation_functions.py
│   ├── historical_imputation_functions.py
│   └── main.py
│  
├── example/
│   └── imputation_functions_for_client_testing.ipynb
│  
├── requirements.txt
├── LICENSE
├── SECURITY.md
└── README.md
```

- The `src/` directory contains the Python scripts that implement the donor imputation and historical imputation features.

- The `examples/` directory provides Jupyter notebooks demonstrating the usage and application of the donor imputation and historical imputation features.


## Getting Started

To get started with the Impudon Python Redesign repository, follow these steps:


1. Navigate to the repository directory:

```bash
cd <path/to/your/folder/>
```

2. Set up the necessary dependencies and environment for running the code. Refer to the documentation provided in each script or notebook for any specific requirements. See below for more information.

3. Explore the `src/` directory to find the Python scripts implementing the donor imputation and historical imputation features.

4. Explore the different .yaml files. Each one is setup for a particular experiment. The YAML file determines the entire imputation being applied. More information is provided below.

5. OPTIONAL: Copy required data sources into the /data/ folder.

6. Refer to the `experiments/` directory for Jupyter notebooks that hosts imputation experiments.

7. Refer to the `examples/` directory for Jupyter notebooks that demonstrate the usage and application of the imputation methods.

## Virtual Environment
We recommend setting up a virtual environment to manage and contain this project's dependencies, as opposed to using the global environment.
 
#### To set up a python virtual environment:
To create a virtual environment named **env_name**.
 
```bash
$ python -m venv <path-to-envs>/env_name
```

 
To activate virtual environment with **Command Prompt (Windows System)**.
 
```bash
$ <path-to-envs>/env_name/Scripts/activate
```
 
To activate virtual environment with **cmd.exe**.
 
```bash
$ <path-to-envs>/env_name/Scripts/activate.bat 
```
 
 
To activate virtual environment with **PowerShell**.
 
```bash
$ <path-to-envs>/env_name/Scripts/activate.ps1
```

To deactivate your virtual environment:
 
```bash
$ deactivate
```

## Requirements
To install the dependencies once your virtual environment is activated:
 
```bash
$ pip install -r requirements.txt
```


## YAML Config files

The YAML config files are used by the /src/main.py script and must be set properly. The first section involves paths to the source data and where the resulting data should be saved. For repeated experiments you may want to change the save paths so that previous experiment outputs are not overrun. Feel free to create as many YAML files as you wish, one for each pertinent experiment.

There are some examples of usage. Note that if you wish to only perform historical imputation, then the donor imputation section should be removed. The converse is also true. If you would like to do donor and historical imputation, then have both respective sections present.

Donor imputation, as can be seen in the example yaml files, requires that you map the donor columns to a list of columns used for nearest neighbours. Example:
```config
...

DONOR_IMPUTATION:
    'column_1': ['column_x', 'column_y', 'column_abc']
    'column_341': ['AABDDB_2', 'column_yyyz']
...
```
 
 In the above example, all the rows that are missing data in the column with title "column_1" will be imputed using the nearest neighbour imputation approach, which uses the columns with names 'column_x', 'column_y' and 'column_abc' for computing distances. After this column is imputed, the next to be imputed will be rows that are missing data in the column titled 'column_341', which will use the columns named 'AABDDB_2', 'column_yyyz' for computing distances in nearest neighbours.

 **NOTE**: The set of donors is found by finding all rows that have NO missing values for each of the columns (ex., ['column_x', 'column_y', 'column_abc']). If there exist no rows that satisfy this, then the program will throw an error, as there are no feasible donors, since each row is missing at least one value from these columns.

 **NOTE**: As well, if a given row is missing a value in 'column_1' but ALSO in any of the columns used for nearest neighbour search (ex. any of these columns: ['column_x', 'column_y', 'column_abc']), then that row will not be subjected to imputation. This is because computing a distance based on those columns is impossible, as it is missing a value for one of those dimensions.
 

## Running the script /src/main.py

From the command line simply cd into </path/to/src/> and execute:
```bash
python main.py <path_to_yaml_file>
```
For example:

```bash
python main.py iris_config.yaml
```
or
```bash
python main.py my_custom_config.yaml
```
or 
```bash
python main.py bank_donor_config.yaml
```
or
```bash
python main.py historical_housing_config.yaml
```
or if your yaml file is in a different folder than /src/:
```bash
python main.py /the/path/to/your/super/special/config.yaml
```


Please keep in mind that the yaml file determines the entire script process. 
Also note that the '#' is a commenting character, meaning anything that follows a '#' is ignored and not read.
Finally, please have a look at the paths among the different yaml files. You will note some using forward and some using backward slashes, example:
```config
path_var = 'C:\some\windows\path\'
linux_var = '/the/path/is/this/way/'
```
This has to do with the structure used by Linux vs Windows machines. Please adapt your paths according to your operating systems.



## License
 
Unless otherwise specified, the source code of this project is covered under Crown Copyright, Government of Canada, and is distributed under the [MIT License](https://github.com/StatCan/impudon-redesign/-/blob/main/LICENSE).
 
The Canada wordmark and related graphics associated with this distribution are protected under trademark law and copyright law. No permission is granted to use them outside the parameters of the Government of Canada's corporate identity program. For more information, see [https://www.canada.ca/en/treasury-board-secretariat/topics/government-communications/federal-identity-requirements.html](Federal identity requirements).
 
 
 
## Authors
 
* **Loïc Muhirwa** - *Initial work* - [lmuhi](https://github.com/lmuhi) <br/>
* **Angela Wang-Lin** - *Initial work* - [angela.wang-lin](https://gitlab.k8s.cloud.statcan.ca/angela.wang-lin) <br/>
