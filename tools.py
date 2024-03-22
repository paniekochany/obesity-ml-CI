import os
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from typing import Tuple, Union
from zlib import crc32

COLUMN_NAMES = [
    'gender',
    'age',
    'height',
    'weight',
    'family_history_overweight',
    'frequent_high_caloric_food_consumption',
    'vegetable_consumption_frequency',
    'number_of_main_meals',
    'consumption_of_food_between_meals',
    'SMOKE',
    'daily_water_consumption',
    'calories_consumption_monitoring',
    'physical_activity_frequency',
    'time_using_technology_devices',
    'alcohol_consumption',
    'transportation_used',
    'obesity_level',
]


def unzip_and_open_dataset(zip_file_path):
    '''Extracts zip file and reads into pandas dataframe.'''

    data_directory = Path('datasets') / os.path.splitext(zip_file_path)[0]
    if not data_directory.is_dir():
        data_directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_file_path) as zip_file:
        zip_file.extractall(data_directory)

    csv_file = [file for file in os.listdir(data_directory) if file.endswith('.csv')][0]

    return pd.read_csv(data_directory / csv_file)


def _id_in_the_test_set(identifier: Union[int, float], test_ratio: float) -> bool:
    """Assigns identifier to either test or train set based on hash value."""

    return crc32(np.int64(identifier)) < test_ratio * 2**32


def split_data_with_id_hash(data: pd.DataFrame, test_ratio: float, id_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data into train and test set based on hashe values of identifiers."""

    in_the_test_set = data[id_column].apply(lambda row_id: _id_in_the_test_set(row_id, test_ratio))
    return data.loc[~in_the_test_set], data.loc[in_the_test_set]
