"""
use this file for loading data - isolated script file used to load the huggingface data
"""

from datasets import load_dataset
import os

# name of the PDMBench data
data_directory_name = "PDMBench_Data_Directory"

# identify path where the data directory will be made
target_path = os.path.join(os.getcwd(), data_directory_name)

# create the directory in case it doesn't exists
os.makedirs(target_path, exist_ok=True)

# List of data from huggingface regarding PDMBench
AVAILABLE_DATASETS = [
    "submission096/XJTU",
    "submission096/MAFAULDA",
    "submission096/Padeborn",
    "submission096/IMS",
    "submission096/UoC",
    "submission096/RotorBrokenBar",
    "submission096/MFPT",
    "submission096/HUST",
    "submission096/FEMTO",
    "submission096/Mendeley",
    "submission096/ElectricMotorVibrations",
    "submission096/CWRU",
    "submission096/Azure",
    "submission096/PlanetaryPdM",
]

# Load and store the data
for curr_data in AVAILABLE_DATASETS:
    try:
        curr_dataset = load_dataset(curr_data)

        for split_name, split_dataset in curr_dataset.items():
            file_path = os.path.join(target_path, f"{curr_data}_{split_name}.csv")
            split_dataset.to_csv(file_path, index=False)
    except Exception as e:
        print(f"Error occured : {e}")
