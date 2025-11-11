import pandas as pd
import platform
from pathlib import Path


train_data = None
test_data = None
ground_truth = None
trained_models = {}
scalers = {}


# Code logic for loading and processing the ground truth data:
def get_ground_truth():
    """
    Load the ground truth RUL values for test data.
    Returns a pandas DataFrame with unit numbers and true RUL values.
    """
    global ground_truth

    # Use relative path from the demo directory
    # data_path = Path("../../../../data/CMAPSSData/RUL_FD001.txt")

    # test with absolute path

    # return absolute path based on corresponding architecture
    decide_path: str = (
        "/home/dasa60196/Desktop/research_ibm/Intent-Based-Industrial-Automation/data/CMAPSSData/RUL_FD001.txt"
        if str(platform.machine() == "aarch64")
        else "/Users/ayandas/Desktop/research_ibm/Intent-Based-Industrial-Automation/data/CMAPSSData/RUL_FD001.txt"
    )

    print(f"Decided path : {decide_path}")
    data_path = Path(decide_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Ground truth data not found at {data_path}")

    ground_truth = pd.read_csv(data_path, header=None, names=["RUL"])
    ground_truth["unit"] = range(1, len(ground_truth) + 1)

    print(f"✅ Ground truth loaded: {len(ground_truth)} test engines")
    print(f"   RUL range: {ground_truth['RUL'].min()} to {ground_truth['RUL'].max()}")

    return ground_truth


get_ground_truth()


# Code logic for importing and processing the test data
def get_reference_test_data():
    """
    Load and preprocess the test data from CMAPSS dataset.
    Returns a pandas DataFrame with proper column names.
    """
    global test_data

    # Use relative path from the demo directory
    # data_path = Path("../../../../data/CMAPSSData/test_FD001.txt")

    # test with absolute path
    decide_path: str = (
        "/home/dasa60196/Desktop/research_ibm/Intent-Based-Industrial-Automation/data/CMAPSSData/test_FD001.txt"
        if str(platform.machine() == "aarch64")
        else "/Users/ayandas/Desktop/research_ibm/Intent-Based-Industrial-Automation/data/CMAPSSData/test_FD001.txt"
    )
    data_path = Path(decide_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Test data not found at {data_path}")

    # Load data with proper column names
    # CMAPSS format: unit, time, op_setting_1, op_setting_2, op_setting_3, sensor_1, ..., sensor_19
    # Note: CMAPSS FD001 only has 19 sensors, not 21!
    column_names = ["unit", "time", "op_setting_1", "op_setting_2", "op_setting_3"] + [
        f"sensor_{i}" for i in range(1, 22)
    ]  # 21 sensors (sensor_1 to sensor_21)

    # Read with proper handling of trailing spaces
    test_data = pd.read_csv(
        data_path, sep="\s+", header=None, names=column_names, engine="python"
    )
    test_data = test_data.dropna(axis=1)

    # Debug: Check the first few rows to verify correct parsing
    print(f"   Raw test data parsing debug:")
    print(
        f"   - First row: unit={test_data.iloc[0]['unit']}, time={test_data.iloc[0]['time']}"
    )
    print(f"   - Unit range: {test_data['unit'].min()} to {test_data['unit'].max()}")
    print(f"   - Time range: {test_data['time'].min()} to {test_data['time'].max()}")
    print(f"   - Data shape: {test_data.shape}")
    print(f"   - Columns: {list(test_data.columns)}")

    # Verify that unit and time are integers
    if not test_data["unit"].dtype in ["int64", "int32"]:
        print(
            f"   ⚠️  WARNING: Unit column is not integer type: {test_data['unit'].dtype}"
        )
    if not test_data["time"].dtype in ["int64", "int32"]:
        print(
            f"   ⚠️  WARNING: Time column is not integer type: {test_data['time'].dtype}"
        )

    # Check for any NaN values in the data
    nan_count = test_data.isnull().sum().sum()
    if nan_count > 0:
        print(f"   ⚠️  WARNING: Found {nan_count} NaN values in the data")
        print(f"   NaN by column: {test_data.isnull().sum()}")
    else:
        print(f"   ✅ No NaN values found in raw test data")

    print(
        f"✅ Test data loaded: {test_data.shape[0]} samples, {test_data.shape[1]} features"
    )
    print(f"   Engines: {test_data['unit'].nunique()}")

    return test_data


get_reference_test_data()


def get_reference_train_data():
    """
    Importing and processing training data.
    Load and preprocess the training data from CMAPSS dataset.
    Returns a pandas DataFrame with proper column names and RUL calculation.
    """
    global train_data

    # Use relative path from the demo directory
    # data_path = Path("../../../../data/CMAPSSData/train_FD001.txt")

    # test with absolute path
    decide_path: str = (
        "/home/dasa60196/Desktop/research_ibm/Intent-Based-Industrial-Automation/data/CMAPSSData/train_FD001.txt"
        if str(platform.machine() == "aarch64")
        else "/Users/ayandas/Desktop/research_ibm/Intent-Based-Industrial-Automation/data/CMAPSSData/train_FD001.txt"
    )
    data_path = Path(decide_path)

    # NOTE : wrap around Path around the string to ensure that the file exists
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    # Load data with proper column names
    # CMAPSS format: unit, time, op_setting_1, op_setting_2, op_setting_3, sensor_1, ..., sensor_19
    # Note: CMAPSS FD001 only has 19 sensors, not 21!
    column_names = ["unit", "time", "op_setting_1", "op_setting_2", "op_setting_3"] + [
        f"sensor_{i}" for i in range(1, 22)
    ]  # 21 sensors (sensor_1 to sensor_21)

    # Read with proper handling of trailing spaces
    train_data = pd.read_csv(
        data_path, sep="\s+", header=None, names=column_names, engine="python"
    )

    # CRITICAL FIX: Convert unit and time to integers
    train_data["unit"] = train_data["unit"].astype(int)
    train_data["time"] = train_data["time"].astype(int)

    # Debug: Check the first few rows to verify correct parsing
    # print(f"   Raw data parsing debug:")
    # print(f"   - First row: unit={train_data.iloc[0]['unit']}, time={train_data.iloc[0]['time']}")
    # print(f"   - Unit range: {train_data['unit'].min()} to {train_data['unit'].max()}")
    # print(f"   - Time range: {train_data['time'].min()} to {train_data['time'].max()}")
    # print(f"   - Data shape: {train_data.shape}")
    # print(f"   - Columns: {list(train_data.columns)}")

    # Verify that unit and time are integers
    if not train_data["unit"].dtype in ["int64", "int32"]:
        print(
            f"   ⚠️  WARNING: Unit column is not integer type: {train_data['unit'].dtype}"
        )
    if not train_data["time"].dtype in ["int64", "int32"]:
        print(
            f"   ⚠️  WARNING: Time column is not integer type: {train_data['time'].dtype}"
        )

    # Check for any NaN values in the data
    nan_count = train_data.isnull().sum().sum()

    # HINT : provides a tabular view, with boolean values indicating whether or not the corresponding individual value is a null or not
    # NOTE : uncomment to see the breakdown (the following 3 print statements)
    # print("---nan-count synatx test--")
    # print(train_data.isnull())

    # This basically prints out a series, aggregating the columns based on number of columns that have null values, which is none
    # print(train_data.isnull().sum())

    # this further aggregates into a singular value
    # print("dual sum null value check?")
    # print(train_data.isnull().sum().sum())

    if nan_count > 0:
        print(f"   ⚠️  WARNING: Found {nan_count} NaN values in the data")
        print(f"   NaN by column: {train_data.isnull().sum()}")
    else:
        print(f"   ✅ No NaN values found in raw data")

    # Calculate RUL for training data
    # RUL = max_cycle - current_cycle (for CMAPSS dataset)
    # Calculate RUL for each engine
    train_data = train_data.copy()
    train_data["RUL"] = 0  # initial value?

    for unit_id in train_data["unit"].unique():
        # print(f"current unique id : {unit_id}")

        # creates a boolean series --> suppose current unit_id is 1 --> then it will appear as [True, False, False, ...]
        unit_mask = train_data["unit"] == unit_id

        # select the rows where unit masking is true, thus a new dataframe gets created
        # basically serves as a filter
        unit_data = train_data[unit_mask]

        max_cycle = unit_data["time"].max()  # fairly straightforward

        # Calculate RUL: cycles remaining until failure
        rul = max_cycle - unit_data["time"]

        # Update the RUL in the main dataframe
        train_data.loc[unit_mask, "RUL"] = rul.values

    # RUL already calculated above

    # Debug RUL calculation

    # TODO : uncomment them as needed, intended to display specific information
    # print(f"   RUL calculation debug:")
    # print(f"   - Time range per engine: {train_data.groupby('unit')['time'].agg(['min', 'max']).head()}")
    # print(f"   - RUL range: {train_data['RUL'].min():.0f} to {train_data['RUL'].max():.0f}")
    # print(f"   - RUL mean: {train_data['RUL'].mean():.2f}")

    # # Keep RUL in original scale (cycles) for meaningful interpretation
    # # No normalization needed - RUL in cycles is more interpretable

    # print(f"✅ Training data loaded: {train_data.shape[0]} samples, {train_data.shape[1]} features")
    # print(f"   Engines: {train_data['unit'].nunique()}")
    # print(f"   RUL range: {train_data['RUL'].min()} to {train_data['RUL'].max()}")

    return train_data


get_reference_train_data()
