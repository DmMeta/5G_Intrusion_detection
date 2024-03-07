
from time import perf_counter as time


def drop_features_(df, thresholds, features_dropped ):
    """
    Drop columns from a DataFrame based on specified thresholds for NaN and zero values.
    A column is also dropped if it contains only a single unique value.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        thresholds (list): A list of two values representing the multipliers for NaN and zero thresholds.
        features_dropped (dict): A dictionary to store the names of the dropped columns.
        
    Returns:
        Tuple[pandas.DataFrame,float]: The DataFrame with dropped columns 
        and the time taken to perform the operation.
    """
    
    nan_multiplier = thresholds[0]
    zero_multiplier = thresholds[1]
    
    t0 = time()

    threshold_nans = nan_multiplier * df.shape[0]
    columns_Nan_to_drop = df.columns[df.isna().sum() > threshold_nans]
    features_dropped["Nan columns"].extend(columns_Nan_to_drop)
    df_cleaned_nan  = df.drop(columns=columns_Nan_to_drop)
    # eq df_cleaned_nan = df.dropna(thresh = ceil(1 - threshold_nans = 0.95 * df.shape[0]), axis = 1)
    print(f"After dropping NaN columns: {df_cleaned_nan.shape}")

    threshold_zeros = zero_multiplier * df.shape[0]
    zero_counts = df_cleaned_nan.apply(lambda col: (col == 0).sum())
    columns_zeros_to_drop = zero_counts[zero_counts > threshold_zeros].index
    features_dropped["Zero columns"].extend(columns_zeros_to_drop)
    df_cleaned_zeros = df_cleaned_nan.drop(columns_zeros_to_drop, axis=1)
    print(f"After dropping NaN & Zero columns: {df_cleaned_zeros.shape}")

    constant_columns = df_cleaned_zeros.columns[df_cleaned_zeros.nunique() == 1]
    features_dropped["Constant columns"].extend(constant_columns)
    df_cleaned = df_cleaned_zeros.drop(columns=constant_columns)
    print(f"After dropping NaN & Zero & Constant columns: {df_cleaned.shape}")

    t1 = time()
    
    return df_cleaned, t1 - t0


def fill_nan_values(df, method='mean'):
    """
    Fill NaN values in a DataFrame's numerical features using the specified method.
    It also fills NaN values in categorical features with the mode.

    Parameters:
    df (DataFrame): The DataFrame to fill NaN values in.
    method (str): The method to use for filling NaN values. Default is 'mean'.

    Returns:
    df (DataFrame): The DataFrame with NaN values filled.
    execution_time (float): The time taken to fill NaN values in seconds.
    """
    t0 = time()

    numeric_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(include='object').columns
    
    fill_method = {
        'mean': df[numeric_cols].mean,
        'mode': df[numeric_cols].mode,
        'median': df[numeric_cols].median
    }
    
    numeric_cols_fill_values = fill_method[method]()
    df[numeric_cols] = df[numeric_cols].fillna(numeric_cols_fill_values)

    #iloc[0] is used to get the first element of the series in case there is more than one category that produces 
    # the same mode.
    categorical_cols_mode = df[categorical_cols].mode().iloc[0]
    df[categorical_cols] = df[categorical_cols].fillna(categorical_cols_mode)

    t1 = time()
    
    return df, t1 - t0
