import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Tuple, Dict, Any, Optional, List


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and validation sets with stratification.
    
    Returns:
        X_train (pd.DataFrame): Training features
        X_val (pd.DataFrame): Validation features
        y_train (pd.Series): Training target
        y_val (pd.Series): Validation target
    """
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target_col]
    )
    return (
        train_df.drop(columns=[target_col]).copy(),
        val_df.drop(columns=[target_col]).copy(),
        train_df[target_col].copy(),
        val_df[target_col].copy()
    )


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drop specified columns from the dataframe.
    """
    return df.drop(columns=columns, errors="ignore")


def scale_numeric_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numeric_cols: list
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Scale numeric columns using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[numeric_cols])
    scaled_train = scaler.transform(train_df[numeric_cols]).astype(np.float64)
    scaled_val = scaler.transform(val_df[numeric_cols]).astype(np.float64)

    train_df[numeric_cols] = scaled_train
    val_df[numeric_cols] = scaled_val

    return train_df, val_df, scaler


def encode_categorical_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    categorical_cols: list
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, list]:
    """
    One-hot encode categorical columns.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_encoded = encoder.transform(train_df[categorical_cols])
    val_encoded = encoder.transform(val_df[categorical_cols])

    # Append encoded columns
    train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=train_df.index)
    val_encoded_df = pd.DataFrame(val_encoded, columns=encoded_cols, index=val_df.index)

    train_df = pd.concat([train_df.drop(columns=categorical_cols), train_encoded_df], axis=1)
    val_df = pd.concat([val_df.drop(columns=categorical_cols), val_encoded_df], axis=1)

    return train_df, val_df, encoder, encoded_cols


def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = True
) -> Dict[str, Any]:
    """
    Preprocess the raw dataframe.

    Args:
        raw_df (pd.DataFrame): Original dataset.
        scaler_numeric (bool): Whether to scale numeric features.

    Returns:
        Dict[str, Any]: Dictionary with processed train/val data and fitted scaler/encoder.
    """
    columns_to_drop = ['CustomerId', 'Surname', 'id']
    raw_df = raw_df.drop(columns=columns_to_drop, errors='ignore')

    train_df, val_df = train_test_split(
        raw_df, test_size=0.2, random_state=42, stratify=raw_df['Exited']
    )

    input_cols = [col for col in train_df.columns if col != 'Exited']
    target_col = 'Exited'
    train_inputs, train_targets = train_df[input_cols].copy(), train_df[target_col].copy()
    val_inputs, val_targets = val_df[input_cols].copy(), val_df[target_col].copy()

    numeric_cols = train_inputs.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_inputs.select_dtypes(include='object').columns.tolist()

    scaler = None
    if scaler_numeric:
        scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
        train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
        val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
    val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])

    final_input_cols = numeric_cols + encoded_cols

    X_train = train_inputs[final_input_cols]
    X_val = val_inputs[final_input_cols]

    return {
        'X_train': X_train,              
        'train_targets': train_targets, 
        'X_val': X_val,                  
        'val_targets': val_targets,      
        'input_cols': final_input_cols,  
        'numeric_cols': numeric_cols,    
        'categorical_cols': categorical_cols,  
        'scaler': scaler,                
        'encoder': encoder               
    }


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler: Optional[MinMaxScaler],
    encoder: OneHotEncoder
) -> pd.DataFrame:
    """
    Preprocess new data using fitted scaler and encoder.

    Args:
        new_df (pd.DataFrame): New dataset to preprocess.
        input_cols (List[str]): Final list of input columns from training.
        numeric_cols (List[str]): List of numeric column names.
        categorical_cols (List[str]): List of categorical column names.
        scaler (Optional[MinMaxScaler]): Fitted scaler (or None if no scaling).
        encoder (OneHotEncoder): Fitted encoder.

    Returns:
        pd.DataFrame: Preprocessed new data ready for prediction.
    """
    new_df = new_df.copy()

    # Scale numeric features if scaler was used
    if scaler is not None:
        new_df[numeric_cols] = scaler.transform(new_df[numeric_cols])

    # One-hot encode categorical features
    encoded_array = encoder.transform(new_df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=new_df.index)
    new_df[encoded_cols] = encoded_df

    preprocessed_df = new_df[input_cols]

    return preprocessed_df

