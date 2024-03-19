import os
import pandas as pd

DATA_PATH = os.path.join("..","..","..",'data',"BTS1_BTS2_fields_preserved.zip")

def load_dataset():
    return pd.read_csv(DATA_PATH, compression="zip", low_memory=False)