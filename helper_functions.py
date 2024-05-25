import pandas as pd

# import constants
from constants import FEATURE_COLUMNS, TARGET_COLUMNS


def preprocess_and_feature_engineer(file_path, output_file_path, include_percentage=100):
    
    data = pd.read_csv(file_path, on_bad_lines='skip')
    
    predata = data[FEATURE_COLUMNS + TARGET_COLUMNS]
        
    reduced_predata = predata.sample(frac=include_percentage/100.0)
    reduced_predata.to_csv(output_file_path, index=False)
    print(f"\nPreprocessed data ({include_percentage}% of original) has been saved to {output_file_path}")




