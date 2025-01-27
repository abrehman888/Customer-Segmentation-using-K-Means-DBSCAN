import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    df.drop(["CustomerID", "Gender", "Age"], axis=1, inplace=True)
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    
    return df, scaled_features
