import os
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(input_path: str, output_path: str):
    # Load dataset
    df = pd.read_csv(input_path)

    # Drop kolom id dan Date
    df = df.drop(columns=['id', 'Date'])

    # Tangani outlier harga dan luas bangunan menggunakan metode IQR
    for col in ['Price','living area']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    print(f"Dataset setelah drop outlier: {df.shape[0]} baris")

    # Split fitur & target
    X = df.drop(columns=['Price'])
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Tandai split dalam satu kolom
    df_clean = df.copy()
    df_clean['split'] = None
    df_clean.loc[X_train.index, 'split'] = 'train'
    df_clean.loc[X_test.index,  'split'] = 'test'

    # Simpan preprocessed + split
    df_clean.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

    return df_clean

if __name__ == '__main__':
    base = os.path.dirname(__file__)
    raw  = os.path.join(base, '..', 'house-price-india_raw.csv')
    outp = os.path.join(base, '..', 'house-price-india_preprocessed.csv')
    preprocess(raw, outp)