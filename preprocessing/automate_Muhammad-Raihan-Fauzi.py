import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_data(raw_csv_path: str) -> pd.DataFrame:
    """
    Memuat data mentah dari file CSV.
    """
    return pd.read_csv(raw_csv_path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
      1. Drop kolom id dan Date
      2. Hapus outlier pada Price dan living area menggunakan IQR
    """
    # 1. Drop kolom id dan Date
    df = df.drop(columns=['id', 'Date'])

    # 2. Hapus outlier menggunakan IQR
    for col in ['Price', 'living area']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


def split_and_save(df: pd.DataFrame, output_dir: str):
    """
    Pisahkan data menjadi train/test lalu simpan sebagai CSV.
    """
    # Pisahkan fitur dan target
    X = df.drop(columns='Price')
    y = df['Price']

    # Bagi data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pastikan folder output ada
    os.makedirs(output_dir, exist_ok=True)

    # Simpan full preprocessed
    full_path = os.path.join(output_dir, 'house-price-india_preprocessed.csv')
    df.to_csv(full_path, index=False)

    # Gabungkan dan simpan train/test
    train = X_train.copy()
    train['Price'] = y_train.values
    test = X_test.copy()
    test['Price'] = y_test.values

    train.to_csv(os.path.join(output_dir, 'house-price-india_train.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'house-price-india_test.csv'), index=False)


def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    raw_path = os.path.join(base_dir, '..', 'house-price-india_raw.csv')
    output_dir = os.path.join(base_dir, '..', 'data')

    df = load_data(raw_path)
    df_clean = preprocess(df)
    print(f"Dataset setelah drop outlier: {df_clean.shape[0]} baris")
    split_and_save(df_clean, output_dir)


if __name__ == '__main__':
    main()