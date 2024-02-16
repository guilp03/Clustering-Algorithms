import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = pd.read_csv("diabetic_data.csv")

initial_len = df.shape[1]
df = df.dropna(axis=1)
print(f'Tamanho inicial: {initial_len}, tamanho final {df.shape[1]} | Descartados {initial_len - df.shape[1]} registros com valores NA')

df = df.drop('weight', axis=1)

print(df)

df = df.reset_index(drop=True)

df_not_numeric = df.select_dtypes(exclude=[np.number])
not_numeric_columns = df_not_numeric.columns
encoder = LabelEncoder()
for column in not_numeric_columns:
    df[column] = encoder.fit_transform(df[column])
    
std_scaler = StandardScaler()

df_scaler = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)
df_scaler = df_scaler.sample(frac=0.1)
#y = df_scaler["type"]
#df_scaler = df_scaler.drop(labels= 'type', axis= 1)


print(df_scaler)

