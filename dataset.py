import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("arquivo_junto.csv", sep=';')

std_scaler = StandardScaler()

df_scaler = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)

print(df_scaler)

