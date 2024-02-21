import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize 
import numpy as np

df = pd.read_csv("attacks_labeled.csv")

initial_len = df.shape[0]
df = df.dropna()
print(f'Tamanho inicial: {initial_len}, tamanho final {df.shape[0]} | Descartados {initial_len - df.shape[0]} registros com valores NA')

#df = df.drop('weight', axis=1)

#print(df)

df = df.reset_index(drop=True)

df_not_numeric = df.select_dtypes(exclude=[np.number])
not_numeric_columns = df_not_numeric.columns
encoder = LabelEncoder()
for column in not_numeric_columns:
    df[column] = encoder.fit_transform(df[column])
    
std_scaler = StandardScaler()

df_scaler = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)
X_normalized = normalize(df_scaler)
X_normalized = pd.DataFrame(X_normalized) 

y = df_scaler['class']
X = df_scaler.drop('class', axis=1)  # Assuming 'type' is the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
train_normalized = normalize(X_train)
train_normalized = pd.DataFrame(train_normalized) 
test_normalized = normalize(X_test)
test_normalized = pd.DataFrame(test_normalized) 
#X_train, X_test= train_test_split(X_normalized, test_size=0.3, random_state=42)





