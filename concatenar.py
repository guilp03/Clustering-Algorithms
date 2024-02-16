import pandas as pd

# Carregar os arquivos CSV
df1 = pd.read_csv('wine+quality/winequality-red.csv', sep=';')
df2 = pd.read_csv('wine+quality/winequality-white.csv', sep = ';')

valor_constante = 0  # O valor que você deseja atribuir a todas as linhas
df1['type'] = valor_constante
valor_constante = 1
df2['type'] = valor_constante# Juntar os dois dataframe
df_merged = pd.concat([df1, df2])

# Salvar o novo arquivo CSV
df_merged.to_csv('wine_quality.csv', index=False)