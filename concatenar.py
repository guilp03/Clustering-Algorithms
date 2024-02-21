import pandas as pd

# Carregar os arquivos CSV
df1 = pd.read_csv('benign_traffic.csv')
df1['class'] = 0
df1 = df1.sample(frac=0.1)
df2 = pd.read_csv('gafgyt_attacks/combo.csv')
df2['class'] = 1
df2 = df2.sample(frac=0.1)
df3 = pd.read_csv('gafgyt_attacks/udp.csv')
df3['class'] = 2
df3 = df2.sample(frac=0.1)
df4 = pd.read_csv('gafgyt_attacks/tcp.csv')
df4['class'] = 3
df4 = df2.sample(frac=0.1)


df_merged = pd.concat([df1, df2,df3,df4])

# Salvar o novo arquivo CSV
df_merged.to_csv('attacks_labeled.csv', index=False)