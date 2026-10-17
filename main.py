# %%
# Importando as bibliotecas necessárias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# # Importando os dados do Github
# Lendo os dados para dataframe


# %%
# Importando os dados

data = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv"

df = pd.read_csv(data)
df.to_csv("data.csv", index=False)
df.head()

# %% [markdown]
# # Realizando a etapa de limpeza dos dados
# Colocando os dados em minúsculos e removendo espaços.


# %%
# Verificando o nome das colunas

df.columns

# %%
# Substituir os nomes das colunas para minúsculo e retirando o espaço.

df.columns = df.columns.str.replace(" ", "_").str.lower()
df.columns

# %%
df.head()

# %%
# Verificando a tipagem e o resumo das colunas
df.info()


# %%
strings = list(df.dtypes[df.dtypes == "str"].index)
strings

# %%
# Limpando os dados das colunas
for col in strings:
    df[col] = df[col].str.strip().str.lower().str.replace(" ", "_")

# %%

df.head()

# %% [markdown]

# # Análise Exploratória dos Dados

# A coluna preço é a mrsp

# %%
# Verificando Estatísticas Básicas
df.describe()

# %%
# Verificando os valores únicos da coluna make
df["make"].unique()

# %%
# Verificando quantidade de Nulos
df.isnull().sum()


# %%

df[df["engine_hp"].isnull()]

# %%
# Longtail
sns.histplot(df.msrp, bins=50)


# %%
# Melhorando a visualização
sns.histplot(df.msrp[df.msrp < 100000], bins=50)

# %%
np.log1p([0, 1, 10, 100, 1000, 10000, 100000, 1000000])


# %%
# Transformação logarítmica para melhorar a distribuição logarítmica

sns.histplot(np.log1p(df.msrp), bins=50)


# %%
price_logs = np.log1p(df.msrp)
price_logs
# %%
price_logs.describe()
# %%
sns.histplot(price_logs, bins=50)

# %% [markdown]

## Seting up the validation framework


# %%
n = len(df)
n_validation = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_validation - n_test
print(n, n_train, n_validation, n_test)

# %%

# Criando os dataset para treino, validação e teste
df_validation = df.iloc[0:n_validation].copy()
df_test = df.iloc[n_validation : n_validation + n_test].copy()
df_train = df.iloc[n_validation + n_test :].copy()

# %%
# Embaralhando os indexes

idx = np.arange(n)
idx

# %%
np.random.seed(42)
np.random.shuffle(idx)
idx

# %%
idx[n_validation:]


# %%
df_validation = df.iloc[idx[0:n_validation]].copy()
df_test = df.iloc[idx[n_validation : n_validation + n_test]].copy()
df_train = df.iloc[idx[n_validation + n_test :]].copy()
# %%
df_validation
# %%
df_test
# %%
df_train
# %%
len(df_train), len(df_validation), len(df_test)

# %%
df_train = df_train.reset_index(drop=True)
df_validation = df_validation.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# %%
df_train.head()

# %%
# Criando os targets
y_train = np.log1p(df_train.msrp.values)
y_test = np.log1p(df_test.msrp.values)
y_validation = np.log1p(df_validation.msrp.values)

# %%
y_validation


# %%
df_test = df_test.drop("msrp", axis=1)
df_validation = df_validation.drop("msrp", axis=1)
df_train = df_train.drop("msrp", axis=1)
# %% [markdown]
## Regressão Linear

# %%
