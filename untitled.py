# Importação das bibliotecas
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# SKLEARN
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder


import os
import sys
#import re
import time
import seaborn as sns
sns.set_style("darkgrid")
%matplotlib inline


import tensorflow as tf
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary as hparams_summary
%load_ext tensorboard

from google.protobuf import struct_pb2


import warnings
warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:.2f}'.format # show only two digits
pd.set_option('display.max_columns', 100) # show up to 100 columns
np.random.seed(2023)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()

Verificando o tamanho das bases

print(train.shape)
print(test.shape)

Utilizando a coluna Id como índice

train.set_index("Id", inplace=True)
test.set_index("Id", inplace=True)

Verificar se há linhas duplicadas

linhas_duplicadas = train[train.duplicated()]

if len(linhas_duplicadas) > 0:
    print("Linhas duplicadas encontradas:")
    print(linhas_duplicadas)
else:
    print("Não há linhas duplicadas no DataFrame.")

# Remover linhas duplicadas
#df_sem_duplicatas = df.drop_duplicates()


train.info()

# Calculando a mediana nas variáveis numéricas na base de treino
median_values = train.select_dtypes(exclude=['object']).median()

# Imputando a mediana nas variáveis numéricas em train, mantendo o índice
dados_numerical = train.select_dtypes(exclude=['object']).fillna(median_values)

# Tratamento das variáveis categóricas em train, mantendo o índice
dados_categoric = train.select_dtypes(include=['object']).fillna('NONE')

# Combina os dados numéricos e categóricos tratados, garantindo a ordem original das linhas
# Aqui, você não precisa fazer um merge já que você está tratando separadamente e quer combinar de volta
train = pd.concat([dados_numerical, dados_categoric], axis=1)

# Imputando a mediana nas variáveis numéricas em test com a mediana de train
dados_numerical_test = test.select_dtypes(exclude=['object']).fillna(median_values)

# Tratamento das variáveis categóricas em test, mantendo o índice
dados_categoric_test = test.select_dtypes(include=['object']).fillna('NONE')

# Combina os dados numéricos e categóricos tratados de test, garantindo a ordem original das linhas
test = pd.concat([dados_numerical_test, dados_categoric_test], axis=1)


# a variável MSSubClass aparece como int, mas é categ+orica
train['MSSubClass'] = train['MSSubClass'].astype(str)
test['MSSubClass'] = test['MSSubClass'].astype(str)


test.info()


variaveis = ("MSSubClass", "LotArea", "LotShape", "LandContour", "LotConfig", "LandSlope", "BldgType", "1stFlrSF", "2ndFlrSF", "BedroomAbvGr", "FullBath", "HalfBath", "Kitchen", "TotRmsAbvGrd", "Heating", "Electrical", "Utilities", "CentralAir", "YrSold", "SaleType", "SaleCondition", "MSZoning", "Neighborhood", "Condition1", "Condition2", "YearBuilt", "YearRemodAdd", "OverallQual", "OverallCond", "ExterQual", "ExterCond", "HouseStyle", "KitchenQual", "BsmtQual", "BsmtCond", "HeatingQC", "FireplaceQu", "Functional", "GarageType", "GarageFinish", "GarageArea", "PoolArea", "Fence", "MiscFeature", "Fireplaces", "TotalBsmtSF", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "SalePrice")

dados_train = train[list(filter(lambda x: x in train.columns, variaveis))]
dados_test = test[list(filter(lambda x: x in test.columns, variaveis))]

train = dados_train
test = dados_test

# Displaying the shapes of the new datasets
print("Train dataset shape:", train.shape)
print("Test dataset shape:", test.shape)

print(train.shape)
print(test.shape)

#### Análise univariada

train.tail()

train['CentralAir'].unique()

train['CentralAir'] = train['CentralAir'].replace({'Y': 1, 'N': 0})
test['CentralAir'] = test['CentralAir'].replace({'Y': 1, 'N': 0})

cols = train.columns

categoricas_nominais= ['MSSubClass', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2','HouseStyle', 'Functional', 'GarageType', 'GarageFinish',
'SaleType', 'SaleCondition','BldgType', 'Heating','Electrical', 'MSZoning',"MiscFeature",'YrSold','YearBuilt', 'YearRemodAdd']

categoricas_ordinais1 = ['OverallQual', 'OverallCond']
categoricas_ordinais2 = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'Fence']
#categoricas_ordinais3 = ['YrSold','YearBuilt', 'YearRemodAdd']


colunas_numericas = train.drop(columns=categoricas_nominais + categoricas_ordinais1 + categoricas_ordinais2).columns
df_numericos = train[colunas_numericas]
colunas_numericas

for col in colunas_numericas:
    plt.figure(figsize=(18,5))
    plt.subplot(1,2,1)
    #sns.distplot(df[col])
    sns.histplot(train[col], kde=True)
    plt.subplot(1,2,2)
    sns.boxplot(x=col, data=train)
    plt.show()

#Variaveis nominais
for col in categoricas_nominais:
    train[col].value_counts().plot(kind="bar", figsize=(5,3))
    plt.show()



#Variaveis nominais
for col in categoricas_ordinais1:
    train[col].value_counts().plot(kind="bar", figsize=(5,3))
    plt.show()


for col in categoricas_ordinais2:
    train[col].value_counts().plot(kind="bar", figsize=(5,3))
    plt.show()


#### Análise bivaliada

plt.figure(figsize=(10,10))
sns.pairplot(train[colunas_numericas].select_dtypes(include='number'))

plt.figure(figsize=(12,12))
sns.heatmap(train[colunas_numericas].corr(), cbar=True, annot=True, cmap='Blues')



# codificando categoricas_ordinais1 da base treino
dados_ord1 = train[categoricas_ordinais1]
dados_ord1.index = range(1, len(train)+1)

# codificando categoricas_ordinais1 da base teste
dados_ord1_test = test[categoricas_ordinais1]
dados_ord1_test.index = range(1, len(test)+1)



print(dados_ord1.shape)
print(dados_ord1_test.shape)

# codificando categoricas_ordinais2 da base treino
from sklearn.preprocessing import LabelEncoder

dados_ord2 = train[categoricas_ordinais2]

# Mapeamento da ordem correta
ordem_correta = {'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4, 'NA': 5}

# Criando um objeto para transformar as colunas categóricas usando LabelEncoder
cat_encoder = LabelEncoder()

# Aplicando o LabelEncoder em cada coluna categórica
for coluna in dados_ord2.columns:
    dados_ord2[coluna] = dados_ord2[coluna].map(ordem_correta)
    dados_ord2[coluna] = cat_encoder.fit_transform(dados_ord2[coluna])

dados_ord2.index = range(1, len(train)+1)



# codificando categoricas_ordinais2 da base teste

dados_ord2_test = test[categoricas_ordinais2]

# Mapeamento da ordem correta
ordem_correta = {'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4, 'NA': 5}

# Criando um objeto para transformar as colunas categóricas usando LabelEncoder
cat_encoder = LabelEncoder()

# Aplicando o LabelEncoder em cada coluna categórica
for coluna in dados_ord2.columns:
    dados_ord2_test[coluna] = dados_ord2_test[coluna].map(ordem_correta)
    dados_ord2_test[coluna] = cat_encoder.fit_transform(dados_ord2_test[coluna])

dados_ord2_test.index = range(1, len(test)+1)

print(dados_ord1.shape)
print(dados_ord1_test.shape)

# Mostrando o resultado
print(dados_ord2_test)

categoricas_nominais

# Tratando variáveis nominais na base treino

dados_nom = train[categoricas_nominais]
dados_nom.head()

# Tratando variáveis nominais na base treino
dados_nom_test = test[categoricas_nominais]
dados_nom_test.head()

dados_nom = dados_nom.T.drop_duplicates().T
dados_nom_test = dados_nom_test.T.drop_duplicates().T


dados_nom_test.info()

#Checking for wrong entries like symbols -,?,#,*,etc.
for col in dados_nom_test.columns:
    print('{} : {}'.format(col, dados_nom_test[col].unique()))

#treino
# Criar um objeto OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False, drop='first')

# Aplicar o OneHotEncoder nas colunas categóricas nominais
dados_categoricas_nominais_one_hot_encoded = onehot_encoder.fit_transform(train[categoricas_nominais])

# Obter os nomes das colunas a partir do objeto OneHotEncoder
colunas = onehot_encoder.get_feature_names_out()

# Transformar o objeto numpy.ndarray em um DataFrame
dados_nom = pd.DataFrame(dados_categoricas_nominais_one_hot_encoded, columns=colunas)

dados_nom.index = range(1, len(train)+1)

dados_nom


#treino
# Criar um objeto OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False, drop='first')

# Aplicar o OneHotEncoder nas colunas categóricas nominais
dados_categoricas_nominais_one_hot_encoded_test = onehot_encoder.fit_transform(test[categoricas_nominais])

# Obter os nomes das colunas a partir do objeto OneHotEncoder
colunas = onehot_encoder.get_feature_names_out()

# Transformar o objeto numpy.ndarray em um DataFrame
dados_nom_test = pd.DataFrame(dados_categoricas_nominais_one_hot_encoded_test, columns=colunas)

dados_nom_test.index = range(1, len(test)+1)

dados_nom_test

print(dados_nom.shape)
print(dados_nom_test.shape)

def encontrar_colunas_diferentes(df1, df2):
    colunas_df1 = set(df1.columns)
    colunas_df2 = set(df2.columns)
    
    colunas_diferentes = colunas_df1.symmetric_difference(colunas_df2)
    
    return colunas_diferentes
colunas_diferentes = encontrar_colunas_diferentes(dados_nom, dados_nom_test)

print("Colunas diferentes nos dois DataFrames:")
print(colunas_diferentes)

colunas_diferentes

dummy_names = dados_nom.columns.tolist()
dummy_names_test = dados_nom_test.columns.tolist()

# Printing the column names
print(dummy_names)

train.info()

#Padronizando as variáveis numéricas 

# Standardizar
#treino
def normalize_column(col):
    return (col - col.mean()) / col.std() if col.dtype in ['int64', 'float64','int32'] else col

# Apply the function to each column
normalized_data = train.apply(normalize_column)

#test
def normalize_column(col):
    return (col - col.mean()) / col.std() if col.dtype in ['int64', 'float64','int32'] else col

# Apply the function to each column
normalized_data_test = test.apply(normalize_column)

# Viewing the resulting DataFrame
normalized_data.head()

print(normalized_data.shape)
print(normalized_data_test.shape)


dados_numericos_padronizados = normalized_data.select_dtypes(exclude=['object'])

dados_numericos_padronizados_test = normalized_data_test.select_dtypes(exclude=['object'])


dados_numericos_padronizados.index = range(1, len(train)+1)
dados_numericos_padronizados_test.index = range(1, len(test)+1)

print(dados_numericos_padronizados.shape)
print(dados_numericos_padronizados_test.shape)

# Juntando as tabelas

train = pd.concat([dados_numericos_padronizados, dados_nom, dados_ord1, dados_ord2], axis=1)

test = pd.concat([dados_numericos_padronizados_test, dados_nom_test, dados_ord1_test, dados_ord2_test], axis=1)


train.head()

# Displaying the shapes of the new datasets
print("Train dataset shape:", train.shape)
print("Test dataset shape:", test.shape)

#train= train.drop(columns="MSSubClass")
#test= test.drop(columns="MSSubClass")

#Vamos fazer o test VIF, para isso, vamos separar uma base de dados sem as variáveis dummies e sem a variável alvo

df_first_19 = train.iloc[:, :22].copy()
df_last_10 = train.iloc[:, -10:].copy()

save_my_sale = train.iloc[:, 22].copy()

#my_template = train
train_SD = pd.concat([df_first_19, df_last_10], axis=1)

print(save_my_sale)
#print(dados)


from statsmodels.stats.outliers_influence import variance_inflation_factor

independent_vars = train_SD

# Compute VIF for each variable
vif_data = pd.DataFrame()
vif_data["Variable"] = independent_vars.columns
vif_data["VIF"] = [variance_inflation_factor(independent_vars.values, i) for i in range(independent_vars.shape[1])]


print(vif_data)

max_value = vif_data['VIF'].max()

print(f"The maximum value in the column is: {max_value}")

train.info()

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Assuming 'train' is your DataFrame containing predictor variables initially
independent_vars = train_SD.copy()  # Make a copy to avoid modifying the original DataFrame

high_vif_variables = pd.DataFrame({'Variable': [], 'VIF': []})

while True:
    vif_data = pd.DataFrame()
    vif_data["Variable"] = independent_vars.columns
    vif_data["VIF"] = [variance_inflation_factor(independent_vars.values, i) for i in range(independent_vars.shape[1])]

    high_vif_variables = vif_data[vif_data['VIF'] > 2]

    if high_vif_variables.empty:
        break

    variable_to_drop = high_vif_variables.loc[high_vif_variables['VIF'].idxmax(), 'Variable']

    independent_vars.drop(variable_to_drop, axis=1, inplace=True)

print("Variables after dropping high VIF variables:")
print(independent_vars.columns)


column_names2 = independent_vars.columns.tolist()

# Printing the column names
print(column_names2)

independent_vars.head()

#excluindo Fence
dados2 = independent_vars.iloc[:, :-1]
dados2.head()

#Juntando as variáveis selecionadas pelo VIF com as dummies
dados1 = train[dummy_names]
#dados2 = train[column_names2]


dados3 = pd.concat([dados2, dados1], axis=1)
#dados3 = pd.concat([dados2], axis=1)
dados3.head()

dados3.shape

my_df = pd.DataFrame(save_my_sale)
my_df.head()

#Juntando a variável alvo com a base
dados3 = pd.concat([dados3, my_df], axis=1)

dados3.head()

dados3.tail()

#Verificar se ficol algum valor ausente
columns_with_nan = dados3.columns[dados3.isnull().any()].tolist()

print("Columns containing NaN values:")
print(columns_with_nan)

#redefinando o nome train para a base de treino com as variáveis selecionadas
train = dados3  # Filter rows where 'id' is from 1 to 1460

# Displaying the shapes of the new datasets
print("Train dataset shape:", train.shape)
print("Test dataset shape:", test.shape)

#### Vamos fazer uma seleção de variáveis com o modelo lasso

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


# Separar variáveis independentes (X) e variável dependente (y)
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)

# Ajustar um modelo Lasso
lasso = Lasso(alpha=0.02)  # O valor de alpha controla a força da penalidade L
lasso.fit(X_train, y_train)

# Verificar os coeficientes não nulos (variáveis selecionadas)
variaveis_selecionadas = X.columns[lasso.coef_ != 0]

# Exibir as variáveis selecionadas
print("Variáveis selecionadas pelo Lasso:", variaveis_selecionadas)


# Separando a variável dependente e independnte
#X_train = train.drop('SalePrice', axis=1)

X_train = train[variaveis_selecionadas]
y_train = train['SalePrice']

colunas_treino = X_train.columns
colunas_treino


X_test = test[colunas_treino]
#X_test = test[variaveis_selecionadas]
#y_test = test['SalePrice']  # Variável dependente


print(X_test.shape)
print(X_train.shape)

Vamos usar a base de treino para treinar e avaliar o modelo. No final usaremos a base de teste para previsão. Chamaremos a base X_teste de Xproducao, para evitar confusão com o teste do treinamento

X_producao = X_test

from sklearn.model_selection import train_test_split

# Supondo que `X` são suas features e `y` é o target
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.3, random_state=42)

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Inicializando e treinando o MLPRegressor
ANN = MLPRegressor(hidden_layer_sizes=(5, 5, 5), activation='relu', random_state=2023)
ANN.fit(X_train, y_train)

# Fazendo previsões
y_pred = ANN.predict(X_test)

# Calculando o erro médio quadrático
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculando o coeficiente de determinação R²
r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2}')

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


# Definindo o modelo
mlp = MLPRegressor(max_iter=1000)

# Definindo os hiperparâmetros para teste
parameter_space = {
    'hidden_layer_sizes': [(10,), (50,), (100,), (10,10), (50,50)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

# Usando r2_score como a métrica de avaliação
scorer = make_scorer(r2_score)

# Configurando GridSearchCV
grid_search = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5, scoring=scorer)

# Fit no modelo (isso pode demorar um pouco dependendo do número de combinações)
grid_search.fit(X_train, y_train)

# Melhor parâmetro encontrado
print("Melhor conjunto de parâmetros:")
print(grid_search.best_params_)

# Melhor modelo
best_model = grid_search.best_estimator_

# Avaliando o melhor modelo no conjunto de teste
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'R² do melhor modelo no conjunto de teste: {r2}')


import pickle
# Salvando o modelo em um arquivo pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score## Comparando com Randon Forest

# Criar um modelo de Random Forest Regressor

modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Treinar o modelo
modelo_rf.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
previsoes = modelo_rf.predict(X_test)

# Avaliar o desempenho do modelo
mse = mean_squared_error(y_test, previsoes)
r2 = r2_score(y_test, previsoes)

print(f'Erro Médio Quadrático (MSE): {mse:.2f}')
print(f'R²: {r2:.4f}')

