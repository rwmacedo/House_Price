import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import pickle
from PIL import Image
from io import BytesIO


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
from sklearn.preprocessing import LabelEncoder

with open("styles.css") as estilo:
   st.markdown(f"<style>{estilo.read()}</style>", unsafe_allow_html=True)



pd.options.mode.chained_assignment = None
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.set_index("Id", inplace=True)
test.set_index("Id", inplace=True)

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

train['MSSubClass'] = train['MSSubClass'].astype(str)
test['MSSubClass'] = test['MSSubClass'].astype(str)

variaveis = ("MSSubClass", "LotArea", "LotShape", "LandContour", "LotConfig", "LandSlope", "BldgType", "1stFlrSF", "2ndFlrSF", "BedroomAbvGr", "FullBath", "HalfBath", "Kitchen", "TotRmsAbvGrd", "Heating", "Electrical", "Utilities", "CentralAir", "YrSold", "SaleType", "SaleCondition", "MSZoning", "Neighborhood", "Condition1", "Condition2", "YearBuilt", "YearRemodAdd", "OverallQual", "OverallCond", "ExterQual", "ExterCond", "HouseStyle", "KitchenQual", "BsmtQual", "BsmtCond", "HeatingQC", "FireplaceQu", "Functional", "GarageType", "GarageFinish", "GarageArea", "PoolArea", "Fence", "MiscFeature", "Fireplaces", "TotalBsmtSF", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "SalePrice")

dados_train = train[list(filter(lambda x: x in train.columns, variaveis))]
dados_test = test[list(filter(lambda x: x in test.columns, variaveis))]

train = dados_train
test = dados_test

train['CentralAir'] = train['CentralAir'].replace({'Y': 1, 'N': 0})
test['CentralAir'] = test['CentralAir'].replace({'Y': 1, 'N': 0})

cols = train.columns

categoricas_nominais= ['MSSubClass', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2','HouseStyle', 'Functional', 'GarageType', 'GarageFinish',
'SaleType', 'SaleCondition','BldgType', 'Heating','Electrical', 'MSZoning',"MiscFeature",'YrSold','YearBuilt', 'YearRemodAdd']

categoricas_ordinais1 = ['OverallQual', 'OverallCond']
categoricas_ordinais2 = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'Fence']

colunas_numericas = train.drop(columns=categoricas_nominais + categoricas_ordinais1 + categoricas_ordinais2).columns
df_numericos = train[colunas_numericas]

# Carregando a média e o desvio padrão salvos
# Carregando os valores salvos
with open('means.pkl', 'rb') as f:
    means = pickle.load(f)

with open('stds.pkl', 'rb') as f:
    stds = pickle.load(f)



# Imagem
imagem = Image.open("casa.jpg")
# Criação de colunas para a imagem e o texto
col1, col2 = st.columns([1, 3])
# Na primeira coluna, adicione a imagem
with col1:
    st.image(imagem, width=100)  # Ajuste a largura conforme necessário
# Na segunda coluna, adicione o texto
with col2:
    st.title("House Prices")



st.write("Produzido por: Renata Werneck de Macedo", unsafe_allow_html=True)
st.subheader("Construindo um modelo de pricificação de imóveis")
st.write("Objetivos: Construir um modelo para prever o preço de casas utilizando a base de dasdos disponível em https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data ")

st.markdown("Introdução: Neste trabalho foram utilizadas técnicas de redes neurais para prever o preço das casas através da base de dados house_prices do Kaggle. No campo da inteligência artificial, as redes neurais vêm desempenhando um papel fundamental na resolução de problemas complexos, tendo a vantagem de aprendizagem de padrões complexos e de lidar com relações não lineares.")
st.markdown("A determinação do preço das casas é influenciada por uma série de fatores que refletem tanto as características físicas da propriedade quanto o contexto do mercado imobiliário. ")
st.markdown("A localização Geográfica (proximidade de centros urbanos, de escolas, acessibilidade a serviços e comodidades), tamanho e Layout da Casa (área construída, número de quartos, banheiros e andares), estado de conservação, idade da propriedade, possuir ou não piscina, jardins, varandas, garagem, e outros recursos especiais e a qualidade dos acabamentos influenciam o preço das propriedades." )
st.markdown("Além das características do imóvel, a dinâmica do mercado imobiliário também pode influenciar no preço, as condições do Mercado Imobiliário ( oferta e demanda no mercado imobiliário local), tendências econômicas, como taxas de juros, inflação e condições gerais do mercado imobiliário, podem impactar os preços.")

### Histograma ###

st.title('Histograma de Preços de Venda')
# Histograma
st.subheader('Distribuição dos Preços de Venda')
# Selecionando o número de bins
bins = st.slider('Número de bins', min_value=10, max_value=100, value=50)
# Criando o histograma com matplotlib
fig, ax = plt.subplots()
ax.hist(train['SalePrice'], bins=bins, color='skyblue', edgecolor='black')
ax.set_xlabel('Preços de Venda')
ax.set_ylabel('Frequência')
ax.set_title('Histograma de Preços de Venda')

# Exibindo o gráfico no Streamlit
st.pyplot(fig)



### heatmap ###

def plot_correlation_matrix(df, columns):
    corr_matrix = train[columns].corr()
    figure = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        annotation_text=corr_matrix.round(2).values,
        showscale=True,
        colorscale='Blues'
    )
    figure.update_layout(
        title='Matriz de Correlação',
        xaxis=dict(tickmode='linear'),
        yaxis=dict(tickmode='linear')
    )
    return figure


if __name__ == "__main__":
    st.title('Matriz de Correlação')
    figure = plot_correlation_matrix(train, colunas_numericas)
    st.plotly_chart(figure)





st.title('Modelo')
st.subheader("Hiper-Parâmetros")
st.markdown("O modelo usado foi uma Rede Neral Multilayer Perceptron (MLP) com os hiper-parâmetros: activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50), 'learning_rate': 'adaptive', 'solver': 'adam'")

st.markdown(" O R² do modelo no conjunto de teste: 0.84 e o R² ajustado foi: 0.83")

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def user_input_features():
    LotArea = st.number_input('Tamanho do lote em pés quadrados', value=10000)
    LotArea = (LotArea-means['LotArea'])/stds['LotArea']
    
    BedroomAbvGr = st.slider('Número de quartos', 0, 10, 2)
    BedroomAbvGr = (BedroomAbvGr-means['BedroomAbvGr'])/stds['BedroomAbvGr']
    
    FullBath = st.slider('Número de banheiros', 0, 10, 1)
    HalfBath = st.slider('Número de lavabos', 0, 10, 1)
    YearRemodAdd = st.slider('Ano da reforma', 1950, 2010, 2010)  
    GarageArea = st.number_input ('Tamanho da garagem em pes quadrados', value=500)  
    Fireplaces = st.slider('Número de lareiras', 0, 10, 1)
    TotalBsmtSF = st.number_input ('Total de pés quadrados de área do porão', value=0) 
    WoodDeckSF = st.number_input ('Área do deck de madeira em metros quadrados', value=100)  
    OpenPorchSF = st.number_input (' Área de varanda aberta em pés quadrados', value=50) 
    EnclosedPorch = st.number_input ('Área de varanda fechada em pés quadrados', value=50) 
    ScreenPorch = st.number_input ('Área da varanda com tela em pés quadrados', value=50)  
    MSSubClass_60 = 1 if st.selectbox('2-STORY 1946 & NEWER?', ['S', 'N']) == 'S' else 0
    LotShape_Reg = 1 if st.selectbox('Forma geral da propriedade é Regular?', ['S', 'N']) == 'S' else 0
    Neighborhood_NoRidge = 1 if st.selectbox('Localizada em Northridge?', ['S', 'N']) == 'S' else 0
    Neighborhood_NridgHt = 1 if st.selectbox('Localizada em Northridge Heights?', ['S', 'N']) == 'S' else 0
    Condition1_Norm = 1 if st.selectbox('Próxima da estrada principal ou ferrovia?', ['S', 'N']) == 'S' else 0
    HouseStyle_1Story = 1 if st.selectbox('É um imóvel de 1 andar?', ['S', 'N']) == 'S' else 0
    GarageFinish_RFn = 1 if st.selectbox('Interior da garagem está em acabamento bruto?', ['S', 'N']) == 'S' else 0
    GarageFinish_Unf = 1 if st.selectbox('Interior da garagem está em inacabado?', ['S', 'N']) == 'S' else 0
  
    data = [LotArea, BedroomAbvGr, FullBath, HalfBath, YearRemodAdd,
       GarageArea, Fireplaces, TotalBsmtSF, WoodDeckSF, OpenPorchSF,
       EnclosedPorch, ScreenPorch, MSSubClass_60, LotShape_Reg,
       Neighborhood_NoRidge, Neighborhood_NridgHt, Condition1_Norm,
       HouseStyle_1Story, GarageFinish_RFn, GarageFinish_Unf]
    return np.array(data).reshape(1, -1)


# Título do app
st.title('Previsão de Preços de Imóveis')

# Coleta das entradas do usuário
input_data = user_input_features()



# Botão para fazer previsão
if st.button('Prever Preço'):
    prediction = model.predict(input_data)
    # Suponha que 'prediction_normalized' seja a previsão normalizada que você deseja reverter
    # E 'column_name' seja o nome da coluna para a qual a previsão foi feita
    prediction_original = prediction * stds['SalePrice'] + means['SalePrice']
    st.success(f'O preço previsto do imóvel é ${prediction_original[0]:,.2f}')




