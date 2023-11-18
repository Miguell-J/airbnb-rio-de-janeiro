# Projeto de Ciência de Dados - Análise e Modelagem Preditiva para Preços de Hospedagens Airbnb
## Visão Geral
Este projeto tem como objetivo realizar uma análise aprofundada dos dados do Airbnb, aplicando técnicas de pré-processamento, análise exploratória e modelagem preditiva para estimar os preços de hospedagens. Utilizando bibliotecas Python como Pandas, Seaborn, Matplotlib e Scikit-Learn, este trabalho visa fornecer insights valiosos para stakeholders do setor de hospedagem.

## Estrutura do Projeto
### Pré-processamento dos Dados
- Consolidamos dados de diversos arquivos CSV em um DataFrame único, incorporando informações temporais.
Removemos colunas com excesso de valores nulos e tratamos valores faltantes em colunas selecionadas.
```python
bs_airbnb = pd.DataFrame()

for arquivo in path_bs.iterdir():
    df = pd.read_csv(path_bs / arquivo.name)
    nome_mes = arquivo.name[:3]
    mes = meses[nome_mes]
    
    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))
    
    df['ano'] = ano
    df['mes'] = mes
    bs_airbnb = bs_airbnb.append(df)
```

### Remover colunas com muitos valores nulos
```python
bs_airbnb = bs_airbnb.dropna(thresh=len(bs_airbnb) - 300000, axis=1)
```

### Tratar valores nulos em colunas selecionadas
```python
for col in ['host_is_superhost', 'host_total_listings_count', 'bathrooms', 'bedrooms', 'beds']:
    bs_airbnb[col].fillna(bs_airbnb[col].mode()[0] if 'host' in col else bs_airbnb[col].mean(), inplace=True)
```
### Análise Exploratória
- Realizamos uma análise estatística descritiva e visualizações gráficas para entender a distribuição de variáveis-chave.
```python
print(bs_airbnb.describe())
sns.pairplot(bs_airbnb[['price', 'accommodates', 'bathrooms', 'bedrooms', 'beds']])
plt.show()
```
### Transformação de Dados
- Convertemos valores monetários para formato numérico e tratamos variáveis categóricas.
```python
bs_airbnb['price'] = bs_airbnb['price'].str.replace('$', '').str.replace(',', '').astype(np.float32, copy=False)
bs_airbnb['extra_people'] = bs_airbnb['extra_people'].str.replace('$', '').str.replace(',', '').astype(np.float32, copy=False)
```
### Visualizações
- Utilizamos uma matriz de correlação e diversos gráficos para explorar as relações entre variáveis.
```python
plt.figure(figsize=(15, 10))
sns.heatmap(bs_airbnb.corr(), annot=True, cmap='magma')
plt.show()
```
### Geolocalização
- Exploramos a geolocalização das hospedagens através de um mapa de densidade de preços usando a biblioteca Plotly Express.
```python
amostra = bs_airbnb.sample(n=40000)
mapa_centro = {'lat': amostra.latitude.mean(), 'lon': amostra.longitude.mean()}
fig = px.density_mapbox(amostra, lat='latitude', lon='longitude', z='price', radius=2.5,
                        center=mapa_centro, zoom=10, mapbox_style='open-street-map')
fig.show()
```
### Modelagem Preditiva
- Preparamos os dados para modelagem, codificando variáveis categóricas, e treinamos modelos de Regressão Linear, Random Forest e Extra Trees para prever os preços das hospedagens.
```python
modelo_exr = ExtraTreesRegressor()
x = bs_airbnb_cod.drop('price', axis=1)
y = bs_airbnb['price']
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)
modelo_exr.fit(x_treino, y_treino)
previsao = modelo_exr.predict(x_teste)
```
### Avaliação dos Modelos
- Avaliamos o desempenho dos modelos utilizando métricas como R², MAE e MSE. Analisamos a importância das variáveis para cada modelo.
```python
print(modelo_exr.score(y_teste, previsao))
print(modelo_exr.feature_importances_)
```
### Resultados e Conclusões
= Salvamos o DataFrame final e os modelos treinados, destacando as principais variáveis que influenciam nos preços das hospedagens.
```python
bs_airbnb_cod.to_csv('bs_airbnb.csv')
x['price'] = y
x.to_csv('dados.csv')
```
## Conclusão
Este projeto oferece uma abordagem abrangente para análise de dados do Airbnb, fornecendo não apenas uma compreensão detalhada dos fatores que afetam os preços das hospedagens, mas também modelos preditivos robustos para estimativas futuras. Os resultados obtidos podem ser valiosos para tomadas de decisão e estratégias de precificação no setor de hospedagem.
