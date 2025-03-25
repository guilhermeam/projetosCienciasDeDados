import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Criar e gerar dados aleatorios de uma empresa ficticia para estudar/trabalhar com graficos/predict/aprendizaod de maquinas/criar e salvar em csv

# Definindo a seed
np.random.seed(42)

# Gerando os dados
produtos = ['Caneta', 'Lapis', 'Borracha', 'Caderno', 'Papel A4', 'Mochila']
preco = [2.50, 1.80, 1.50, 10.30, 0.50, 100.00]
funcionarios = ['Juliano','Juliana','Mario','Maria','Fernando','Fernanda']


# criando DataFrame vendas
df_vendas = pd.DataFrame({
    'Produtos': produtos,
    'Preço': preco,
    'Qtd Vendida': np.random.randint(1000, 5000, size=6)
})

# DataFrame funcionarios
df_funcionarios = pd.DataFrame({
    'Vendedor': funcionarios
})

# salvando o data frame vendas em csv
df_vendas.to_csv('vendaMaterial.csv', index=False)
df_funcionarios.to_csv('vendedoresMaterial.csv', index=False)

#imprimindo os produtos e os vendedores
print("///--PRODUTOS--///")
print(df_vendas)
print("///--VENDEDORES--///")
print(df_funcionarios)

# gerando vendas detalhadas e lista de produtos repetidos
produtos_repetidos = np.repeat(df_vendas['Produtos'], df_vendas['Qtd Vendida'])

# gerando datas aleatórias para cada venda
datas_aleatorias = np.random.choice(
    pd.date_range(start='2024-01-01', end='2024-12-30'),
    size=df_vendas['Qtd Vendida'].sum()
)

# gerando vendedores aleatorios para as cada venda
vendedores_aleatorios = np.random.choice(
    funcionarios,
    size=df_vendas['Qtd Vendida'].sum()
)

# Criando um dataFrame detalhado das vendas
df_vendas_detalhado = pd.DataFrame({
    'Data': datas_aleatorias,
    'Produtos': produtos_repetidos,
    'Vendedor': vendedores_aleatorios
})

# juntando os df = vendas com vendas_detalhados
df_vendas_detalhado = df_vendas_detalhado.merge(
    df_vendas[['Produtos', 'Preço']],
    on='Produtos',
    how='left'
)

# adicionando as informações temporais
df_vendas_detalhado['Dia'] = df_vendas_detalhado['Data'].dt.day_name()
df_vendas_detalhado['Mês'] = df_vendas_detalhado['Data'].dt.month_name()
df_vendas_detalhado['Trimestre'] = 'T' + df_vendas_detalhado['Data'].dt.quarter.astype(str)

# reordenando as colunas
df_vendas_detalhado = df_vendas_detalhado[[
    'Data', 'Dia', 'Mês', 'Trimestre', 'Produtos', 'Preço', 'Vendedor'
]]

# imprimindo e salvando
print("///--VENDAS DETALHADAS--///")
print(df_vendas_detalhado.head())

df_vendas_detalhado.to_csv('vendas_2024.csv', index=False)

# Gerando os gráficos
plt.figure(figsize=(12, 6))
df_vendas_detalhado['Vendedor'].value_counts().plot(kind='bar', color='green')
plt.title('Vendas por vendedores')
plt.ylabel('Quantidade vendida')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('vendas_vendedor.png')
plt.show()

plt.figure(figsize=(12, 6))
df_vendas_detalhado['Produtos'].value_counts().plot(kind='bar', color='blue')
plt.title('Vendas por produtos')
plt.ylabel('Quantidade vendida')
plt.xticks(rotation=45)
plt.tight_layout
plt.savefig('vendas_produtos.png')
plt.show()


