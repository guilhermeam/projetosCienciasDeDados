import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# carregando dados do projeto Vendas
def carregar_dados():
    df = pd.read_csv('vendas_2024.csv', parse_dates=['Data'])

    df['Dia_Semana'] = df['Data'].dt.dayofweek
    df['Dia_Mes'] = df['Data'].dt.day
    df['Mes'] = df['Data'].dt.month
    df['Semana_Ano'] = df['Data'].dt.isocalendar().week

    return df

# gerando visualizações exploratorias dos dados
def explorar_dados(df):
    # Vendas ao longo do tempo
    plt.figure(figsize=(12, 6))
    df.groupby('Data').size().plot(title='Vendas diárias ao longo do tempo')
    plt.savefig('exploratorio_vendas_tempo.png')
    plt.close()

    # mapa de calor(Heatmap) vendas por dia da semana e mês
    pivot = df.pivot_table(index='Dia_Semana', columns='Mes',
                           values='Preço', aggfunc='count')
    plt.figure(figsize=(12, 6))
    # cmap='YlGnBu' = Yellow/Green/Blue
    sns.heatmap(pivot, cmap='YlGnBu', annot=True, fmt='.0f')
    plt.title('Vendas por dia da semana e mês')
    plt.savefig('exploratorio_heatmap.png')
    plt.close()

# Preparando os dados de modelagem
def preprocessar_dados(df):
    # agregando vendas por dia
    vendas_diarias = df.groupby('Data').agg({
        'Preço': 'sum',
        'Produtos': 'count'
    }).rename(columns={'Produtos': 'Qtd_Vendas', 'Preço': 'Faturamento'})

    # add features temporais
    vendas_diarias['Dia_Semana'] = vendas_diarias.index.dayofweek
    vendas_diarias['Dia_Mes'] = vendas_diarias.index.day
    vendas_diarias['Mes'] = vendas_diarias.index.month
    vendas_diarias['Semana_Ano'] = vendas_diarias.index.isocalendar().week

    # add lags - valores passados como features
    for lag in [1, 7, 30]:
        vendas_diarias[f'Lag_{lag}'] = vendas_diarias['Qtd_Vendas'].shift(lag)

    # remover linhas com valores NaN (criadas pelo lag)
    vendas_diarias = vendas_diarias.dropna(subset=['Lag_1', 'Lag_7', 'Lag_30'])

    return vendas_diarias

# treinamento e avaliação  do modelo de previsão
def treinar_modelo(df):
    # separando features e target
    X = df.drop(['Qtd_Vendas', 'Faturamento'], axis=1)
    y = df['Qtd_Vendas']

    # usar validação cruzada temporal
    tscv = TimeSeriesSplit(n_splits=5)
    maes, r2s, rmses = [], [], []
    modelos = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        modelo = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)

        maes.append(mean_absolute_error(y_test, pred))
        r2s.append(r2_score(y_test, pred))
        rmses.append(np.sqrt(mean_absolute_error(y_test, pred)))
        modelos.append(modelo)

    # selecionar o melhor modelo
    best_idx = np.argmin(maes)
    modelo = modelos[best_idx]

    # ± = Alt + 0177
    print(f'MAE médio: {np.mean(maes):.2f} ± {np.std(maes):.2f}')
    print(f'RMES médio: {np.mean(rmses):.2f} ± {np.std(rmses):.2f}')
    print(f'R² médio: {np.mean(r2s):.2f} ± {np.std(r2s):.2f}')

    # features importantes
    features = X.columns
    importances = modelo.feature_importances_
    plt.figure(figsize=(12, 6))
    pd.Series(importances, index=features).sort_values().plot(kind='barh')
    plt.title('Importância das Features')
    plt.savefig('features_importance.png')
    plt.close()
    
    return modelo, np.mean(maes), np.mean(r2s)

# Gera previsões para os proximos dias
def prever_proximos_dias(modelo, df, dias=30):

    # garantir que o dataframe tenha pelomenos 30 dias de historico
    if len(df) <= 30:
        raise ValueError("Necessário pelo menos 30 dias de dados para previsão")
    
    # criar dataframe futuro com todas feature de treinamento
    futuro = pd.DataFrame(
        index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=dias)
    )

    futuro['Dia_Semana'] = futuro.index.dayofweek
    futuro['Dia_Mes'] = futuro.index.day
    futuro['Mes'] = futuro.index.month
    futuro['Semana_Ano'] = futuro.index.isocalendar().week.astype(int)

    # inicializar todas as colunas de lag que o modelo espera
    for lag in [1, 7, 30]:
        futuro[f'Lag_{lag}'] = np.nan
    
    # preencher lags com valores conhecidos (últimos dias)
    for i in range(len(futuro)):

            futuro.loc[futuro.index[i], 'Lag_1'] = df['Qtd_Vendas'].iloc[-1] if i == 0 else futuro.loc[futuro.index[i-1], 'Pred']
            futuro.loc[futuro.index[i], 'Lag_7'] = df['Qtd_Vendas'].iloc[-7] if i < 7 else futuro.loc[futuro.index[i-7], 'Pred']
            futuro.loc[futuro.index[i], 'Lag_7'] = df['Qtd_Vendas'].iloc[-30] if i < 30 else futuro.loc[futuro.index[i-30], 'Pred']

        # garante a ordem das features
            input_data = futuro.loc[futuro.index[i], modelo.feature_names_in_].values.reshape(1, -1)
            futuro.loc[futuro.index[i], 'Pred'] = modelo.predict(input_data)[0]
       
    return futuro[['Pred']]

    # Plotar previsões
    plt.figure(figsize=(12, 6))
    df['Qtd_Vendas'].plot(label='Histórico')
    futuro['Pred'].plot(label='Previsão', style='--')
    plt.title(f'Previsão de vendas para os próximos {dias} Dias')
    plt.legend
    plt.savefig('previsao_futura.png')
    plt.close()

    return futuro[['Pred']]

if __name__ == "__main__":
    try:
        print("Carregando dados...")
        df = carregar_dados()

        print('Explorando dados...')
        explorar_dados(df)

        print("Processando dados...")
        dados_modelo = preprocessar_dados(df)

        print("Treinando modelo...")
        modelo, mae, r2 = treinar_modelo(dados_modelo)

        print(f'Modelo treinado - MAE: {mae:.2f}, R²: {r2:.2f}')

        print("Salvando modelo...")
        joblib.dump(modelo, 'modelo_vendas.joblib')

        print("Gerando previsões futuras...")
        previsoes = prever_proximos_dias(modelo, dados_modelo, dias=30)
        print(previsoes.head())
        
    except Exception as err:
        print(f'Erro: {str(err)}')
        print("Verifique se:")
        print("1. O arquivo vendas_2024.csv existe")
        print("2. O arquivo contém dados no formato esperado")
        print("3. Você tem as permisões necessárias")






