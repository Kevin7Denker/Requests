import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Definir a criptomoeda e o período
criptomoeda = 'bitcoin'
periodo = 'max'  # 'max' para obter o máximo de dados disponíveis

# 2. Construir a URL da API (usando Yahoo Finance como exemplo)
url = f'https://query1.finance.yahoo.com/v8/finance/chart/{criptomoeda}-USD?range={periodo}&interval=1d'

# 3. Coletar os dados da API
response = requests.get(url)
data = response.json()

# 4. Verificar se a requisição foi bem-sucedida
if response.status_code == 200 and data['chart']['result']:
    # 5. Extrair os dados de preço e tempo
    timestamp = data['chart']['result'][0]['timestamp']
    close_prices = data['chart']['result'][0]['indicators']['quote'][0]['close']

    # 6. Criar um DataFrame com os dados
    df = pd.DataFrame({'Timestamp': timestamp, 'Close': close_prices})
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.set_index('Timestamp', inplace=True)

    # 7. Engenharia de features: criar variáveis como a variação do preço no dia anterior
    df['Close_Anterior'] = df['Close'].shift(1)
    df['Variacao'] = df['Close'] - df['Close_Anterior']
    df.dropna(inplace=True)  # Remover a primeira linha que terá valor NaN

    # 8. Criar a variável alvo: 1 se o preço subiu, 0 se o preço caiu
    df['Target'] = (df['Variacao'] > 0).astype(int)

    # 9. Dividir os dados em conjuntos de treinamento e teste
    X = df[['Close_Anterior']]  # Usar apenas o preço anterior como feature
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 10. Criar e treinar o modelo de Regressão Logística
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)

    # 11. Fazer as previsões no conjunto de teste
    y_pred = modelo.predict(X_test)

    # 12. Calcular a acurácia do modelo
    acuracia = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo: {acuracia:.2f}')

    # 13. Visualizar a importância da variável
    print(f'Coeficiente da variável (Close_Anterior): {modelo.coef_[0][0]:.2f}')

    # 14. Salvar os dados em um arquivo CSV
    df.to_csv('precos_bitcoin.csv')

else:
    print('Erro ao coletar os dados da API')
