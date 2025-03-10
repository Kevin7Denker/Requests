import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

data = {
    'Combustivel': ['Gasolina', 'Diesel', 'Etanol', 'Gasolina', 'Diesel', 'Etanol', 'Gasolina', 'Diesel'],
    'Idade': [5, 3, 7, 2, 6, 4, 8, 1],
    'Quilometragem': [80000, 50000, 120000, 30000, 100000, 60000, 150000, 10000],
    'Preco': [20000, 30000, 15000, 35000, 18000, 25000, 12000, 40000]
}
df = pd.DataFrame(data)

X = df[['Combustivel', 'Idade', 'Quilometragem']]
y = df['Preco']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['Combustivel']
numeric_features = ['Idade', 'Quilometragem']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                       ('regressor', LinearRegression())])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Erro Quadrático Médio (MSE): {mse}')
