import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv(r'Consumo_cerveja.csv', sep=";")

x = dataset.iloc[:, 4:5].values

y = dataset.iloc[:, -1].values

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size = 0.15, random_state=0)

linearRegression = LinearRegression()

y_pred = linearRegression.fit(x_treinamento, y_treinamento)

plt.scatter (x_treinamento, y_treinamento, color="red")
plt.plot (x_treinamento, linearRegression.predict(x_treinamento), color="blue")
plt.title("Sao Paulo: Preciptacao de Chuva x Consumo de Cerveja (Treinamento)")
plt.xlabel("Preciptacao de Chuva")
plt.ylabel("Consumo de Cerveja")
plt.show()

plt.scatter(x_teste, y_teste, color="red")
plt.plot(x_treinamento, linearRegression.predict(x_treinamento), color="blue")
plt.title("Sao Paulo: Preciptacao de Chuva x Consumo de Cerveja (Teste)")
plt.xlabel("Preciptacao de Chuva")
plt.ylabel("Consumo de Cerveja")
plt.show()