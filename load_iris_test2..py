import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Preguntar al usuario la magnitud del ruido
magnitud_ruido = float(input("¿Qué magnitud de ruido deseas agregar? (ej. 0.1, 0.5, 1.0): "))

# Cargar el dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Agregar ruido con la magnitud especificada
np.random.seed(42)
X_train_ruido = X_train + np.random.normal(0, magnitud_ruido, X_train.shape)
X_test_ruido = X_test + np.random.normal(0, magnitud_ruido, X_test.shape)

# Modelos
modelo_arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo_logistico = LogisticRegression(max_iter=200)

# Entrenar y evaluar
modelo_arbol.fit(X_train_ruido, y_train)
modelo_logistico.fit(X_train_ruido, y_train)

accuracy_arbol = accuracy_score(y_test, modelo_arbol.predict(X_test_ruido))
accuracy_logistico = accuracy_score(y_test, modelo_logistico.predict(X_test_ruido))

print(f"\nPrecisión Árbol de Decisión con ruido (σ={magnitud_ruido}): {accuracy_arbol * 100:.2f}%")
print(f"Precisión Regresión Logística con ruido (σ={magnitud_ruido}): {accuracy_logistico * 100:.2f}%")
