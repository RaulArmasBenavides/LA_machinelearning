import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_iris

# 1. Cargar el dataset Iris
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 2. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Agregar ruido
np.random.seed(42)
X_train_ruido = X_train + np.random.normal(0, 0.1, X_train.shape)
X_test_ruido = X_test + np.random.normal(0, 0.1, X_test.shape)

# 4. Modelos
modelo_arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo_logistico = LogisticRegression(max_iter=200)

# 5. Entrenar
modelo_arbol.fit(X_train_ruido, y_train)
modelo_logistico.fit(X_train_ruido, y_train)

# 6. Predecir
y_pred_arbol = modelo_arbol.predict(X_test_ruido)
y_pred_log = modelo_logistico.predict(X_test_ruido)

accuracy_arbol = accuracy_score(y_test, y_pred_arbol)
accuracy_log = accuracy_score(y_test, y_pred_log)

# 7. Mostrar precisión
print(f"Precisión Árbol de Decisión con ruido: {accuracy_arbol * 100:.2f}%")
print(f"Precisión Regresión Logística con ruido: {accuracy_log * 100:.2f}%")

# 8. Visualización de los datos antes y después del ruido
plt.figure(figsize=(14, 5))

# Antes del ruido
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, palette='Set1')
plt.title("Datos originales (entrenamiento)")
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])

# Después del ruido
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_train_ruido[:, 0], y=X_train_ruido[:, 1], hue=y_train, palette='Set1')
plt.title("Datos con ruido (entrenamiento)")
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])

plt.tight_layout()
plt.show()

# 9. Matriz de confusión
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_arbol, ax=axes[0], display_labels=target_names)
axes[0].set_title("Árbol de Decisión")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_log, ax=axes[1], display_labels=target_names)
axes[1].set_title("Regresión Logística")

plt.tight_layout()
plt.show()

# 10. Comparación visual de precisión
plt.figure(figsize=(6, 4))
modelos = ['Árbol de Decisión', 'Regresión Logística']
precisiones = [accuracy_arbol * 100, accuracy_log * 100]

sns.barplot(x=modelos, y=precisiones, palette='viridis')
plt.title("Comparación de precisión (%)")
plt.ylabel("Precisión (%)")
plt.ylim(0, 100)
for i, v in enumerate(precisiones):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
