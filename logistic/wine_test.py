from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score,classification_report
import matplotlib.pyplot as plt

# Paso 1: Cargar datos y dividirlos
wine = load_wine()
X, y = wine.data, wine.target

# Dividir el dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Escalar (solo para regresión logística)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Paso 2: Regresión logística multiclase
log_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

print("Regresión Logística:")
print(classification_report(y_test, y_pred_log, target_names=wine.target_names))

# Paso 3: Árbol de decisión
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

print("\nÁrbol de Decisión:")
print(classification_report(y_test, y_pred_tree, target_names=wine.target_names))

accuracy_arbol = accuracy_score(y_test, y_pred_tree)
accuracy_log = accuracy_score(y_test, y_pred_log)

# 7. Mostrar precisión
print(f"Precisión Árbol de Decisión con ruido: {accuracy_arbol * 100:.2f}%")
print(f"Precisión Regresión Logística con ruido: {accuracy_log * 100:.2f}%")

# Paso 4: Visualización del árbol
plt.figure(figsize=(14, 8))
plot_tree(
    tree_model,
    filled=True,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    rounded=True,
    fontsize=10
)
plt.title("Árbol de Decisión - Dataset Wine")
plt.tight_layout()
plt.show()