¿Por qué la regresión logística acierta más en el dataset de vinos?
1. Distribución linealmente separable
El dataset Wine tiene características que se separan relativamente bien en el espacio mediante fronteras lineales.

La regresión logística multinomial (cuando usamos multi_class='multinomial') modela esto muy bien.

2. El árbol de decisión puede sobreajustar (overfitting)
Los árboles tienden a memorizar los datos si no se poda o limita su profundidad.

Aunque el árbol puede ser muy preciso en el conjunto de entrenamiento, a veces no generaliza bien al conjunto de prueba.

3. El escalado ayuda a la regresión logística
Al escalar los datos con StandardScaler, la regresión logística puede encontrar coeficientes óptimos más fácilmente.

El árbol de decisión no se ve afectado por el escalado, ya que compara valores brutos al hacer splits.