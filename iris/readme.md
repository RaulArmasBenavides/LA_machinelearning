LogisticRegression(multi_class='auto')

scikit-learn detecta automáticamente si se trata de una tarea multiclase (como el caso de iris, que tiene 3 clases), y elige la estrategia adecuada según el solver.

modelo_logistico = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)