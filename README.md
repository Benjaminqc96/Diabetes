# Diabetes
Comparativa de dos modelos de clasificación aplicados a la detección de diabetes usando variables biométricas

Éste es un proyecto de tesis en el cuál se compara dos algoritmos de clasificación: la regresión logística y el perceptrón multicapa. Más que el estudio de la diabetes, en realidad se hace la comparativa de los dos algoritmos para determinar con cuál se puede detectar con mayor eficacia la diabetes, se emplea el conjunto de datos *Pima Indian Diabetes* para el estudio. Dicho conjunto de datos consiste de 9 variables biométricas con 768 observaciones pertenecientes mujeres de la tribu pima presente en la frontera México-Estados Unidos, dichas variables son la variables clasificatoria que determina si es diabética o no, numero de embarazos que tuvo, concentración de glucosa en pruebas orales, presión de sangre diastólica, grosor de la piel del triceps, prueba de insulina sérica, índice de masa corporal, pedegree de diabetes y la edad.

Lo primero que se hace es el análisis exploratorio de los datos para saber la calidad de los datos que se dispone. En general los conjuntos de datos contienen errores o datos faltantes, por lo que el análisis es indispensable para poder determinar la forma de corregir o completar los datos. En éste caso las variables de  estudio únicamente serán las continuas y dado que hay dos clases presentes el gŕafico ideal sería el gráfico de violín.



<p align="center">
  Numero de casos
</p>

<p align="center"> 
  <img width="500" src="https://github.com/Benjaminqc96/Diabetes/blob/main/Numerocasos.png">
</p>

<p align="center">
  Concentración de glucosa
</p>

<p align="center"> 
  <img width="500" src="https://github.com/Benjaminqc96/Diabetes/blob/main/Glucosa.png">
</p>

<p align="center">
  Presión arterial
</p>

<p align="center"> 
  <img width="500" src="https://github.com/Benjaminqc96/Diabetes/blob/main/Presion.png">
</p>

<p align="center">
  Grosor de la piel del triceps 
</p>

<p align="center"> 
  <img width="500" src="https://github.com/Benjaminqc96/Diabetes/blob/main/Grosorp.png">
</p>

<p align="center">
  Insulina sérica 
</p>

<p align="center"> 
  <img width="500" src="https://github.com/Benjaminqc96/Diabetes/blob/main/Insulina.png">
</p>
<p align="center">
  Índice de masa corporal
</p>

<p align="center"> 
  <img width="500" src="https://github.com/Benjaminqc96/Diabetes/blob/main/IMC.png">
</p>
<p align="center">
  Pedegree de diabetes
</p>

<p align="center"> 
  <img width="500" src="https://github.com/Benjaminqc96/Diabetes/blob/main/Antecedentes.png">
</p>


Como hipótesis en el párrafo anterior se plantea que los datos contienen imperfecciones, y en efecto se cumplen dado que algunas variables contienen datos faltantes y el conjunto está desbalanceado, *i.e* hay más casos de mujeres sanas que de mujeres que padecen diabetes, esto ultimo representa un problema dado que si se entrena un modelo con un conjunto de datos desbalanceado va a sesgar los estimadores a elegir la clase predominante, por tanto se procede a hacer un balanceo de los datos mediante muestreo sintético.



<p align="center">
  Conjunto balanceado
</p>

<p align="center"> 
  <img width="500" src="https://github.com/Benjaminqc96/Diabetes/blob/main/Balance.png">
</p>


Una vez con el conjunto balanceado entonces se procede a hacer la ingesta de datos a los modelos, obteniendo los siguientes resultados.


<p align="center">
  Resumen de la regresión logística
</p>

<p align="center"> 
  <img width="500" src="https://github.com/Benjaminqc96/Diabetes/blob/main/rlb.png">
</p>

<p align="center">
  Resumen del perceptrón multicapa
</p>

<p align="center"> 
  <img width="500" src="https://github.com/Benjaminqc96/Diabetes/blob/main/Precision.png">
</p>


Ambos modelos aparentan robustez, sin embargo el desempeño debe medirse de acuerdo a varios criterios. En cuestión de precisión la regresión logística clasifica de manera acertada el 76% de los casos en el entrenamiento mientras que el perceptrón clasifica de manera acertada 86%, evidenciando una superioridad numerica, sin embargo la verdadera bracha entre los dos algoritmos se verá plasmada en las curvas ROC y el área bajo la curva, así como en la validación cruzada.

<p align="center">
  Curva ROC RL
</p>

<p align="center"> 
  <img width="500" src="https://github.com/Benjaminqc96/Diabetes/blob/main/rocrl.png">
</p>

<p align="center">
  Curva ROC PML
</p>

<p align="center"> 
  <img width="500" src="https://github.com/Benjaminqc96/Diabetes/blob/main/rocmlp.png">
</p>


Analizando las curvas ROC tenemos que la regresión logística tiene una probabilidad de acierto de clasificar acertadamente un caso de 82%, mientras que el PML tiene un 88% de probabilidad, por tanto, hay superioridad numérica del perceptron sobre la regresión logística.

Tomando en cuenta la validación cruzada la precisión promedio de la regresión logística es de 76.1%, con una variabilidad del 4.8%. Para el PML la precisión promedio es de 86.8%, con una variabilidad de 3.62%. Nuevamente hay superioridad por parte del PML.

En todos los criterios evaluados es superior un algortimo del otro, por lo cuál en preferente hacer uso del "mejor", si bien no es perfecto, las desventajas asociadas al PML son poco considerables dado que únicamente se trata de infraestructura para realizar el cómputo, necesidad que puede ser satisfecha con la infraestructura disponible en internet de acceso "gratuito".
