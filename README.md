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
  <img width="500" src="https://github.com/Benjaminqc96/Diabetes/blob/main/Presión.png">
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


Como hipótesis en el párrafo anterior se plantea que los datos contienen imperfecciones, y en efecto se cumplen dado que algunas variables contienen datos faltantes y el conjunto está desbalanceado, *i.e* hay más casos de mujeres sanas que de mujeres que padecen diabetes, esto ultimo representa un problema dado que si se entrena un modelo con un conjunto de datos desbalanceado va a sesgar los estimadores a elegir la clase predominante.
