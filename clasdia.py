#modulos
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#lectura de los datos
datos = pd.read_csv('/home/benjamin/Documentos/tesis/diabe.csv')
datos.columns = ['Embarazos','Glucosa','Presion','Grosorp','Insulina',
                 'IMC','Antecedentes','Edad','Salida']
datos['Salida'].value_counts()

plt.figure()
sb.countplot(x='Salida',data=datos,palette='hls')
plt.xlabel('Clase')
plt.ylabel('Numero de casos')
#plt.savefig('Numerocasos.png')

datos.groupby('Salida').mean()
datos['Glucosa'].describe()
datos['Presion'].describe()
datos['Grosorp'].describe()
datos['Insulina'].describe()
datos['IMC'].describe()
datos['Antecedentes'].describe()

#graficos de violin
plt.figure()
sb.violinplot(x = 'Salida', y = 'Glucosa', data = datos, 
              palette = 'muted', split = True)
plt.xlabel('Clase')
plt.ylabel('mg/dl')
#plt.savefig('Glucosa.png')

plt.figure()
sb.violinplot(x = 'Salida', y = 'Presion', data = datos, 
              palette = 'muted', split = True)
plt.xlabel('clase')
plt.ylabel('mmHg')
#plt.savefig('Presion.png')

plt.figure()
sb.violinplot(x = 'Salida', y = 'Grosorp', data = datos, 
              palette = 'muted', split = True)
plt.xlabel('Clase')
plt.ylabel('mm')
#plt.savefig('Grosorp.png')

plt.figure()
sb.violinplot(x = 'Salida', y = 'Insulina', data = datos, 
              palette = 'muted', split = True)
plt.xlabel('Clase')
plt.ylabel('mU/ml')
#plt.savefig('Insulina.png')

plt.figure()
sb.violinplot(x = 'Salida', y = 'IMC', data = datos, 
              palette = 'muted', split = True)
plt.xlabel('Clase')
plt.ylabel('kg/m^2')
#plt.savefig('IMC.png')

plt.figure()
sb.violinplot(x = 'Salida', y = 'Antecedentes', data = datos, 
              palette = 'muted', split = True)
plt.xlabel('Clase')
plt.ylabel('FDP')
#plt.savefig('Antecedentes.png')

#reemplazar los valores de cero
#Glucosa, insulina, IMC, Grosorp 
marco_1 = datos.loc[datos['Salida'] == 1]
marco_2 = datos.loc[datos['Salida'] == 0]

marco_1 = marco_1.replace({'Glucosa':0}, 
                          np.median(marco_1['Glucosa']))

marco_1 = marco_1.replace({'Insulina':0}, 
                          np.median(marco_1['Insulina']))

marco_1 = marco_1.replace({'IMC':0}, 
                          np.median(marco_1['IMC']))

marco_1 = marco_1.replace({'Grosorp':0}, 
                          np.median(marco_1['Grosorp']))



marco_2 = marco_2.replace({'Glucosa':0}, 
                          np.median(marco_2['Glucosa']))

marco_2 = marco_2.replace({'Insulina':0}, 
                          np.median(marco_2['Insulina']))

marco_2 = marco_2.replace({'IMC':0}, 
                          np.median(marco_2['IMC']))

marco_2 = marco_2.replace({'Grosorp':0}, 
                          np.median(marco_2['Grosorp']))

datos = pd.concat([marco_1, marco_2])
features = ['Embarazos','Glucosa','Presion','Grosorp','Insulina',
                 'IMC','Antecedentes','Edad']
#variables
var_ind = datos.loc[:,features].values
#estandarizacion
var_dep = datos.loc[:,['Salida']].values
var_ind_t = StandardScaler().fit_transform(var_ind)
pca = PCA(n_components=var_ind.shape[1])
pca.fit(var_ind_t)
x_pca = pca.transform(var_ind_t)
var_exp = pca.explained_variance_ratio_
#variables seleccionadas
features_selected = ['Glucosa','Presion','Grosorp','Insulina',
                 'IMC','Antecedentes']
var_ind_t = pd.DataFrame(var_ind_t)
var_ind_t.columns = features
X = var_ind_t.loc[:,features_selected]
##oversampling
over_samp = SMOTE(random_state=0)
X_train,X_test,y_train,y_test = train_test_split(X,var_dep,test_size = 0.3,
                                               random_state = 0)
os_dx,os_dy = over_samp.fit_sample(X_train,y_train)
os_dxp,os_dyp = over_samp.fit_sample(X_test,y_test)
x_comp, y_comp = over_samp.fit_sample(X, var_dep)

plt.figure()
sb.countplot(x = os_dy, palette = 'hls')
plt.ylabel('Numero de casos')
plt.xlabel('Clase')
#plt.savefig('Balance.png')

#regresion logistica
reg_log = LogisticRegression()
reg_log.fit(os_dx, os_dy)
reg_log.score(os_dxp, os_dyp)
reg_log2 = sm.Logit(exog = os_dx, endog = os_dy).fit()
print(reg_log2.summary())

#curva ROC regresion logistica

rl_rocauc = roc_auc_score(y_true = os_dyp, 
                          y_score = reg_log.predict(os_dxp))
fpr, tpr, thresholds = roc_curve(os_dyp, 
                       reg_log.predict_proba(os_dxp)[:,1])

plt.figure()
plt.plot(fpr, tpr, label = 'Regresión logística (area = %0.2f)'%
         rl_rocauc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falsos Positivos')
plt.ylabel('Verdaderos Positivos ')
plt.legend(loc="lower right")
#plt.savefig('rocrl')

#validacion cruzada para la regresion logistica
val_cru_rl = cross_val_score(estimator = reg_log, X = x_comp,
                             y = y_comp, cv = 10)

prec_prom_rl = np.mean(val_cru_rl)
variacion_rl = np.std(val_cru_rl)
##seleccion de la tasa de aprendizaje

itera = 1000
cont = np.zeros(itera)
precision = np.zeros(itera)   
for i in range(itera):
    cont[i] = i
    np.random.seed(0)
    mlp=MLPClassifier(hidden_layer_sizes=5,activation='relu',
                      learning_rate_init = (0.001+(i/1000)) ,max_iter=1700)
    mlp.fit(os_dx,os_dy)
    precision[i] = mlp.score(os_dxp,os_dyp)

lr_opt = pd.DataFrame({'preci':precision})
lr = lr_opt[lr_opt.eq(lr_opt.preci.max()).any(1)].index.to_numpy()
lear_rate = 0.001 + (lr/1000)
##Entrenando la RNA

def red_neuronal(num_neu,X_ent,y_ent,X_pru,y_pru,X_comp,y_comp):
    np.random.seed(0)
    mlp=MLPClassifier(hidden_layer_sizes=num_neu,activation='relu',
                      learning_rate_init = lear_rate[0] ,max_iter = 1700)
    mlp.fit(X_ent,y_ent)
    val_cruz = cross_val_score(mlp,X_comp,y_comp,cv = 10)
    return val_cruz
#Precisión de la red
num_iter = 15
precision_list = {}
prec_prom = np.zeros(num_iter)
variacion = np.zeros(num_iter)
num_neu = np.zeros(num_iter)
for i in range(num_iter):
   print(i)
   precision_list[i] = red_neuronal(num_neu = i+1,X_ent = os_dx,y_ent = os_dy,
         X_pru = os_dxp,y_pru = os_dyp,X_comp = x_comp,
         y_comp = y_comp) 
   prec_prom[i] = np.mean(precision_list[i])
   variacion[i] = np.std(precision_list[i])
   num_neu[i] = i+1

plt.figure()
plt.xlabel('Numero de nodos en la capa oculta')
plt.ylabel('Precision de la RNA')
plt.grid()
sb.scatterplot(x = num_neu,y = prec_prom)
#plt.savefig('Precision.png')
#ROC perceptron
mlp = MLPClassifier(hidden_layer_sizes = 6, activation = 'relu',
                    learning_rate_init = lear_rate[0], max_iter = 1700)
mlp.fit(X = os_dx, y = os_dy)
mlp_rocauc = roc_auc_score(y_true = os_dyp, 
                          y_score = mlp.predict(X = os_dxp))
fpr_mlp, tpr_mlp, thresholds = roc_curve(os_dyp, 
                       mlp.predict_proba(os_dxp)[:,1])


plt.figure()
plt.plot(fpr_mlp, tpr_mlp, label = 'Perceptrón multicapa (area = %0.2f)'%
         mlp_rocauc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falsos Positivos')
plt.ylabel('Verdaderos Positivos')
plt.legend(loc="lower right")
#plt.savefig('rocmlp')

