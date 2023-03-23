# This is a sample Python script.

# Press Mayus+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from datetime import datetime
import getopt
import sys
from sklearn import tree
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

k=1
d=1
p='./'
f="SantanderTraHalfHalf.csv"
oFile=""
m="uniform"
r=0
classifier="TARGET"

def datetime_to_epoch(d):
    return datetime.datetime(d).strftime('%s')
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'o:k:d:D:m:M:p:f:h',['output=','k=','d=','D=','m=','M=','path=','iFile','h'])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-o','--output'):
            oFile = arg
        elif opt == '-k':
            k = arg
        elif opt ==  '-d':
            d = arg
        elif opt == '-D':
            D = arg
        elif opt == '-m':
            m = arg
        elif opt == '-M':
            M = arg
        elif opt in ('-p', '--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-h','--help'):
            print(' -o outputFile \n -k numberOfItems \n -d distanceParameter \n -p inputFilePath \n -f inputFileName \n ')
            exit(1)

    if p == './':
        iFile=p+str(f)
    else:
        iFile = p+"/" + str(f)
    # astype('unicode') does not work as expected

    def coerce_to_unicode(x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x, 'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)

    #Abrir el fichero .csv y cargarlo en un dataframe de pandas
    ml_dataset = pd.read_csv(iFile)

    #comprobar que los datos se han cargado bien. Cuidado con las cabeceras, la primera linea por defecto la considerara como la que almacena los nombres de los atributos
    # comprobar los parametros por defecto del pd.read_csv en lo referente a las cabeceras si no se quiere lo comentado

    #print(ml_dataset.head(5))

   # ml_dataset = ml_dataset[
        #Nombres de todas las cabeceras de las columnas de las tablas
    colums = list(ml_dataset.columns)
    ml_dataset=ml_dataset[colums]

    # Q tipo de dato tengo


    categorical_features = []
    #numerical_features = [Nombres de la cabeceras de las columnas de la tabla separadas por '','']

    numerical_features =list(ml_dataset.columns)
    numerical_features.remove(classifier)
    text_features = []

    for feature in categorical_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in text_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in numerical_features:
        if ml_dataset[feature].dtype == np.dtype('M8[ns]') or (
                hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')):
            ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
        else:
            ml_dataset[feature] = ml_dataset[feature].astype('double')



    target_map = {'0': 0, '1': 1} # q valor le asignas a cada atributo
    ml_dataset['__target__'] = ml_dataset[classifier].map(str).map(target_map)
    del ml_dataset[classifier]

    # Remove rows for which the target is unknown.
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    print(f)
    print(ml_dataset.head(5))


    train, test = train_test_split(ml_dataset,test_size=0.2,random_state=42,stratify=ml_dataset[['__target__']])#Elegimos la muestra para entrenar
    print(train.head(5))                                                                                        #El 20% para test, indice aleatorio
    print(train['__target__'].value_counts())
    print(test['__target__'].value_counts())

    drop_rows_when_missing = []
    #impute_when_missing = [{'feature': 'num_var45_ult1', 'impute_with': 'MEAN'},
    #                       {'feature': 'num_op_var39_ult1', 'impute_with': 'MEAN'},
    #                      {'feature': 'num_op_var40_comer_ult3', 'impute_with': 'MEAN'},
    #                     {'feature': 'num_var45_ult3', 'impute_with': 'MEAN'},
    #                    {'feature': 'num_aport_var17_ult1', 'impute_with': 'MEAN'}]


    valoresdecolumna= list(ml_dataset.columns)
    valoresdecolumna.remove('__target__')
    impute_when_missing = []              
    for i in range(0, len(valoresdecolumna)):
        impute_when_missing.append({'feature': valoresdecolumna[i],'impute_with' : 'MEAN'})


    trainX = train.drop('__target__', axis=1)
    #trainY = train['__target__']

    testX = test.drop('__target__', axis=1)
    #testY = test['__target__']

    trainY = np.array(train['__target__'])
    testY = np.array(test['__target__'])

    # MAs de una clase y haya mucha diferencia entre unos y otras

    #undersample = RandomUnderSampler(sampling_strategy=0.5)#la mayoria va a estar representada el doble de veces

    #trainXUnder,trainYUnder = undersample.fit_resample(trainX,trainY)
    #testXUnder,testYUnder = undersample.fit_resample(testX, testY)

    # el max_depth va de 3 en 3
    for valorD in range(int(d), int(D) +1, 3):
        for valorM in range(int(m), int(M) +1):
            # Se crea el modelo KNN
            clf = tree.DecisionTreeClassifier(
                random_state=1,
                criterion='gini',
                splitter='best',
                max_depth=valorD,
                min_samples_leaf=valorM
            )

            # Se modifica el metodo de utilizar los pesos de KNN
            clf.class_weight = "balanced"

            # Finalmente, se entrena el modelo con la particion de trainUndersampled
            clf.fit(trainX, trainY)

            # Ya se ha entrenado el modelo
            predictions = clf.predict(testX)
            probas = clf.predict_proba(testX)

            predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
            cols = [
                u'probability_of_value_%s' % label
                for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
            ]
            probabilities = pd.DataFrame(data=probas, index=testX.index, columns=cols)

            
            results_test = testX.join(predictions, how='left')
            results_test = results_test.join(probabilities, how='left')
            results_test = results_test.join(test['__target__'], how='left')
            results_test = results_test.rename(columns= {'__target__': 'TARGET'})

            i=0
            for real,pred in zip(testY,predictions):
                #print(real,pred)
                i+=1
                if i>5:
                    break
    

    print(f1_score(testY, predictions, average=None))
    print(classification_report(testY,predictions))
    print(confusion_matrix(testY, predictions, labels=[1,0]))
print("bukatu da")
