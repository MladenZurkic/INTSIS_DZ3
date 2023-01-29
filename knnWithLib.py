import pandas
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score


def correlationMatrixPlot(dataWithCorr):
    plt.figure('Korelaciona matrica')
    plt.title('Korelaciona matrica')
    plt.tight_layout()
    sb.heatmap(dataWithCorr, annot=True, square=True, fmt=".2f")
    plt.show()

def continualDataPlot(dataToRepresent):
    plotDataIn = data[[dataToRepresent]]
    plotDataOut = dataForPlot['type']

    plt.figure('Zavisnost type od ulaza ' + dataToRepresent)
    plt.title('Zavisnost type od ulaza ' + dataToRepresent)
    plt.xlabel(dataToRepresent)
    plt.ylabel("type")
    plt.scatter(plotDataIn, plotDataOut, s=23, c='green', marker='o', alpha=0.7,
                edgecolors='black', linewidths=2, label=dataToRepresent)
    plt.legend()
    plt.tight_layout()
    plt.show()

pd.set_option('display.max_columns', 8)
pd.set_option('display.width', None)
data = pd.read_csv('datasets/cakes.csv')
data.insert(loc=0, column="RowId", value=np.arange(0, len(data), 1))
print(data.head())

# 2. DATA ANALYSIS
# data profiling - koliko kojih ima i kog su tipa
print(data.info())

# feature statistic
print(data.describe())
print(data.describe(include=[object]))

dataForPlot = data.__deepcopy__()

le = LabelEncoder()
data.type = le.fit_transform(data.type)

data_train = data[['flour', 'eggs', 'sugar', 'milk', 'butter', 'baking_powder']] #DataFrame type
typeOfCake = data['type'] #Series type

# #calculate new values
data_train = (data_train - pandas.DataFrame.min(data_train)) / (pandas.DataFrame.max(data_train) - pandas.DataFrame.min(data_train)).values

#round float values
data_train["flour"] = data_train["flour"].map(lambda x: round(x, 4))
data_train["eggs"] = data_train["eggs"].map(lambda x: round(x, 4))
data_train["sugar"] = data_train["sugar"].map(lambda x: round(x, 4))
data_train["milk"] = data_train["milk"].map(lambda x: round(x, 4))
data_train["butter"] = data_train["butter"].map(lambda x: round(x, 4))
data_train["baking_powder"] = data_train["baking_powder"].map(lambda x: round(x, 4))

print(data_train.head(20))
# #Print HEATMAP
# correlationMatrixPlot(data_train.corr())

# Split data
X_train, X_test, y_train, y_test = train_test_split(data_train, typeOfCake, train_size=0.80, random_state=110, shuffle=True)

# ISPIS GRAFIKA
# continualDataPlot("flour")
# continualDataPlot("eggs")
# continualDataPlot("sugar")
# continualDataPlot("milk")
# continualDataPlot("butter")
# continualDataPlot("baking_powder")

print("print(X_test)")
print(X_test)
print("print(y_test)")
print(y_test)


#-----------------------------------------------------------------------------------------------------------------------
#KNN MODEL CREATION
numOfNeighbours = int(np.round(np.sqrt(data.shape[0]))) - 1
knnModel = KNeighborsClassifier(n_neighbors=3)

knnModel.fit(X_train, y_train)

predictions = knnModel.predict(X_test)
ser_pred = pd.Series(data=predictions, name="Predicted", index=X_test.index)
res_df = pd.concat([X_test, y_test, ser_pred], axis=1)
print(res_df.head())

accuracy = knnModel.score(X_test, y_test)

print("MSE: " + str(mean_squared_error(y_test, predictions)))
print("Score: " + str(accuracy))
print("Score with accuracy from sklearn: " + str(accuracy_score(y_test, predictions)))