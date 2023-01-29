import pandas
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def correlationMatrixPlot(dataWithCorr):
    plt.figure('Korelaciona matrica')
    plt.title('Korelaciona matrica')
    plt.tight_layout()
    sb.heatmap(dataWithCorr, annot=True, square=True, fmt=".2f")
    plt.show()

def continualDataPlot(dataToRepresent):
    plotDataIn = data[[dataToRepresent]]
    plotDataOut = data['CO2EMISSIONS']

    plt.figure('Zavisnost CO2EMISSIONS od ulaza ' + dataToRepresent)
    plt.title('Zavisnost CO2EMISSIONS od ulaza ' + dataToRepresent)
    plt.xlabel(dataToRepresent)
    plt.ylabel("CO2EMISSIONS")
    plt.scatter(plotDataIn, plotDataOut, s=23, c='green', marker='o', alpha=0.7,
                edgecolors='black', linewidths=2, label=dataToRepresent)
    plt.legend()
    plt.tight_layout()
    plt.show()

def categoricalDataPlot(dataToRepresent):
    plotData = data.groupby(dataToRepresent)["CO2EMISSIONS"].mean()
    print(plotData)
    plt.figure('Zavisnost CO2EMISSIONS od ulaza ' + dataToRepresent)
    plt.title('Zavisnost CO2EMISSIONS od ulaza ' + dataToRepresent)
    plt.xlabel(dataToRepresent)
    plt.ylabel("CO2EMISSIONS")
    plotData.plot.bar()
    plt.tight_layout()
    plt.show()

pd.set_option('display.max_columns', 14)
pd.set_option('display.width', None)
data = pd.read_csv('datasets/fuel_consumption.csv')
data.insert(loc=0, column="RowId", value=np.arange(0, len(data), 1))
print(data.head())

# 2. DATA ANALYSIS
# data profiling - koliko kojih ima i kog su tipa
print(data.info())

# WHAT HAPPENED with RowId?
# data["RowId"] = data["RowId"].map(lambda x: int(x))
# nothing. ignore

# feature statistic
print(data.describe())
print(data.describe(include=[object]))

# # 3. Data cleansing
# print(data.loc[data['ENGINESIZE'].isnull()])
# print(data.loc[data['FUELTYPE'].isnull()])

#fill NaN with mean value
print(type(data['ENGINESIZE'].mean()))
data['ENGINESIZE'] = data['ENGINESIZE'].fillna(np.around(data['ENGINESIZE'].mean(), decimals=1))
data['FUELTYPE'] = data['FUELTYPE'].fillna(data['FUELTYPE'].mode()[0])

# Check if data is null
# print(data.loc[data['ENGINESIZE'].isnull()])
# print(data.loc[data['FUELTYPE'].isnull()])

#4. FEATURE ENGINEERING
# useful features?: ENGINESIZE, CYLINDERS, FUELTYPE, FUELCONSUMPTION_COMB
# Result: CO2EMISSIONS



data_train = data[['ENGINESIZE', 'CYLINDERS', 'FUELTYPE', 'FUELCONSUMPTION_COMB']] #DataFrame type
emissions = data[['CO2EMISSIONS']] #Series type
emissions = (emissions - pandas.DataFrame.min(emissions)) / (pandas.DataFrame.max(emissions) - pandas.DataFrame.min(emissions)).values

emissions = emissions['CO2EMISSIONS']

#OH promena

ohe = OneHotEncoder(dtype=int, sparse_output=False)
fueltype = ohe.fit_transform(data_train['FUELTYPE'].to_numpy().reshape(-1, 1))


data_train = data_train.drop(columns=["FUELTYPE"], inplace=False)
data_train = data_train.join(pd.DataFrame(data=fueltype, columns=ohe.get_feature_names_out(["FUELTYPE"])))
# print(data_train.tail(30))

#calculate new values
data_train = (data_train - pandas.DataFrame.min(data_train)) / (pandas.DataFrame.max(data_train) - pandas.DataFrame.min(data_train)).values

#round float values
data_train["ENGINESIZE"] = data_train["ENGINESIZE"].map(lambda x: round(x, 4))
data_train["CYLINDERS"] = data_train["CYLINDERS"].map(lambda x: round(x, 4))
data_train["FUELCONSUMPTION_COMB"] = data_train["FUELCONSUMPTION_COMB"].map(lambda x: round(x, 4))

#float to int for FUELTYPE
data_train["FUELTYPE_D"] = data_train["FUELTYPE_D"].map(lambda x: int(x))
data_train["FUELTYPE_E"] = data_train["FUELTYPE_E"].map(lambda x: int(x))
data_train["FUELTYPE_X"] = data_train["FUELTYPE_X"].map(lambda x: int(x))
data_train["FUELTYPE_Z"] = data_train["FUELTYPE_Z"].map(lambda x: int(x))

print(data_train.head(10))

#Print HEATMAP
correlationMatrixPlot(data_train.corr())

# Split data
X_train, X_test, y_train, y_test = train_test_split(data_train, emissions, train_size=0.75, random_state=112, shuffle=False)

linRegModel = LinearRegression()
linRegModel.fit(X_train, y_train)

# ISPIS GRAFIKA
# categoricalDataPlot("MAKE")
# categoricalDataPlot("VEHICLECLASS")
# continualDataPlot("ENGINESIZE")
# continualDataPlot("CYLINDERS")
# categoricalDataPlot("TRANSMISSION")
# categoricalDataPlot("FUELTYPE")
# continualDataPlot("FUELCONSUMPTION_CITY")
# continualDataPlot("FUELCONSUMPTION_HWY")
# continualDataPlot("FUELCONSUMPTION_COMB")

emissionsPredicted = linRegModel.predict(X_test)
ser_pred = pd.Series(data=emissionsPredicted, name="Predicted", index=X_test.index)
res_df = pd.concat([X_test, y_test, ser_pred], axis=1)
print(res_df.head())

np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})

print(f"Coefficients: {linRegModel.coef_}")
print("Mean Squared Error: " + str(mean_squared_error(y_test, ser_pred)))
print("Model score: " + str(linRegModel.score(X_test, y_test)))


# print("Coefficients: " + str(linRegModel.coef_))