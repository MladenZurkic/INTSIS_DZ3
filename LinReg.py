import pandas
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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

# 3. Data cleansing
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

# ----------------------------------------------------------------------------------------------------------------------
#Add intercept

data_train['intercept'] = 1

X = data_train[['intercept', "ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "FUELTYPE_D", "FUELTYPE_E", "FUELTYPE_X", "FUELTYPE_Z"]]
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, emissions, train_size=0.75, random_state=112, shuffle=False)

# print(X_train.shape[1])
coefficients = np.random.rand(X_train.shape[1])

mseList = []
learningRate = 0.1
numOfIterations = 10000

for i in range(numOfIterations):
    # Calculate the prediction
    prediction = X_train.dot(coefficients)
    # Calculate the error
    error = prediction - y_train
    #print(prediction)
    # Calculate the gradient
    gradient = X_train.T.dot(error) / X_train.shape[0]
    # Update the coefficients
    coefficients = coefficients - learningRate * gradient
    mse = np.mean(error ** 2)
    mseList.append(mse)


X_test["intercept"] = 1
prediction = X_test.dot(coefficients)

print("Coefficients: "+ str(coefficients))
print("Final MSE: ", mseList[-1])
print("Score: " + str(r2_score(y_test, prediction)))


# plt.figure('MS Error')
# plt.plot(np.arange(0, len(mseList[:100]), 1), mseList[:100])
# plt.xlabel('Iteration', fontsize=13)
# plt.ylabel('MS error value', fontsize=13)
# plt.xticks(np.arange(0, len(mseList[:100]), 2))
# plt.title('Mean-square error function')
# plt.tight_layout()
# plt.legend(['MS Error'])
# plt.show()


#-----------------------------------------------------------------------------------------------------------------------
# KOD SA PREZENTACIJE
# class LinearRegressionGradientDescent:
#     def __init__(self):
#         self.coeff = None
#         self.features = None
#         self.target = None
#         self.mse_history = None
#
#     def set_coefficients(self, *args):
#         self.coeff = np.array(args).reshape(-1, 1)
#
#     def cost(self):
#         predicted = self.features.dot(self.coeff)
#         s = pow(predicted - self.target, 2).sum()
#         return (0.5 / len(self.features)) * s
#
#     # Argument mora biti DataFrame
#     def predict(self, features):
#         features = features.copy(deep=True)
#         features.insert(0, 'c0', np.ones((len(features), 1)))
#         features = features.to_numpy()
#         return features.dot(self.coeff).reshape(-1, 1).flatten()
#
#     def gradient_descent_step(self, learning_rate):
#         # learning_rate - korak ucenja; dimenzije ((n + 1) x 1);
#         # korak ucenja je razlicit za razlicite koeficijente
#         # m - broj uzoraka
#         # n - broj razlicitih atributa (osobina)
#         # features – dimenzije (m x (n + 1));
#         # n + 1 je zbog koeficijenta c0
#         # self.coeff – dimenzije ((n + 1) x 1)
#         # predicted – dimenzije (m x (n + 1)) x ((n + 1) x 1) = (m x 1)
#         predicted = self.features.dot(self.coeff)
#         # koeficijeni se azuriraju po formuli:
#         # coeff(i) = coeff(i) - learning_rate * gradient(i)
#         # za i-ti koeficijent koji mnozi i-ti atribut
#         # gledaju se samo vrednosti i-tog atributa za sve uzorke
#         # gradient(i) = (1 / m) * sum(y_predicted - y_target) * features(i)
#
#         # (predicted - self.target) - dimenzije (m x 1)
#         # features - dimenzije (m x (n + 1));
#         # transponovana matrica ima dimenzije ((n + 1) x m)
#         # gradient - dimenzije ((n + 1) x m) x (m x 1) = (n + 1) x 1
#         s = self.features.T.dot(predicted - self.target)
#         gradient = (1. / len(self.features)) * s
#         self.coeff = self.coeff - learning_rate * gradient
#         return self.coeff, self.cost()
#
#     def perform_gradient_descent(self, learning_rate, num_iterations=1000):
#             # Istorija Mean-square error-a kroz iteracije gradijentnog spusta.
#             self.mse_history = []
#             for i in range(num_iterations):
#                 _, curr_cost = self.gradient_descent_step(learning_rate)
#                 self.mse_history.append(curr_cost)
#             return self.coeff, self.mse_history
#
#     # features mora biti DataFrame
#     def fit(self, features, target):
#         self.features = features.copy(deep=True)
#
#         # Pocetna vrednost za koeficijente je 0.
#         # self.coeff - dimenzije ((n + 1) x 1)
#         coeff_shape = len(features.columns) + 1
#         self.coeff = np.zeros(shape=coeff_shape).reshape(-1, 1)
#         # Unosi se kolona jedinica za koeficijent c0,
#         # kao da je vrednost atributa uz c0 jednaka 1.
#         self.features.insert(0, 'c0', np.ones((len(features), 1)))
#         # self.features - dimenzije (m x (n + 1))
#         self.features = self.features.to_numpy()
#         # self.target - dimenzije (m x 1)
#         self.target = target.to_numpy().reshape(-1, 1)
#
# X_train, X_test, y_train, y_test = train_test_split(data_train, emissions, train_size=0.75, random_state=112, shuffle=False)
#
# spots = 200
# estates = pd.DataFrame(data=np.linspace(0, max(X_train['CYLINDERS']), num=spots))
# # Kreiranje i obucavanje modela
# lrgd = LinearRegressionGradientDescent()
# lrgd.fit(X_train, y_train)
# learning_rates = np.array([[0.17], [0.0000475], [0.0000475], [0.0000475], [0.0000475], [0.0000475], [0.0000475], [0.0000475]])
# res_coeff, mse_history = lrgd.perform_gradient_descent(learning_rates, 20)
#
# print(1 - mse_history[-1])
# # --------------------------------------------------------------------------------------------------------------------