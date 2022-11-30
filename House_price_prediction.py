#Importing the libraries and modules, functions and programs

# Matplotlib is a comprehensive library for creating static, animated, 
#and interactive visualizations in Python. 
#Matplotlib makes easy things easy and hard things possible. 
#mpl is just simply the abbreviation word to avoid long dictation in coding
import matplotlib as mpl

#Pyplot function im Matplotlib makes some change to a figure: e.g., creates a figure, 
#creates a plotting area in a figure, plots some lines in a plotting area, 
#decorates the plot with labels, etc.
import matplotlib.pyplot as plt

#Seaborn is a library for making statistical graphics in Python. 
#It builds on top of matplotlib and integrates closely with pandas data structures
#This library will be used in plotting the bivariate analysis of the data later on #sns.pairplot(real_estate_data_col)
import seaborn as sns

#NumPy (Numerical Python) is an open source Python library that’s used in 
#almost every field of science and engineering. In short, code that has to do with number.
import numpy as np

#Simple and efficient tools for predictive data analysis
#Accessible to everybody, and reusable in various contexts
#Built on NumPy, SciPy, and matplotlib
import sklearn as sk

#The import train_test_split function splits arrays or matrices into random train and test subsets.
from sklearn.model_selection import train_test_split

#Importing the class LinearRegression 
#Ordinary least squares Linear Regression
#LinearRegression fits a linear model with coefficients w = (w1, …, wp) 
#to minimize the residual sum of squares between the observed targets in the dataset
#and the targets predicted by the linear approximation.
from sklearn.linear_model import LinearRegression

#Defining the mean absolute error regression loss
from sklearn.metrics import mean_absolute_error

#Numericalunits package from pypi.org lets you define quantities with units, which can then be used in almost 
#any numerical calculation in any programming language. 
#Checks that calculations pass dimensional analysis, performs unit conversions, 
#and defines physical constants.
import numericalunits

#Sepcifying the unit required for the model 'm' as in meters
from numericalunits import m


# Importing pandas
import pandas as pd

# read csv file using pandas
df=pd.read_csv('Real_estate.csv')
data=df.to_dict('records')
print(data)

    
#Importing pymongo module so as to be able to use our dataset from MongoDB database and collection
import pymongo

#Importing MongoClient to create a client folder on MongoDB
from pymongo import MongoClient 

#Creating client variable for the dataset access from MongoDB cloud or a local host
client = MongoClient("mongodb+srv://username:password@cluster0.qcurgqp.mongodb.net/?retryWrites=true&w=majority")

#Local host code lines might look like this
#client = MongoClient("mongodb://localhost:27017/")

#Creating the database called Real_estate on MongoDB
db = client['Real_estate']

#Creating the collection called Real_estate_collection on MongoDB
collection = db["Real_estate_collection"]

#To avoid duplicate data in the collection. We will drop the rows and columns that has duplicate 
#from the csv file before it is inserted into the MongoDB collection path
collection.drop()

#Calling the function 'insert' to populate the data unto MongoDB collection
collection.insert_many(data)

#Retrieving the data from MongoDB by giving a variable to the dataframe for find function
real_estate_data_col = pd.DataFrame((collection.find({},{"_id":0})))

#Removing the unwanted number column 'No'
real_estate_data_col = real_estate_data_col.drop('No', axis=1)

#Redefining the dataframe variable after removing unwanted column
df = pd.DataFrame(real_estate_data_col)
print(df)

#Defining the features
features = ['house_age', 'dist_to_station', 'no_of_store']

#Dropping the features that are not required for our target prediction
X = df.drop(['latitude','longitude','house_price_of_unit_area','house_price_in_pound'],axis=1)

#Selecting our target feature 
y = real_estate_data_col["house_price_in_pound"]


#Checking the number of rows and columns
real_estate_data_col.shape
print(real_estate_data_col.shape)

#Checking the description of data
real_estate_data_col.describe()
print(real_estate_data_col.describe())

#Checking the types of data
real_estate_data_col.dtypes
print(real_estate_data_col.dtypes)

#Checking the basic information on dataset
real_estate_data_col.info()
print(real_estate_data_col.info())

#Checking the column titles at a glance
real_estate_data_col.columns
print(real_estate_data_col.columns)

#Checking if there are no empty row or column in the dataset
real_estate_data_col.isnull() 
print(real_estate_data_col.isnull())

#Checking for duplicates
real_estate_data_col.duplicated().sum()
print(real_estate_data_col.duplicated())

#Printing only the first 3 rows of the dataset to vividly observe the cleaned data
real_estate_data_col[:3]
print(real_estate_data_col[:3])

#Printing the first 3 rows of the data with focus on the columns we will be using
real_estate_data_col[:3][["house_age", "dist_to_station", "no_of_store"]]
print(real_estate_data_col[:3][["house_age", "dist_to_station", "no_of_store"]] )

#Detail classification for training and testing the data, 
#defining the model to use and verifying it
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=1)
real_estate_model = LinearRegression()
print(real_estate_model) 

#Model training and prediction
real_estate_model.fit(X_train, y_train)
real_estate_model.score(X_test, y_test)
print(real_estate_model.fit(X_train, y_train))
print(real_estate_model.score(X_test, y_test))


#To check the best model giving the best result
#We use ShuffleSplit class import for random permutation cross-validator
from sklearn.model_selection import ShuffleSplit
#While we evaluate the score by importing cross_val_score function
from sklearn.model_selection import cross_val_score

#Defining the Class as a variable 'cv' for subsequent use
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv)
print(cross_val_score(LinearRegression(), X, y, cv=cv))

                                     
                                  
#Printing the feature train content
X_train
print(X_train)

#Printing the target train content
y_train
print(y_train)

#Checking the accuracy value for the house predictions
house_price_prediction = real_estate_model.predict([[32, 84, 10]])
print("House price prediction: ", house_price_prediction)

print("Coefficients") 
print(real_estate_model.coef_)
print("Intercept") 
print("%2f"%real_estate_model.intercept_)


#Checking for multicolinearity
corr = real_estate_data_col.corr()
print(corr)

# Getting the Variance Inflation Factor (VIF)
pd.DataFrame(np.linalg.inv(corr.values), index = corr.index, columns=corr.columns)
print(pd.DataFrame(np.linalg.inv(corr.values), index = corr.index, columns=corr.columns))

#Analysing the dataset in a data visualisation mode
#Importing the sweetviz module
import sweetviz as sv

#Defining the analysis file
house_EDA = sv.analyze(df)

#Defining the file format and the view function
house_EDA.show_html("./house_EDA.html")

house_EDA.show_notebook(None, None, None, 'widescreen', './house_EDA')


#Checking for errors and content analysis
columns_dict = {'house_age':1,'dist_to_station':2,
                'no_of_store':3, 'house_price_in_pound':7}


#Doing univariate analysis
plt.figure(figsize=(25,30))

for key, value in columns_dict.items(): 
  plt.subplot(5,4,value)
  sns.displot(real_estate_data_col[key])
  plt.title(key)
  
  plt.show()
  
#Check for and deal with outliers
plt.figure(figsize=(25,30))

for key, value in columns_dict.items(): 
  plt.subplot(5,4,value)
  plt.boxplot(real_estate_data_col[key])
  plt.title(key)
  
  plt.show()
  
#Ploting the bivariate analysis of the data.
sns.pairplot(real_estate_data_col)

plt.show()

#Defining the House price class
class House_price:
    def __init__(self):
        self.X_test = X_test
    def house_price_prediction(self):
        house_price_prediction = real_estate_model.predict(self.X_test)
        print(house_price_prediction)

def main():
    house1 = House_price()
    house1.house_price_prediction()
   
#Calling the Main
main()

import tkinter

class House_price_GUI:
    def __init__(self):
        
        #Create main window
        self.mw = tkinter.Tk()
        #Set the main title of GUI
        self.mw.title("House_Price_Calculator")
        
        #Set the background colour
        self.mw.configure(bg='magenta')
        
        #Defining the String variables
        self.lab_var = tkinter.StringVar()
        
        #Defining the String for the User Interface
        self.lab_Label = tkinter.Label(self.mw, textvariable = self.lab_var)        
        
        #Defining the String  for the User Interface
        #self.lab_Frame = tkinter.Frame(self.mw, bg="magenta") 
        
        #Setting the label variable with the display and input reset to 0
        self.lab_var.set("House price:           £0")
                
        #Creating the buttons: House age, distance from store, number of stores and Reset
        self.house_age_label = tkinter.Label(self.mw, text = "House age")
        self.dist_to_station_label = tkinter.Label(self.mw, text = "Dist from station in m")
        self.no_of_store_label = tkinter.Label(self.mw, text = "No of stores")
        self.entryhouse_ageLabel = tkinter.Entry(self.mw, width = 15)
        self.entrydist_to_stationLabel = tkinter.Entry(self.mw, width = 15)
        self.entryno_of_storeLabel = tkinter.Entry(self.mw, width = 15)
        self.calculator_house_price = tkinter.Button(self.mw, text = "Calculate house price", command = self.calc_house_price)
        self.Reset = tkinter.Button(self.mw, text = "Reset", command = self.calc_reset)
        
        #Equating the calculation to 0 for reset
        self.calc_total = 0
       
        #Packing all the features and target
        self.lab_Label.pack(side = "top")
        self.house_age_label.pack(side = "top")
        self.entryhouse_ageLabel.pack(side = "top")
        self.dist_to_station_label.pack(side = "top")
        self.entrydist_to_stationLabel.pack(side = "top")
        self.no_of_store_label.pack(side = "top")
        self.entryno_of_storeLabel.pack(side = "top")
        self.calculator_house_price.pack(side = "top")
        self.Reset.pack(side = "top")
        
               
        #Creating a standard tkinter function for operation
        
        tkinter.mainloop()
    
        
    def calc_house_price(self):
        self.house_age=self.entryhouse_ageLabel.get()
        self.dist_stn=self.entrydist_to_stationLabel.get()
        self.no_of_stores=self.entryno_of_storeLabel.get()
        house_price = real_estate_model.predict([[self.house_age,self.dist_stn,self.no_of_stores]])
        self.lab_var.set("House price:           £%d" %house_price)
    
    def calc_reset(self):
        reset = 0
        self.calc_total = 0
        self.lab_var.set("Total:           £%d" %reset)
    
#Calling the House price class
House_price_GUI()