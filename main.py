### IMPORTS ###

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns


import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point
import psycopg2

### CONSTANTS ###

ROOT = Path('.')
HOST = 'dtim.essi.upc.edu'
PORT = 27017
DBNAME = "DBkleber_reyes"


##############################
##      DATA CLEANING       ##
##############################

print("##############################")
print("##      DATA CLEANING       ##")
print("##############################")
print()

'''
In this section I will calculate and report the % of missing values for temperature and chlorophyll dataSets.
First of all, I will read the source datasets as pandas dataframes since for now, I do not need the geometries.
'''

DataDir = ROOT / "Data"

fileNames = ["ais.csv", "fishing.csv", "temperature_012020-062020_res05.csv", "chlorophylla_012020-062020_res05.csv"]

aisDF = pd.read_csv(DataDir / fileNames[0])           # Shape: (81907, 7) 
fishingDF = pd.read_csv(DataDir / fileNames[1])       # Shape: (644, 10)
temperatureDF = pd.read_csv(DataDir / fileNames[2])   # Shape: (436800, 5)
chlorophyllDF = pd.read_csv(DataDir / fileNames[3])   # Shape: (218400, 4)

print(aisDF.head())
print(fishingDF.head())
print(temperatureDF.head())
print(chlorophyllDF.head())

## Missing Values ##
print("\n## MISSING VALUES##")

print("\n# Missing values of temperature dataset:")
temperatureNA = temperatureDF.isnull().sum()
print(temperatureNA)

print("\n# Missing values of chlorophyll dataset:")
chlorophyllNA = chlorophyllDF.isnull().sum()

'''
The previous functions give me the quantity of missing values in the 2 datasets.
    * "temperature" column has 208824 NaNs
    * "chlor_a" column has 118184 NaNs

Now, in the next piece of code I will generate the total number of NaNs of the dataset.
In this case, as I only have 1 column with missing values per dataset, the value is the same.
'''

print("\n# Percentage of missing values:")

# Total number of missing values
totalTemperatureNA = temperatureNA.sum() 


rowsTemp = temperatureDF.shape[0]
totalTemp = rowsTemp*temperatureDF.shape[1] # Total number of elements



temp_percentage = round((temperatureNA["temperature"] / rowsTemp) *100,  2)
print("Percentage of missing values on temperature column: {}%".format(temp_percentage))

tempDS_percentage = round((totalTemperatureNA / totalTemp) *100,  2)
print("Percentage of missing values on temperature dataset: {}%".format(tempDS_percentage))




###############################################
##      DATA NORMALIZATION/UNIFICATION       ##
###############################################
print("\n###############################################")
print("##      DATA NORMALIZATION/UNIFICATION       ##")
print("###############################################")

#################################
##      DATA INTEGRATION       ##
#################################
print("\n#################################")
print("##      DATA INTEGRATION       ##")
print("#################################")

#########################
##      QUERYING       ##
#########################
print("\n#########################")
print("##      QUERYING       ##")
print("#########################")





















#pd.get_option('display.max_columns')
#db = mongoDB(HOST, PORT, DBNAME)
    #client = MongoClient()

#client = MongoClient('localhost', 27017)
#client = MongoClient('dtim.essi.upc.edu', 27017)


#db = client.DBkleber_reyes

#countries = db.country

#pprint.pprint(countries.find_one())