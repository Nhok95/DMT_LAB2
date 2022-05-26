### IMPORTS ###

from datetime import datetime
from turtle import color
import pandas as pd
import numpy as np

# Paths
from pathlib import Path
import os

# Plot
import matplotlib.pyplot as plt
from matplotlib import colors as pltc

# Geometry libraries
import geopandas as gpd
from shapely.geometry import LineString, Point

### CONSTANTS ###
TEST = True # If True, some pieces of code will no be executed, just for testing. Default will be False

ROOT = Path.cwd()
#HOST = 'dtim.essi.upc.edu'
#PORT = 27017
#DBNAME = "DBkleber_reyes"

DataDir = ROOT / "Data"

# We generate here the folders
if not os.path.exists(DataDir / "Cleaned"):
    os.mkdir(DataDir / "Cleaned")

if not os.path.exists(DataDir / "Shapefiles"):
    os.mkdir(DataDir / "Shapefiles")


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

fileNames = ["ais.csv", "fishing.csv", "temperature_012020-062020_res05.csv", "chlorophylla_012020-062020_res05.csv"]

aisDF = pd.read_csv(DataDir / fileNames[0])           # Shape: (81907, 7) 
fishingDF = pd.read_csv(DataDir / fileNames[1])       # Shape: (644, 10)
temperatureDF = pd.read_csv(DataDir / fileNames[2])   # Shape: (436800, 5)
chlorophyllDF = pd.read_csv(DataDir / fileNames[3])   # Shape: (218400, 4)

if (not TEST):
    print(aisDF.head())
    print()
    print(fishingDF.head())
    print()
    print(temperatureDF.head())
    print()
    print(chlorophyllDF.head())

###################################
### Missing Values ###
print("\n### MISSING VALUES ###")

if (not TEST):
    print("\n# Missing values of temperature dataset:")
    temperatureNA = temperatureDF.isnull().sum()
    print(temperatureNA)

    print("\n# Missing values of chlorophyll dataset:")
    chlorophyllNA = chlorophyllDF.isnull().sum()
    print(chlorophyllNA)

'''
The previous functions give me the quantity of missing values in the 2 datasets.
    * "temperature" column has 208824 NaNs
    * "chlor_a" column has 118184 NaNs

Now, in the next piece of code I will generate the total number of NaNs of the dataset.
In this case, as I only have 1 column with missing values per dataset, the value is the same.
'''
if (not TEST):
    print("\n# Percentage of missing values:")

    # Total number of missing values
    totalTemperatureNA = temperatureNA.sum()
    totalchlorophyllNA = chlorophyllNA.sum() 

    # Total number of rows and elements
    rowsTemp = temperatureDF.shape[0]
    totalTemp = rowsTemp*temperatureDF.shape[1] 

    rowsChlor = chlorophyllDF.shape[0]
    totalChlor = rowsChlor*chlorophyllDF.shape[1]

    # Temperature
    temp_percentage = round((temperatureNA["temperature"] / rowsTemp) *100,  2)
    print("\nPercentage of missing values on temperature column: {}%".format(temp_percentage))

    tempDS_percentage = round((totalTemperatureNA / totalTemp) *100,  2)
    print("Percentage of missing values on temperature dataset: {}%".format(tempDS_percentage))

    # Chlorophyll
    chlor_percentage = round((chlorophyllNA["chlor_a"] / rowsChlor) *100,  2)
    print("\nPercentage of missing values on chlor_a column: {}%".format(chlor_percentage))

    chlorDS_percentage = round((totalchlorophyllNA / totalChlor) *100,  2)
    print("Percentage of missing values on chlorophyll dataset: {}%".format(chlorDS_percentage))

    # Total
    totalNA_percentage = round(((totalchlorophyllNA + totalTemperatureNA) / (totalChlor + totalTemp)) *100,  2)
    print("\nTotal percentage of missing values on datasets: {}%".format(totalNA_percentage))


'''
We can conclude that half of our values for temperature and clorophyll are missing (47.81% and 54.11%, respectively).

That represents a 9.56% and a 13.53% of missing values for the temperature dataset and the clorophyll dataset.

We have an overall of 10.69% of missing values.
'''

###################################
### Imputation ###
print("\n### IMPUTATION ###")
print()

'''
In this subsection I will impute the NaNs as the document recommends (manually calculate +/-3 value mean)

I can also try to do it using the library 'fancyimput' in order to use some machine learning algorithms that take into
account the correlation between variables (like KNN or MICE), but I prefer to follow the recommendation.

But if we want to use the previous algorithms we should look for some properties of our data (correlations with the target)
'''

'''
This piece of code can be used for the previous purpouse (temperature dataset example):

    temperatureDF_DropNA = temperatureDF.dropna()

    # Check the distribution of the variables
    temperatureDF.hist() 
    temperatureDF_DropNA.hist()

    # We can notice that we have more NA in latitude values near to -35 and longitude values near -65

    import seaborn as sns

    corr = temperatureDF.corr(method='spearman')
    matrix = np.triu(corr)
    corr
    sns.heatmap(corr, annot=True, cmap="Reds", mask=matrix)

    # The heatmap shows us that the temperature is strongly correlated with latitude and weakly correlated
    # with longitude (both positive). We can use these relations in order to make a imputation taking 
    # into account the correlated variables.

    # For categorical variables we can use the kendalltau test
    from scipy import stats

    stats.kendalltau(temperatureDF_DropNA['temperature'], temperatureDF_DropNA['partOfTheDay'])
        > KendalltauResult(correlation=-0.0016320705867739545, pvalue=0.3398852836331612)

    stats.kendalltau(temperatureDF_DropNA['temperature'], temperatureDF_DropNA['time'])
        > KendalltauResult(correlation=-0.252934211499414, pvalue=0.0)
    
    # We cannot use 'partOfTheDay' since pvalue > 0.05, this means that variables are not correlated,
    #  but 'time' seems to be weakly correlated.

'''

'''
To calculate the mean of the windows of 7 days (i-3, i-2, i-1, i, i+1, i+2, i+3) I want to transform the 
date of the time column into a day of the year.

To do this I aggregate the values of temperature and chlorophyll by day.
'''
if (not TEST):
    temperatureDFAggregation = temperatureDF[['time','temperature','partOfTheDay']]
    tempAgg = temperatureDFAggregation.groupby(['time', 'partOfTheDay']).mean()
    tempAgg = tempAgg.reset_index().sort_values(by=['partOfTheDay', 'time'])
    tempAgg = tempAgg.reset_index(drop=True)

    tempAgg['day'] = [datetime.strptime(x, '%Y-%m-%d').timetuple().tm_yday for x in tempAgg['time']]

    print(tempAgg.head()) # Shape: (364, 4) 

    print()

    chlorophyllDFAggregation = chlorophyllDF[['time','chlor_a']]
    chlorAgg = chlorophyllDFAggregation.groupby(['time']).mean()
    chlorAgg = chlorAgg.reset_index().sort_values(by=['time'])
    chlorAgg

    chlorAgg['day'] = [datetime.strptime(x, '%Y-%m-%d').timetuple().tm_yday for x in chlorAgg['time']]

    print(chlorAgg.head()) # hape: (182, 3) 

'''
It seems that we have all the needed days (with the average values), since we have 182 days in 
chlorophyll dataset and 364 (182*2) in temperature dataset (the double since we have day and night values).

We have to do the same with our original datasets.
'''
if (not TEST):
    tempWithYDay = temperatureDF.copy()
    chlorWithYDay = chlorophyllDF.copy()

    # Generate the year day from the time column
    chlorWithYDay['day'] = [datetime.strptime(x, '%Y-%m-%d').timetuple().tm_yday for x in chlorWithYDay['time']]
    tempWithYDay['day'] = [datetime.strptime(x, '%Y-%m-%d').timetuple().tm_yday for x in tempWithYDay['time']]

'''
Once we have every row day, we can calculate the mean using the "days" window. For temperature I also take into
account the partOfTheDay feature.
'''
if (not TEST):
    print("\nImputing values...")

    for index, _ in tempWithYDay.iterrows():
        if (np.isnan(tempWithYDay['temperature'][index])):
            day = tempWithYDay['day'][index]
            part = tempWithYDay['partOfTheDay'][index]
            days = [day-3, day-2, day-1, day, day+1, day+2, day+3]

            if day < 4:
                days = [x for x in days if x > 0]
            
            aux = tempAgg[(tempAgg['day'].isin(days)) & (tempAgg['partOfTheDay'] == part)]
            tempWithYDay['temperature'] = aux['temperature'].sum() / aux.shape[0]

    tempWithYDay['temperature'] = tempWithYDay['temperature'].round(decimals=6)

    for index, _ in chlorWithYDay.iterrows():
        if (np.isnan(chlorWithYDay['chlor_a'][index])):
            day = chlorWithYDay['day'][index]
            days = [day-3, day-2, day-1, day, day+1, day+2, day+3]

            if day < 4:
                days = [x for x in days if x > 0]
            
            aux = chlorAgg[chlorAgg['day'].isin(days)]
            chlorWithYDay['chlor_a'] = aux['chlor_a'].sum() / aux.shape[0]

    chlorWithYDay['chlor_a'] = chlorWithYDay['chlor_a'].round(decimals=2)


'''
We can check again the number of missing values:
'''
if (not TEST):  
    print("\nNumber of missing values on temperature dataset after imputation: {}".format(tempWithYDay.isnull().sum().sum()))
    print("Number of missing values on chlorophyll dataset after imputation: {}".format(chlorWithYDay.isnull().sum().sum()))

'''
I have checked that I do not have NaNs anymore.
'''
if (not TEST):
    print("\n Now we have datasets without missing values:")
    clean_temperatureDF = tempWithYDay.copy().iloc[:, 0:5]

    print(clean_temperatureDF.head())
    print()

    clean_chlorophyllDF = chlorWithYDay.copy().iloc[:, 0:4]
    print(clean_chlorophyllDF.head())

    '''
    I store the cleaned datasets into the cleaned folder
    '''
    clean_temperatureDF.to_csv(DataDir / "Cleaned/clean_temperatureDF.csv", index=False)
    clean_chlorophyllDF.to_csv(DataDir / "Cleaned/clean_chlorophyllDF.csv", index=False)

###################################
### Sample Frequency ###
print("\n### SAMPLE FREQUENCY ###")
print()

'''
In this last subsection of data cleaning I will reduce the sampling frequency of the AIS data from minutes to hours.
For this purpouse I need that the Date columns becomes a timestamp type.

I create a new dataframe ignoring the Speed and the Course.
'''
if (not TEST):
    aisResample = aisDF.copy()
    aisResample["Date"] = pd.to_datetime(aisResample["Date"])
    aisResample["Date"] = [pd.Timestamp(x) for x in aisResample["Date"]]

    aisResample = aisResample[["BoatName","BoatID", "Date", "Latitude", "Longitude"]]

'''
I round the timestamp to hour level (round down). These new values let me aggregate in the following steps.
'''
if (not TEST):
    aisResample['Date'] = [x.floor(freq='H') for x in aisResample['Date']]

'''
I use groupby in order to aggregate the latitude and longitude using the mean.

The new dataframe has 17669 rows, compared with the original one (81907 rows), 
I have more or less a 20% of the original size.
'''
if (not TEST):
    ais_res = aisResample.groupby(['BoatName', 'BoatID', 'Date']).agg({'Latitude': 'mean', 'Longitude': 'mean'})
    ais_res = ais_res.reset_index().sort_values(by=['BoatID', 'Date'])

    print("New AIS dataset: ")
    print(ais_res.head()) # Shape: (17669, 5) 

'''
Lastly, I can store the new dataset in order to reuse it in the future.
'''
if (not TEST):
    ais_res.to_csv(DataDir / "Cleaned/ais_resample.csv", index=False)


###############################################
##      DATA NORMALIZATION/UNIFICATION       ##
###############################################
print("\n###############################################")
print("##      DATA NORMALIZATION/UNIFICATION       ##")
print("###############################################")
print()

'''
I will use the following DF generated in the previous section:
    * clean_temperatureDF
    * clean_chlorophyllDF
    * ais_res
    * fishingDF

'''
ais_res = pd.read_csv(DataDir / "Cleaned/ais_resample.csv") 
clean_temperatureDF = pd.read_csv(DataDir / "Cleaned/clean_temperatureDF.csv") 
clean_chlorophyllDF= pd.read_csv(DataDir / "Cleaned/clean_chlorophyllDF.csv") 

print(clean_temperatureDF.head())
print()
print(clean_chlorophyllDF.head())
print()
print(ais_res.head())
print()
print(fishingDF.head())

###################################
### Normalize to the same grid ###
print("\n### GRID NORMALIZATION ###")
print()

'''
We want to normalize the latitude and longitude coordinates of every dataset to a grid of square. Each square will be 
identified by a positive pair of index (i,j).

    * For us the lat: -90 and the lon: -180 represents the origin (0,0).
    * I will map my original coordinates (in lat/lon) to this new grid in the follwing way:
        - the indexs i and j represents the lower left corner:
               +---+
               |   |
         (i,j) +---+
        - the square represents a size of 0.5^2, this means that from i to i+1 I add 0.5 to the previous coordinate
        - each value of the original latitude/longitude is mapped to one of these squares

The formula I use is the following one:
    new latitude  -> abs(-90  - latitude ) // 0.5
    new longitude -> abs(-180 - longitude) // 0.5

If i take the latitude -90 and the longitude -180 the values of the new pair of coordinates I generate is the next:
    abs(-90  - ( -90)) // 0.5 = 0
    abs(-180 - (-180)) // 0.5 = 0

The map is the expected one

If I want to map the latitude -89.7 and longitude -179.7 it should be mapped to (0,0) square
    abs(-90  - ( -89.7)) // 0.5 = 0.3 // 0.5 = 0
    abs(-180 - (-179.7)) // 0.5 = 0.3 // 0.5 = 0

This previous example also returns an expected result.

What happens with a difference higher than 0.5 (but below 1.0)
    abs(-90  - ( -89.3)) // 0.5 = 0.7 // 0.5 = 1
    abs(-180 - (-179.3)) // 0.5 = 0.7 // 0.5 = 1

Since it seems the mapping works we can continue with the normalization piece of code.
'''
clean_temperatureDF['normLat'] = abs(-90 - clean_temperatureDF.latitude) // 0.5
clean_temperatureDF['normLon'] = abs(-180 - clean_temperatureDF.longitude) // 0.5

clean_chlorophyllDF['normLat'] = abs(-90 - clean_chlorophyllDF.latitude) // 0.5
clean_chlorophyllDF['normLon'] = abs(-180 - clean_chlorophyllDF.longitude) // 0.5

ais_res['normLat'] = abs(-90 - ais_res.Latitude) // 0.5
ais_res['normLon'] = abs(-180 - ais_res.Longitude) // 0.5

fishingDF['normLat'] = abs(-90 - fishingDF.Latitude) // 0.5
fishingDF['normLon'] = abs(-180 - fishingDF.Longitude) // 0.5


print("\nThe datasets now contains a new pair or columns that represents the normalized coordinates:")
print(clean_temperatureDF.head())
print()
print(clean_chlorophyllDF.head())
print()
print(ais_res.head())
print()
print(fishingDF.head())

'''
We can create a new set of file in order to store these normalized datasets
'''


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
print()

'''
In this last section I will perform some queries to the new data I generated.

First I will compare trajectories from fishing and ais datasets (previous to any preprocess). I want create 
linestrings from the points defined by latitude and longitude
'''

###################################
### Comparing trajectories ###
print("\n### COMPARING TRAJECTORIES ###")
print()

aisDF = aisDF.sort_values(by=["BoatID", "Date"]).reset_index(drop=True)

# Create a geoDataframe with points
geoAisDF = gpd.GeoDataFrame(aisDF, 
                        geometry=gpd.points_from_xy(aisDF.Latitude, aisDF.Longitude),
                        crs = "EPSG:3857")

geoFishingDF = gpd.GeoDataFrame(fishingDF, 
                                geometry=gpd.points_from_xy(fishingDF.Latitude, fishingDF.Longitude),
                                crs = "EPSG:3857")

print("Original ais and fishing datasets as GeoPandas DataFrames:")
print(geoAisDF.head())
print()
print(geoFishingDF.head())
print()

'''
I tried to use dissolve() in order to aggregate the geometries, but the result geometry is an unordered set of point 
(A MultiPoint geometry)

For this reason I will compare only the Mason trajectory since the other comparations will be equivalent.
'''

boatId = 111 # Mason Id
masonLineAis = []

geoAisBoat = geoAisDF[geoAisDF["BoatID"] == boatId] # Sub geo dataframe
for i, _ in geoAisBoat.iterrows():
    masonLineAis.append(geoAisBoat['geometry'][i])
masonLinestringAis = LineString([x for x in masonLineAis])

masonLineFishing = []

geoFishingBoat = geoFishingDF[geoFishingDF["BoatID"] == boatId] # Sub geo dataframe
for i, _ in geoFishingBoat.iterrows():
    masonLineFishing.append(geoFishingBoat['geometry'][i])
masonLinestringFishing = LineString([x for x in masonLineFishing])



if (not TEST):
    #plt.figure(figsize=(18,16), dpi=80)
    print("\nPlotting the Points...")

    xs = [point.x for point in masonLineAis]
    ys = [point.y for point in masonLineAis]

    plt.scatter(xs, ys)

    xs = [point.x for point in masonLineFishing]
    ys = [point.y for point in masonLineFishing]

    plt.scatter(xs, ys)

    plt.show()

    #plt.figure(figsize=(18,16), dpi=80)
    print("\nPlotting the Trajectories...")

    x, y = masonLinestringAis.coords.xy
    plt.plot(x, y)

    x, y = masonLinestringFishing.coords.xy
    plt.plot(x, y)

    plt.show()

'''
We can realize that the line from the fishing dataset (1 coordinate per day) not represents the complete path
that the boat does throughout the day. In fact, the fishing dataset has gaps of several days and we don't 
know what course each ship had during this space of time. If we look the ais dataset we can check what 
was the true trajectory it took.

I has to highlight that the ais dataset also has some gaps, and for this reason the scatter plot give us a
different "trajectory" than the generated with the Linestring (remember that it puts straight lines between points).
'''

print("Mason trajectory length according to Fishing dataset: {}".format(masonLinestringFishing.length))
print("Mason trajectory lengt haccording to AIS dataset: {}".format(masonLinestringAis.length))

print("\nMason number of coordinates according to Fishing dataset: {}".format(len(masonLinestringFishing.coords)))
print("Mason number of coordinates according AIS dataset: {}".format(len(masonLinestringAis.coords)))

print("\nMason representative point according to Fishing dataset: {}".format(masonLinestringFishing.representative_point()))
print("Mason representative point according to AIS dataset: {}".format(masonLinestringAis.representative_point()))
print("Mason difference of representative points: {}".format(
        (masonLinestringFishing.representative_point()).distance(masonLinestringAis.representative_point())))

print("\nAre trajectories equals? {}".format(masonLinestringFishing.equals(masonLinestringAis)))

'''
We can realize that the number of coordinates and the length of every trajectory is shorter in the fishing dataset, 
this is related with the previous conclusions. Since it exists several time gap between the sampling the transformation
I made to generate the trajectory assumes that the boat goes from one point to another following a straight line, 
when we already know that this not occurs in reality.

Thanks to the ais dataset we can know the real path every boat did (that is not considered equal to the gived by the
fishing dataset). 
'''












#pd.get_option('display.max_columns')
#db = mongoDB(HOST, PORT, DBNAME)
    #client = MongoClient()

#client = MongoClient('localhost', 27017)
#client = MongoClient('dtim.essi.upc.edu', 27017)


#db = client.DBkleber_reyes

#countries = db.country

#pprint.pprint(countries.find_one())