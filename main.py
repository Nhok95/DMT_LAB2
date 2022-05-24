import pprint
import pandas as pd
from pathlib import Path

from mongoDB import *

ROOT = Path('.')
HOST = 'dtim.essi.upc.edu'
PORT = 27017
DBNAME = "DBkleber_reyes"


if __name__ == "__main__":

    InputDataDir = ROOT / "InputData"

    fileNames = ["fishing.csv", "ais.csv", "temperature.csv", "chlorophyll.csv"]

    aisDF = pd.read_csv(InputDataDir / "ais.csv")                   # Shape: (81907, 7) 
    fishingDF = pd.read_csv(InputDataDir / "fishing.csv")           # Shape: (644, 10)
    temperatureDF = pd.read_csv(InputDataDir / "temperature.csv")   # Shape: (43680000, 5)
    chlorophyllDF = pd.read_csv(InputDataDir / "chlorophyll.csv")   # Shape: (31449600, 4)

    print(fishingDF.shape)

    print(fishingDF)
    
    pd.get_option('display.max_columns')

    db = mongoDB(HOST, PORT, DBNAME)





















    #client = MongoClient()

#client = MongoClient('localhost', 27017)
#client = MongoClient('dtim.essi.upc.edu', 27017)


#db = client.DBkleber_reyes

#countries = db.country

#pprint.pprint(countries.find_one())