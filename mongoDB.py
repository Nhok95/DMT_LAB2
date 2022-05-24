from pymongo import MongoClient


class mongoDB:

    '''
    This class represents a Mongo Data Base
    '''

    def __init__(self, host, port, db_name):
        client = MongoClient(host = host, port = port)
        self.database = client[db_name]
