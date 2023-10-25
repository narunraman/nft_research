import requests
import json
from pymongo import MongoClient

class MongoDB(object):
    def __init__(self, host='localhost', port=27017, database_name=None, collection_name=None):
        try:
            self._connection = MongoClient(host=host, port=port, maxPoolSize=200)
        except Exception as error:
            raise Exception(error)
        self._database = None
        self._collection = None
        if database_name:
            self._database = self._connection[database_name]
        if collection_name:
            self._collection = self._database[collection_name]

    def insert(self, post):
        # add/append/new single record
        post_id = self._collection.insert_one(post).inserted_id
        return post_id


url = "https://testnets-api.opensea.io/api/v1/collections?offset=0&limit=300"

headers={'Accept':'application/json'}
params={'limit':300}

response = requests.request("GET", url, headers=headers, params=params)
print(response.text)
data_list = response.json()['collections']


# check empty lists
# for collection in data_list:
#     if 'year,data' not in collection:
#         if value:
#             value = value.split(',')
#             data_list.append({'year': int(value[0]), 'data': float(value[1])})

# print('[*] Pushing data to MongoDB ')
# mongo_db = MongoDB(database_name='Climate_DB', collection_name='climate_data')

# for collection in data_list:
#     print('[!] Inserting - ', collection)
#     mongo_db.insert(collection)