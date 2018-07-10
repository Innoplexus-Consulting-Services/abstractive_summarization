"""it dumps random data samples from mongo database."""
from pymongo import MongoClient
import json
from random import randint
import argparse
from random import random

parser = argparse.ArgumentParser()
parser.add_argument("--db", help="Specifies the dababase name")
parser.add_argument("--col", help="Specifies the collection name")
parser.add_argument('-u',"--user", help="Specifies the login user-name")
parser.add_argument("-p","--password", help="Specifies the password")
parser.add_argument("--ip", help="Specifies the server ip")
parser.add_argument("--port", help="Specifies the server port")
parser.add_argument("-s","--samples", help="Specifies the number of samples (-1 for all)", type=int)

args = parser.parse_args()

# Global parameters.
DB_NAME = args.db
COLLECTION_NAME = args.col
USER_NAME = args.user
PASSWORD = args.password
IP_ADD = args.ip
PORT = args.port
DOCUMENT_SAMPLES = args.doc  # mention 0 if the complete extraction is needed
DUMP_DATA_FILENAME = 'data_dump.json'   #filename where the data will be dumped

# QUERY selects the documents with both body and abstract available and
# not null and outputs just the Abstract and Body part of the find() query
CONDITION_QUERY = {"article.body": { "$exists": True, "$ne": None },\
"article.front.article-meta.Abstract": {"$exists": True, "$ne": None}}
OUTPUT_QUERY = {"article.body":1, "article.front.article-meta.Abstract":1, \
"_id":0}

def get_mongo_collection():
    """it returns a hoook to mongo."""
    mongo_connection = "mongodb://" + USER_NAME + ":" + PASSWORD + "@" + IP_ADD + ":" + PORT
    client = MongoClient(mongo_connection)
    db = client[DB_NAME]
    coll = db[COLLECTION_NAME]
    return coll

def get_data_from_db():
    """it extracts data in structured format as required by textsum."""
    collection = get_mongo_collection()
    overall_data = []
    count = 0
    """for mongo version >3.2"""
    # for document in collection.aggregate([{ "$sample" : {"size": (DOCUMENT_SAMPLES)}}, { "$match":  CONDITION_QUERY }]):
    """for all versions""""
    for document in collection.find(CONDITION_QUERY,OUTPUT_QUERY):
        if random() < .1:
            instance_data = {"abstract": document["article"]["front"]["article-meta"]["Abstract"], "body": document["article"]["body"]}
            overall_data.append(instance_data)
            count += 1
            if (NUMBER_OF_DOCUMENT_SAMPLES == 0):
                continue
            elif (count == NUMBER_OF_DOCUMENT_SAMPLES):
                break
        else:
            continue
    return overall_data

def dump_data_json(overall_data):
    """it dumps the loaded data in the predifined DUMP_DATA_FILENAME file in json format"""
    with open(DUMP_DATA_FILENAME, mode='w') as feed_json:
        json.dump(overall_data, feed_json)
        return True
    return False

if __name__ == "__main__":
    data = get_data_from_db()
    response = dump_data_json(data)
    if response:
        print("Extraction was successfull. You can find your data here:",
              DUMP_DATA_FILENAME)
    else:
        print("Extraction failed!")
        exit(-1)
