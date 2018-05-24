"""it creates training data for textsum."""
from pymongo import MongoClient
import json
from random import randint

# Global parameters.
# QUERY selects the documents with both body and abstract available and not null and outputs just the Abstract and Body part of the find() query with random skipping
CONDITION_QUERY = {"article.body": { "$exists": True, "$ne": None }, "article.front.article-meta.Abstract": {"$exists": True, "$ne": None}}
OUTPUT_QUERY = {"article.body":1, "article.front.article-meta.Abstract":1, "_id":0}
COLLECTION_NAME = "flat_pmc_data"

# TODO: Take all these sensitive information from command line.
DB_NAME = "testpmc"
USER_NAME = 'read'
PASSWORD = 'fdfREsse'
IP_ADD = '10.240.0.14'
PORT = '10070'
NUMBER_OF_DOCUMENT_SAMPLES = 100     # mention 0 if the complete extraction is needed
DUMP_DATA_FILENAME = '100_Training_sample_data.json'      #filename where the data will be dumped


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
    #TODO: Change this to random -https://docs.mongodb.com/manual/reference/operator/aggregation/sample/
    for document in collection.find(CONDITION_QUERY, OUTPUT_QUERY).skip(randint(1,10000)):
        instance_data = {"abstract": document["article"]["front"]["article-meta"]["Abstract"], "body": document["article"]["body"]}
        overall_data.append(instance_data)
        count += 1
        if (NUMBER_OF_DOCUMENT_SAMPLES == 0):
            continue
        elif (count == NUMBER_OF_DOCUMENT_SAMPLES):
            break
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
