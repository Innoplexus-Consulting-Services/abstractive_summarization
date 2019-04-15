"""it dumps random data samples from mongo database and
    drops it individually in a folder (Best for handling
    large number of samples)."""
from pymongo import MongoClient
import json
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
parser.add_argument("-f","--outFolder", help="Specifies the folder to dump instance json files", type=str)

args = parser.parse_args()

# Global parameters.
DB_NAME = args.db
COLLECTION_NAME = args.col
USER_NAME = args.user
PASSWORD = args.password
IP_ADD = args.ip
PORT = args.port
DOCUMENT_SAMPLES = args.samples  # mention -1 if the complete extraction is needed
out_folder = args.outFolder
out_file = 'sample_'  #filename where the data will be dumped

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
    print("Connection Made! \n")
    db = client[DB_NAME]
    coll = db[COLLECTION_NAME]
    return coll

def dump_data_json(out_file_name, instance_data):
    """it dumps the instances in the predifined directory 'out_folder' as files in json format"""
    with open(out_folder + '/' + out_file_name, mode='w') as out_json:
        json.dump(instance_data, out_json)
        return True
    return False

def get_data_from_db():
    """it extracts data in structured format as required by textsum."""
    collection = get_mongo_collection()
    count = 1
    """for mongo version >3.2"""
    # for document in collection.aggregate([{ "$sample" : {"size": (DOCUMENT_SAMPLES)}}, { "$match":  CONDITION_QUERY }]):
    """for all versions"""
    print("Data quering and dumping started ....")
    for document in collection.find(CONDITION_QUERY,OUTPUT_QUERY):
        instance_data = {"abstract": document["article"]["front"]["article-meta"]["Abstract"], "body": document["article"]["body"]}
        instance_data['abstract'] = instance_data['abstract'].lower()
        instance_data['body'] = instance_data['body'].lower()

        out_file_name = out_file + str(count) + '.json'
        if not dump_data_json(out_file_name, [instance_data]):
            print("Extraction failed at file count " + str(count))
        count += 1

        #progress marker
        if count % 50000 == 0:
            print count

        #conditional breaking
        if (DOCUMENT_SAMPLES == -1):
            continue
        elif (count == DOCUMENT_SAMPLES):
            print(".... quering and dumping ends! Extraction was successfull!!")
            break

if __name__ == "__main__":
    data = get_data_from_db()
