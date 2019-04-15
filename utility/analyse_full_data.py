""" it analyses all the data samples of the database collection and give the required stats"""
#dependencies
from pymongo import MongoClient
import argparse
from scipy import stats
import numpy as np
import tableDisplay as td
import pickle
import time

parser = argparse.ArgumentParser()
parser.add_argument("--db", help="Specifies the dababase name")
parser.add_argument("--col", help="Specifies the collection name")
parser.add_argument('-u',"--user", help="Specifies the login user-name")
parser.add_argument("-p","--password", help="Specifies the password")
parser.add_argument("--ip", help="Specifies the server ip")
parser.add_argument("--port", help="Specifies the server port")
parser.add_argument("-s","--sample", help="How much part of the database to analyse (-1 for complete)", type=int)

args = parser.parse_args()

# Global parameters.
DB_NAME = args.db
COLLECTION_NAME = args.col
USER_NAME = args.user
PASSWORD = args.password
IP_ADD = args.ip
PORT = args.port
DOCUMENT_SAMPLES = args.doc  # mention 0 if the complete extraction is needed
body_length = []
abstract_length = []

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
    if client:
        print("Connection Made!")
    else:
        print("Connection Failed! :( ")
    db = client[DB_NAME]
    coll = db[COLLECTION_NAME]
    return coll

def get_data_from_db():
    """it extracts data in structured format as required by textsum."""
    collection = get_mongo_collection()
    count = 0
    """For mongo version > 3.2"""
    # for document in collection.aggregate([{ "$sample" : {"size": DOCUMENT_SAMPLES}}, { "$match":  CONDITION_QUERY }]):
    print('Progress in reading #documents: ')
    for document in collection.find(CONDITION_QUERY,OUTPUT_QUERY):
        instance_data = {"abstract": document["article"]["front"]["article-meta"]["Abstract"], "body": document["article"]["body"]}
        body_length.append([len(instance_data['body'])])
        abstract_length.append([len(instance_data['abstract'])])
        count += 1
        if count % 10000 == 0:
            print (count)
    print("Length Extraction Successful! Documents parsed: " + str(count))
    print()


def data_stats():
    """Writes the stats in txt file"""
    #Abstract stats
    print('Stats of data-abstract parts:- ')
    abstract_stats = stats.describe(abstract_length)
    abs_quartiles = np.percentile(abstract_length, [25,50,75])
    body_stats = stats.describe(body_length)
    body_quartiles = np.percentile(body_length, [25,50,75])

    tf = td.TableFormatter(["Stats-Criteria", "Abstract", "Body"])
    tf.setDataPadding(True)
    # tf.setDataAlignment(0)
    tf.setHeaderAlignment(1)
    #tf.setDataRowSeparation(False)
    #tf.setPaddingString('*')
    tf.addDataColumns(['Min', str(int(abstract_stats[1][0])), str(int(body_stats[1][0]))])
    tf.addDataColumns(['Max', str(int(abstract_stats[1][1])), str(int(body_stats[1][1]))])
    tf.addDataColumns(['Mean', str(int(abstract_stats[2])), str(int(body_stats[2]))])
    tf.addDataColumns(['Variance', str(abstract_stats[3]), str(body_stats[3])])
    tf.addDataColumns(['1st-Quartile', str(abs_quartiles[0]), str(body_quartiles[0])])
    tf.addDataColumns(['Median', str(abs_quartiles[1]), str(body_quartiles[1])])
    tf.addDataColumns(['3rd-Quartile', str(abs_quartiles[2]), str(body_quartiles[2])])
    tf.addDataColumns(['Skewness', str(int(abstract_stats[4])), str(int(body_stats[4]))])
    tf.addDataColumns(['Kurtosis', str(int(abstract_stats[5])), str(int(body_stats[5]))])
    with open('full_db_stats.txt', 'w') as f:
        f.write(tf.createTable() + '\n')
    print(tf.createTable())
    print()

def dump_data():
    """dumping the list for future distribution analysis"""
    print('Dumping body lengths ...')
    with open('full_body_lengths.pkl','wb') as f:
        pickle.dump(body_length, f)

    print('Dumping abstract lengths ...')
    with open('full_abs_lengths.pkl','wb') as f:
        pickle.dump(abstract_length, f)

    print('Congratulations! You are dumped ... ')
    time.sleep(5)
    print()
    print()
    print('... with the required data! :P')

if __name__ == "__main__":
    get_data_from_db()
    data_stats()
    choice = input("\nDo you wanna dump the body and abstract length lists?(y/n): ")
    if (choice == 'y') or (choice == 'Y'):
        dump_data()
    else:
        print('Thanks! :D ')
        exit(-1)
