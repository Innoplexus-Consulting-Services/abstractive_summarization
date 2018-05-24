""" it reads the dumped data in json format and deliver the insights"""
#dependencies
import json
from pprint import pprint

#Global Variable definitions
DUMP_DATA_FILENAME = "10000_Training_data.json"


def get_json_file():
    """ reads the json file and returns the loaded data"""
    with open(DUMP_DATA_FILENAME, mode='r') as file_json:
        loaded_data = json.load(file_json)
    return loaded_data


if __name__ == '__main__':
    data = get_json_file()
    print ('Data read Successfully!')
    length_of_documents = len(data)
    print ( 'Number of document samples: ' + str(length_of_documents))

    #Average length of bodies and abstracts
    len_abs = []
    len_body = []
    for document in data:
        abstract = document['abstract']
        body = document['body']
        len_abs.extend([len(abstract)])
        len_body.extend([len(body)])
        #sample output testing

    #Info of Abstract and body
    print ("              Abstract Info                 ")
    print('Average length of abstract: %d ' % (sum(len_abs)/len(len_abs)))
    print('Abstract Max length %d and min length %d ' %(max(len_abs), min(len_abs)))
    print ("               Body Info                 ")
    print('Average length of body: %d ' % (sum(len_body)/len(len_body)))
    print('Body Max length %d and min length %d ' %(max(len_body), min(len_body)))
