""" it reads the dumped data in json format and converts it into textsum compatible format of data"""
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
