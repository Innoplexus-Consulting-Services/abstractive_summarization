'''
iPlexus Entity tagger

Contact Vivek Verma for any queries
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from elasticsearch import Elasticsearch
import redis
import re


INDEX = "publications_alias"
REDIS_DB = 6
class Analyze(object):
    """Class which analyzes a given text and outputs the concepts
       in the given text.
    """
    def __init__(
        self,
        es_host,
        es_port,
        auth,
        analyzer,
        redis_host,
        redis_port
    ):
        '''Constructor for the given class.
        '''
        self.host = es_host
        self.port = es_port
        self.index = INDEX
        self.auth = auth
        self.REDIS_HOST = redis_host
        self.REDIS_PORT = redis_port
        self.redis_db = REDIS_DB
        self.redis_connection_obj = None
        self.analyzer = analyzer

    def create_es_connection(self):
        '''Creates a client for Elasticsearch.
        '''
        es_client = Elasticsearch(host=self.host,port=self.port,\
                    http_auth=self.auth)

        return es_client

    def invoke_analyze(self, analyzer, text):
        '''Function to get seperate token in a json format.
        '''
        client = self.create_es_connection()
        body = {
            "analyzer": analyzer,
            "text": text
        }
        response = client.indices.analyze(index=self.index, body=body)
        return response

    def create_redis_connection(self, redis_db):
        '''
        Description:
                    Initialize Redis connection
        Args:
            self.redis_connection_config (list): redis configuration
        Returns:
            self.r (object): redis instance

        '''
        self.redis_connection_obj = redis.StrictRedis(
            host=self.REDIS_HOST,
            port=self.REDIS_PORT,
            db=redis_db
        )
        return self.redis_connection_obj

    def get_ids(self, response):
        '''Function that returns all possible concepts.
        '''
        id_dict = {
            "indications": [],
            "drugs": [],
            "proteins": [],
            "genes": []
        }
#         entities = {
#             "indications": [],
#             "drugs": [],
#             "proteins": [],
#             "genes": []
#         }
        disease_regex = re.compile('(DISE)')
        drug_regex = re.compile('(DRUG)')
        gene_regex = re.compile('(PROT)')
        prot_regex = re.compile('(GENE)')
        for token in response['tokens']:
            # print(token)
            if token['type'] == 'SYNONYM':
                if disease_regex.match(token['token']):
                    id_dict['indications'].append((token['token'],token['start_offset'],token['end_offset']))
                elif drug_regex.match(token['token']):
                    id_dict['drugs'].append((token['token'],token['start_offset'],token['end_offset']))
                elif prot_regex.match(token['token']):
                    id_dict['proteins'].append((token['token'],token['start_offset'],token['end_offset']))
                elif gene_regex.match(token['token']):
                    id_dict['genes'].append((token['token'],token['start_offset'],token['end_offset']))
                else:
                    pass
            else:
                continue
        return id_dict

    def redis_mget(self, id_list):
        '''Simple implementation of the redis mget function
        '''
        return self.redis_connection_obj.mget(id_list)

    def get_concepts_from_ids(self, id_dict):
        '''Function that returns preferred terms of concepts based
           on the IDs.
        '''
        self.create_redis_connection(self.redis_db)
        concepts_dict = {
            "indications": [],
            "drugs": [],
            "proteins": [],
            "genes": []
        }
        if id_dict['indications']:
            concepts_dict['indications'].extend(
                self.redis_mget(id_dict['indications'])
            )
        if id_dict['drugs']:
            concepts_dict['drugs'].extend(
                self.redis_mget(id_dict['drugs'])
            )
        if id_dict['proteins']:
            concepts_dict['proteins'].extend(
                self.redis_mget(id_dict['proteins'])
            )
        if id_dict['genes']:
            concepts_dict['genes'].extend(
                self.redis_mget(id_dict['genes'])
            )
        return concepts_dict

    def post_process_concepts(self, concepts):
        for key, value in concepts.iteritems():
            concepts[key] = list(set(value))
        return concepts

    def convert_in_format(self,text,ids):
        format_out = {'text':text,
                      'ents':[]}
        for key, values in ids.items():
            for value in values:
                format_out['ents'].append({'start':value[1],'end':value[2],'label':key})
        return format_out

    def analyze(self, text):
        '''Function which contains all the logic to return the preferred terms
        of the concepts.
        '''
#         tracer = logging.getLogger('elasticsearch')
#         tracer.setLevel(logging.CRITICAL)
        response = self.invoke_analyze(self.analyzer, text)
        ids = self.get_ids(response)
        # concepts = self.get_concepts_from_ids(ids)
        # concepts = self.post_process_concepts(concepts)
        return self.convert_in_format(text,ids)

# On Cloud use this
# my_analyze = Analyze("10.240.0.146", 9400,('ctcuser', 'WSZwd3dd34ff4f'),  "analyzer_summary", "10.240.0.53", 6379)
# On Local Use this
my_analyze = Analyze("35.185.19.184",9400,('ctcuser', 'WSZwd3dd34ff4f'),"analyzer_summary", "10.240.0.53", 6379)

def tag_entities(text):
    concepts = my_analyze.analyze(text)
    return concepts

if __name__ == '__main__':
    text = "A Phase 1b, Multicenter, Randomized, Double-Blind, Placebo-Controlled, Dose-Escalation Study to Evaluate the Safety and Tolerability of Multiple Intravenous Doses of MEDI-545, a Fully Human Anti-Interferon-Alpha Monoclonal Antibody, in Patients With Systemic Lupus Erythematosus"
    concepts = tag_entities(text)
