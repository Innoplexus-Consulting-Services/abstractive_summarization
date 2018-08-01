import requests
import json

url = "http://IP:4567/summarize"
###############################
# CAUTION: Do not add '\n' in #
# a single text sample. It is #
# used for seperation of      #
# multi-text documents.       #
###############################
text = ""
user = "user"
password  = "asdhBBBXyAH#$%^"
batch = [{"password":password,"username":user, "text": text.replace("\n",""), "ratio":0.05,"split":True}]
resp = requests.post(url, data={"batch": json.dumps(batch)})
print (resp.content)
        # b'{"results": [{"ratio": 0.05, "split": true, "summarized_text": ["Neoplasms change over time through a process of cell-level evolution, driven by genetic and epigenetic alterations.", "On the basis of a consensus conference of experts in the fields of cancer evolution and cancer ecology, we propose a framework for classifying tumours that is based on four relevant components.", "A classification system for the evolution and ecology of neoplasms would provide clinicians and researchers with a foundation for developing better prognostic and predictive assessments of tumour behaviour, such as response to an intervention.", "However, studies have not yet been done to test whether measures of the evolvability of a tumour from a single biopsy sample are sufficient or whether multiple samples substantially improve predictions of clinical outcomes15.", "Both diversity and changes in the clonal structure of a tumour over time are objective measures and may be assessed as part of preclinical studies or clinical trials."]}]}'

