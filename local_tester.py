import os
import json
import together
import pymongo
from typing import List
from tqdm import tqdm

from llm_utils import *

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
YOUR_MONGODB_URI = os.environ.get("MONGODB_URI")
YOUR_POSTGRES_URI = os.environ.get("POSTGRES_URI")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

together.api_key = TOGETHER_API_KEY
embedding_model_string = 'togethercomputer/m2-bert-80M-8k-retrieval' # model API string from Together.
vector_database_field_name = 'embedding_together_m2-bert-8k-retrieval' # define your embedding field name.
NUM_DOC_LIMIT = 200 # the number of documents you will process and generate embeddings.

# ! DATABASE ENTRY
# table_name = "journal_entries"
# journal_title = """
# Job Interview
# """
# journal_entry = """
# I had a job interview this morning for a marketing coordinator role. I prepared and practiced my responses, but still felt nervous, especially at the start. Overall I think it went pretty well. Some questions threw me off guard but I tried to keep my composure. Now the waiting game begins to see if I get called back.
# """
# journal_date = "07/03/2022"
# response = add_new_journal_entry(1, journal_title, journal_entry, journal_date)


start_date = "07/01/2022"
end_date = "07/03/2022"
summary = day_wise_summary(start_date, end_date)
print(summary)

# # ! RAG Query / Vector Search
# # Example query.
# query = "apartment with a great view near a coast or beach for 4 people"
# query_emb = generate_embeddings([query], embedding_model_string)[0]

# results = collection.aggregate([
#   {
#     "$vectorSearch": {
#       "queryVector": query_emb,
#       "path": vector_database_field_name,
#       "numCandidates": 100, # this should be 10-20x the limit
#       "limit": 10, # the number of documents to return in the results
#       "index": "vector_index", # the index name you used in Step 4.
#     }
#   }
# ])
# results_as_dict = {doc['name']: doc for doc in results}
# # print(f"results_as_dict: {results_as_dict}")



#json mode