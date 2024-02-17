import requests
from openai import OpenAI
import os
import json
import together
import pymongo
from typing import List
from tqdm import tqdm
import time

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
YOUR_MONGODB_URI = os.environ.get("MONGODB_URI")
together.api_key = TOGETHER_API_KEY
client = pymongo.MongoClient(YOUR_MONGODB_URI)
embedding_model_string = 'togethercomputer/m2-bert-80M-8k-retrieval' # model API string from Together.
vector_database_field_name = 'embedding_together_m2-bert-8k-retrieval' # define your embedding field name.
NUM_DOC_LIMIT = 200 # the number of documents you will process and generate embeddings.

db = client.sample_airbnb
collection = db.listingsAndReviews

# ! Extract and generate embeddings for the documents.
keys_to_extract = ["name", "summary", "space", "description", "neighborhood_overview", "notes", "transit", "access", "interaction", "house_rules", "property_type", "room_type", "bed_type", "minimum_nights", "maximum_nights", "accommodates", "bedrooms", "beds"]
# Loop through each document in the collection where the 'summary' field exists, limiting the number of documents to NUM_DOC_LIMIT
for doc in tqdm(collection.find({"summary":{"$exists": True}}).limit(NUM_DOC_LIMIT), desc="Document Processing "):
  # Extract the values of the keys specified in keys_to_extract from the document and join them into a string, separated by newlines
  extracted_str = "\n".join([k + ": " + str(doc[k]) for k in keys_to_extract if k in doc])
  print(extracted_str)
  # If the document does not already have an embedding, generate one using the extracted string
  if vector_database_field_name not in doc:
    doc[vector_database_field_name] = generate_embeddings([extracted_str], embedding_model_string)[0]
  # Replace the existing document in the collection with the updated document (which now includes the embedding)
  collection.replace_one({'_id': doc['_id']}, doc)
  # Pause for 1 second before moving on to the next document
  time.sleep(1)


# ! RAG Query / Vector Search
# Example query.
query = "apartment with a great view near a coast or beach for 4 people"
query_emb = generate_embeddings([query], embedding_model_string)[0]

results = collection.aggregate([
  {
    "$vectorSearch": {
      "queryVector": query_emb,
      "path": vector_database_field_name,
      "numCandidates": 100, # this should be 10-20x the limit
      "limit": 10, # the number of documents to return in the results
      "index": "vector_index", # the index name you used in Step 4.
    }
  }
])
results_as_dict = {doc['name']: doc for doc in results}
# print(f"results_as_dict: {results_as_dict}")


# ! Print the results
print(f"From your query \"{query}\", the following airbnb listings were found:\n")
print("\n".join([str(i+1) + ". " + name for (i, name) in enumerate(results_as_dict.keys())]))

your_task_prompt = (
    "From the given airbnb listing data, summarize the general qualities of the offerings in the area. "
    "Talk about the range of things I can potentially do at each airbnb."
)
listing_data = ""
for doc in results_as_dict.values():
  listing_data += f"Listing name: {doc['name']}\n"
  for (k, v) in doc.items():
    if not(k in keys_to_extract) or ("embedding" in k): continue
    if k == "name": continue
    listing_data += k + ": " + str(v) + "\n"
  listing_data += "\n"

augmented_prompt = (
    "airbnb listing data:\n"
    f"{listing_data}\n\n"
    f"{your_task_prompt}"
)

time.sleep(1)
response = together.Complete.create(
    prompt=augmented_prompt,
    model="mistralai/Mistral-7B-Instruct-v0.2",
    max_tokens = 512,
    temperature = 0.8,
    top_k = 60,
    top_p = 0.6,
    repetition_penalty = 1.1,
    stop = "\n\n\n",
    )

print(response["output"]["choices"][0]["text"])



#json mode





# client = OpenAI(api_key=TOGETHER_API_KEY,
#   base_url='https://api.together.xyz',
# )

# url = "https://api.together.xyz/v1/embeddings"

# payload = {
#     "model": "togethercomputer/m2-bert-80M-8k-retrieval",
#     "input": "How is the weather"
# }
# headers = {
#     "accept": "application/json",
#     "content-type": "application/json",
#     "Authorization": f"Bearer {TOGETHER_API_KEY}"
# }

# response = requests.post(url, json=payload, headers=headers)

# response_json = json.loads(response.text)
# embedding_array = response_json.get('data')[0].get('embedding')
# print(embedding_array)


# chat_completion = client.chat.completions.create(
#   messages=[
#     {
#       "role": "system",
#       "content": "You are an AI assistant",
#     },
#     {
#       "role": "user",
#       "content": "Tell me about San Francisco",
#     }
#   ],
#   model="mistralai/Mixtral-8x7B-Instruct-v0.1",
#   max_tokens=1024
# )

# print(chat_completion.choices[0].message.content)

# url = "https://api.together.xyz/v1/embeddings"

# payload = {
#     "model": "togethercomputer/m2-bert-80M-8k-retrieval",
#     "input": "Our solar system orbits the Milky Way galaxy at about 515,000 mph"
# }
# headers = {
#     "accept": "application/json",
#     "content-type": "application/json"
# }

# response = requests.post(url, json=payload, headers=headers)

# print(response.text)