import random
import requests
import os
import json
import together
from typing import List
import base64
from datetime import datetime, timedelta
from supabase import create_client, Client
import hashlib

table_name = "journal_entries"
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
embedding_model_string = 'togethercomputer/m2-bert-80M-8k-retrieval' # model API string from Together.
vector_database_field_name = 'embedding_together_m2-bert-8k-retrieval' # define your embedding field name.
headers = {
    "apikey": SUPABASE_ANON_KEY,
    "Content-Type": "application/json",
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}"
}
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def parse_simplified_date(date_str):
    try:
        journal_date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S%z').strftime('%m/%d/%Y')
    except ValueError:
        try:
            journal_date = datetime.strptime(date_str, '%m/%d/%Y').strftime('%m/%d/%Y')
        except ValueError:
            raise ValueError("Incorrect data format, should be MM/DD/YYYY")
        
    return journal_date

def parse_full_date(date_str):
    try:
        journal_date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S%z').strftime('%Y-%m-%dT%H:%M:%S%z')
    except ValueError:
        try:
            journal_date = datetime.strptime(date_str, '%m/%d/%Y').strftime('%Y-%m-%dT%H:%M:%S%z')
        except ValueError:
            raise ValueError("Incorrect data format, should be MM/DD/YYYY")
        
    return journal_date

def generate_embeddings(input_texts: List[str], model_api_string: str) -> List[List[float]]:
    """Generate embeddings from Together python library.

    Args:
        input_texts: a list of string input texts.
        model_api_string: str. An API string for a specific embedding model of your choice.

    Returns:
        embeddings_list: a list of embeddings. Each element corresponds to the each input text.
    """
    together_client = together.Together()
    outputs = together_client.embeddings.create(
        input=input_texts,
        model=model_api_string,
    )
    return [x.embedding for x in outputs.data]

def add_new_journal_entry(user_id, journal_title, journal_content, journal_date, audio_url=None, image_urls=None):
    """
    Creates a new entry in MongoDB collection, with its embeddings
    """
    journal_date = parse_simplified_date(journal_date)
    
    processed_string = \
f"""{journal_date} {journal_title}
{journal_content}
"""
    embedding = generate_embeddings([processed_string], embedding_model_string)

    # Define the data to be inserted
    data = {
        "title": journal_title,
        "body": journal_content,
        "created_at": journal_date,
        "user_id": user_id,
        "embedding_text": embedding[0]
    }

    data_json = json.dumps(data)
    endpoint = f"{SUPABASE_URL}/rest/v1/{table_name}"
    response = requests.post(endpoint, headers=headers, data=data_json)
    return response


DEFAULT_SUMMARIZE_PROMPT = f"From the given journal entries, summarize the content. Include how the user felt and what they did. Give trends and insights. Use first-person language."
KEY_FEELINGS_SUMMARIZE_PROMPT = f"From the given journal entries, provide key phrases describing appropriate themes and feelings of the writer."
KEY_IMAGE_SUMMARIZE_PROMPT = f"From the given journal entries, produce a DallE prompt to represent an abstract vibe of user's day. Keep it concise with max one sentence."
DEEPER_PROMPT = "Based on the summary, ask probing questions specific to the content that may lead to deeper thoughts. Limit to 3 questions. Use second-person language, directing the questinons towards 'you'."

def summarize_journal_entries(user_id, start_date, end_date, specific_focus=None, summarize_prompt=DEFAULT_SUMMARIZE_PROMPT):
    """
    Summarizes journal entries with different methods according to summarize_prompt,
    selects entries between start_date and end_date
    """
    start_date = parse_full_date(start_date)
    end_date = parse_full_date(end_date)

    endpoint = f"{SUPABASE_URL}/rest/v1/{table_name}?select=created_at,title,body&created_at=gte.{start_date}&created_at=lte.{end_date}&user_id=eq.{user_id}"
    response = requests.get(endpoint, headers=headers)
    if response.status_code == 200:
        data = response.json()
    else:
        return response

    # ! Format combined entries
    combined_journal_entries = ""
    for entry in data:
        journal_date = parse_simplified_date(entry['created_at'])
        combined_journal_entries += f"{journal_date} {entry['title']}\n{entry['body']}\n"

    # ! Final prompt
    augmented_prompt = f"""journal_entries:\n{combined_journal_entries}\n\n{summarize_prompt}"""
    return call_instruct_llm(augmented_prompt)

def summarize_and_deepize(user_id, start_date, end_date, specific_focus=None):
    start_date = parse_simplified_date(start_date)
    end_date = parse_simplified_date(end_date)

    summary = summarize_journal_entries(user_id, start_date, end_date, specific_focus, DEFAULT_SUMMARIZE_PROMPT)

    augmented_prompt = f"""{summary}\n{DEEPER_PROMPT}"""
    
    deeper_question = call_instruct_llm(augmented_prompt)
    longer_context = f"{summary}\n{deeper_question}"
    return deeper_question, longer_context

def receive_deeper_thoughts(user_id, user_response, previous_question, previous_context):
    """
    Receives the response from the user, and stores them in the database
    """
    question_answer_pair = f"{previous_question}\n{user_response}"
    add_new_journal_entry(user_id, "Deeper Thoughts", question_answer_pair, datetime.now().strftime('%m/%d/%Y'))

    DEEPERER_PROMPT = "Ask another question that may lead to even deeper thoughts. Use second-person language."
    augmented_prompt = f"""{previous_context}\n{user_response}\n{DEEPERER_PROMPT}"""
    deeper_question = call_instruct_llm(augmented_prompt)

    longer_context = f"{previous_context}\n{user_response}\n{deeper_question}"
    return deeper_question, longer_context

# def day_wise_summary(start_date: str, end_date: str, specific_focus=None):
#     start_date = parse_simplified_date(start_date)
#     end_date = parse_simplified_date(end_date)

#     combined_individual_summaries = ""
#     start_date = datetime.strptime(start_date, '%m/%d/%Y')
#     end_date = datetime.strptime(end_date, '%m/%d/%Y')
#     for individual_date in (start_date + timedelta(days=n) for n in range(int((end_date - start_date).days))):
#         individual_date_str = individual_date.strftime('%m/%d/%Y')
#         keywords = summarize_journal_entries(individual_date_str, individual_date_str, specific_focus, KEY_FEELINGS_SUMMARIZE_PROMPT)
#         combined_individual_summaries += f"{individual_date}: {keywords}\n"

#     KEYWORDS_CLUSTERING_PROMPT = "Based on the individual summaries, cluster the keywords into themes and feelings."
#     augmented_prompt = f"""keywords by date:\n{combined_individual_summaries}\n{KEYWORDS_CLUSTERING_PROMPT}"""
    
#     return call_instruct_llm(augmented_prompt)

STYLES = [
    "8-bit",
    "abstract",
    "art",
    "cartoon",
    "comic",
    "futuristic"
]

def generate_image(user_id, start_date: str, end_date: str, specific_focus=None):
    vision_generation_prompt = summarize_journal_entries(user_id, start_date, end_date, specific_focus, KEY_IMAGE_SUMMARIZE_PROMPT)

    # sample from styles
    modifier = random.choice(STYLES)
    styled_prompt = f"{modifier} style: {vision_generation_prompt}"
    print(styled_prompt)
    response = together.Image.create(prompt=styled_prompt, model="prompthero/openjourney", height=1024, width=1024, steps=50)
    image = response["output"]["choices"][0]
    with open("last_generated_image.png", "wb") as f:
        f.write(base64.b64decode(image["image_base64"]))

    image_path = "last_generated_image.png"
    bucket_id = "summary_images"

    with open(image_path, "rb") as f:
        # file = f.read(
        file_content = f.read()
        hash_value = hashlib.md5(file_content).hexdigest()
        supabase.storage.from_(bucket_id).upload(file=f, path=f"{user_id}_{hash_value}", file_options={"content-type": "image/jpeg"})

    return image["image_base64"], styled_prompt


def parse_llm_response(response):
    """
    Parses the response from LLM
    """
    # TODO: Error code handling
    return response["output"]["choices"][0]["text"]

def call_instruct_llm(prompt, model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    """
    Calls the instruct model
    """
    response = together.Complete.create(
        prompt=prompt,
        model=model,
        max_tokens = 512,
        temperature = 0.8,
        top_k = 60,
        top_p = 0.6,
        repetition_penalty = 1.1,
        stop = "\n\n\n",
    )
    return parse_llm_response(response)
