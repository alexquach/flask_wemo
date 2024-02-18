from io import BytesIO
from flask import Flask, jsonify, request
import os
import modal
from PIL import Image

app = Flask(__name__)

from llm_utils import *

@app.route('/')
def index():
    return jsonify({"Hehe": "Welcome to the empty Flask app home page 🚅"})

@app.route('/create_journal', methods=['POST'])
def create_journal():
    data = request.form
    user_id = data.get('user_id')
    journal_title = data.get('title')
    journal_content = data.get('body')
    journal_date = data.get('created_at')
    audio_url = data.get('audio_url', None)
    image_urls = data.get('image_urls', None)
    
    response = add_new_journal_entry(user_id, journal_title, journal_content, journal_date, audio_url, image_urls)
    if response.status_code == 201:
        return jsonify({"status": "success", "message": "Journal entry created successfully"})
    else:
        return jsonify({"status": "error", "message": "Failed to create journal entry", "error": response.text}), response.status_code

@app.route('/generate_embedding', methods=['GET'])
def generate_and_store_embedding_endpoint():
    data = request.args
    entry_id = data.get('entry_id')
    
    response = generate_and_store_embedding(entry_id)
    return jsonify(response)

DEFAULT_SUMMARIZE_PROMPT = f"From the given journal entries, summarize the content. Include how the user felt and what they did. Give trends and insights. Use first-person language."
KEY_FEELINGS_SUMMARIZE_PROMPT = f"From the given journal entries, provide key phrases describing appropriate themes and feelings of the writer."
KEY_IMAGE_SUMMARIZE_PROMPT = f"From the given journal entries, produce a DallE prompt to represent an abstract vibe of user's day. Keep it concise with max one sentence."

@app.route('/summary', methods=['GET'])
def summary():
    user_id = request.args.get('user_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    specific_focus = request.args.get('specific_focus', None)

    summary = summarize_journal_entries(user_id, start_date, end_date, specific_focus=specific_focus, summarize_prompt=DEFAULT_SUMMARIZE_PROMPT)
    return jsonify(summary)

@app.route('/summarize_and_deepize', methods=['GET'])
def summarize_and_deepize_endpoint():
    user_id = request.args.get('user_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    specific_focus = request.args.get('specific_focus', None)

    deeper_question, longer_context = summarize_and_deepize(user_id, start_date, end_date, specific_focus=specific_focus)
    return jsonify({"deeper_question": deeper_question, "longer_context": longer_context})

@app.route('/go_deeper', methods=['GET'])
def go_deeper_endpoint():
    user_id = request.args.get('user_id')
    user_response = request.args.get('user_response')
    previous_question = request.args.get('previous_question')
    previous_context = request.args.get('previous_context')

    deeper_question, longer_context = receive_deeper_thoughts(user_id, user_response, previous_question, previous_context)
    return jsonify({"deeper_question": deeper_question, "longer_context": longer_context})

@app.route('/generate_image', methods=['GET'])
def generate_image_endpoint():
    user_id = request.args.get('user_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    specific_focus = request.args.get('specific_focus', None)

    base64_image, styled_prompt = generate_image(user_id, start_date, end_date, specific_focus=specific_focus)

    return jsonify({"styled_prompt": styled_prompt, "base64_image": base64_image})

@app.route('/process_image_embeddings', methods=['GET'])
def process_image_embeddings_endpoint():
    user_id = request.args.get('user_id')
    entry_id = request.args.get('entry_id')
    embedding_exists, array_urls = does_embedding_images_exist(user_id, entry_id)
    if not embedding_exists and array_urls is not None:
        # read images from array_urls
        blip_images = []
        for path in array_urls:
            url = f"{SUPABASE_URL}/storage/v1/object/public/journal_images/{path}"
            response = requests.get(url, headers=headers)
            image = Image.open(BytesIO(response.content))
            blip_images.append(image)

        # blip_image = Image.open("last_generated_image.png")
        f = modal.Function.lookup("image-to-text-app", "image_to_text")
        text_descriptions = f.remote(blip_images)

        # update text_description in the database
        response = store_image_embeddings(entry_id, text_descriptions)

        return jsonify(text_descriptions)
    else:
        return jsonify({"status": "error", "message": "Embeddings for this image already exist"})


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
