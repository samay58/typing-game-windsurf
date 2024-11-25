from flask import Flask, render_template, jsonify
import random
import os
from openai import OpenAI
from dotenv import load_dotenv
import threading
from flask import Response
import time
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Store generated images in memory cache
image_cache = {}

# Sample text passages for typing practice
SAMPLE_TEXTS = [
    "The old maple tree stood majestically in the center of the park casting long shadows across the freshly cut grass as children played beneath its branches swaying gently in the summer breeze.",
    "In the depths of the ancient library dusty volumes lined endless shelves while scholars hunched over manuscripts searching for long forgotten wisdom from civilizations past.",
    "Waves crashed rhythmically against the weathered cliffs as seabirds soared overhead their mournful cries echoing across the desolate beach at sunrise.",
    "Through winding cobblestone streets the aroma of freshly baked bread wafted from small cafes while early morning shoppers bustled about their daily routines.",
    "Deep in the misty forest ancient trees whispered secrets to one another their branches intertwined creating a natural cathedral ceiling high above the forest floor.",
    "The garden burst with vibrant colors as butterflies danced from flower to flower while bees hummed their gentle songs collecting nectar in the warm afternoon sun.",
    "Along the quiet country road wildflowers swayed in the gentle breeze their colors painting a natural masterpiece across the rolling hills that stretched toward the horizon.",
    "In the bustling marketplace vendors called out to passing customers as the scent of exotic spices and fresh fruits filled the air creating an atmosphere of excitement and discovery.",
    "Beneath the starlit sky crickets performed their nightly symphony while fireflies danced through the tall grass creating a magical display of natural light.",
    "The old grandfather clock in the hallway marked time with steady precision its gentle ticking echoing through the quiet house as afternoon sunlight streamed through lace curtains."
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-text')
def get_text():
    text = random.choice(SAMPLE_TEXTS)
    session_id = str(time.time())  # Use timestamp as session ID
    
    # Start image generation in background
    def generate_image():
        try:
            logger.info(f"Starting image generation for session {session_id}")
            image_prompt = " ".join(text.split()[:10]) + "..."
            logger.info(f"Image prompt: {image_prompt}")
            
            response = client.images.generate(
                model="dall-e-3",
                prompt=image_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            logger.info(f"Image generated successfully: {image_url}")
            image_cache[session_id] = image_url
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            image_cache[session_id] = None

    # Start image generation in background
    thread = threading.Thread(target=generate_image)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'text': text,
        'session_id': session_id
    })

@app.route('/check-image/<session_id>')
def check_image(session_id):
    logger.info(f"Checking image for session {session_id}")
    if session_id in image_cache:
        image_url = image_cache[session_id]
        logger.info(f"Image found in cache: {image_url}")
        return jsonify({
            'ready': True,
            'image_url': image_url
        })
    logger.info(f"Image not ready yet for session {session_id}")
    return jsonify({'ready': False})

# Cleanup old cache entries periodically
def cleanup_cache():
    while True:
        time.sleep(300)  # Clean every 5 minutes
        current_time = time.time()
        for session_id in list(image_cache.keys()):
            if current_time - float(session_id) > 300:  # Remove entries older than 5 minutes
                del image_cache[session_id]

threading.Thread(target=cleanup_cache, daemon=True).start()

if __name__ == '__main__':
    app.run(debug=True)
