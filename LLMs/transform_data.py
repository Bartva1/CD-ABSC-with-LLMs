import os
import json
import time
from collections import deque
from tqdm import tqdm

from dotenv import load_dotenv
from groq import Groq



MAX_REQUESTS_PER_MINUTE = 30
REQUEST_WINDOW = 60
request_times = deque()

def enforce_rate_limit():
    current_time = time.time()
    while request_times and current_time - request_times[0] > REQUEST_WINDOW:
        request_times.popleft()
    if len(request_times) >= MAX_REQUESTS_PER_MINUTE:
        sleep_time = REQUEST_WINDOW - (current_time - request_times[0]) + 0.1
        time.sleep(sleep_time)
        return enforce_rate_limit()
    request_times.append(current_time)


def get_response(prompt, client, model):
    model_map = {
        "gpt-4o": "gpt-4o-mini",
        "llama3": "llama3-70b-8192",
        "llama4": "meta-llama/llama-4-scout-17b-16e-instruct",
        "deepseek_llama": "deepseek-r1-distill-llama-70b",
        "gemma": "gemma2-9b-it",
        "qwen32": "qwen-qwq-32b",
        "llama4_mav": "meta-llama/llama-4-maverick-17b-128e-instruct"
    }
    model = model_map.get(model, model)
    messages = [{"role": "user", "content": prompt}]
    output = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return output.choices[0].message.content



    # Example:
    # - Sentence: The battery life of this laptop is impressive.
    # - Aspect: battery life
    # - Paraphrased: The duration the device can be used without recharging is impressive.

def paraphrase_sentence(sentence, aspect, client, model_name):
    prompt = f"""
    Instruction:
    You will receive a sentence containing an aspect. Please paraphrase the sentence to preserve the meaning and sentiment, while replacing any domain-specific terminology with domain-neutral or generic phrasing. Maintain the aspect term as is.


    Tested Sample: 
    - Sentence: {sentence}
    - Aspect: {aspect}

    Output: Just provide the paraphrased sentence. No extra text or formatting.
    """
    try:
        enforce_rate_limit()
        response = get_response(prompt, client, model_name)
        return response.strip()
    except Exception as e:
        print(f"[Error] Paraphrasing failed: {e}")
        return sentence  # fallback to original



def transform_and_cache(data, cache_path, model_name, api_key):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        print(f"[Cache] Loaded paraphrased data from {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    print(f"[Transforming] Paraphrasing data and caching to {cache_path}")
    client = Groq(api_key=api_key)

    for sample in tqdm(data):
        sentence = sample["text"]
        aspect = sample["aspect"]
        sample["paraphrased_text"] = paraphrase_sentence(sentence, aspect, client, model_name)

    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)

    return data
