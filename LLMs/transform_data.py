import os
import json
import time
from collections import deque
from tqdm import tqdm

from dotenv import load_dotenv
from groq import Groq
from utilities import get_response, enforce_rate_limit, load_txt_data, load_json_data

def get_response_with_correction(prompt, client, model, aspect, max_retries, i):
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
    
    output = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    
    response = output.choices[0].message.content.strip()
    
    # Check if aspect is missing
    while aspect not in response and max_retries > 0:
        print(f"\n Reprompting for index {i}")
        correction_prompt = f"""
        The previous response was a paraphrase of a sentence, but it did not include the required aspect term "{aspect}".

        Please revise the paraphrased sentence to:
        - Include the word "{aspect}" exactly as written
        - Preserve the original meaning and sentiment with regards to the word "{aspect}".

        The sentence you wrote was:
        "{response}"

        Now provide the corrected paraphrase using "{aspect}":
        """
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": correction_prompt})
        
        correction_output = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0
        )
        response = correction_output.choices[0].message.content.strip()
        max_retries -= 1
    
    return response


def paraphrase_sentence(sentence, aspect, client, model_name, i):
    prompt = f"""
    Instruction:
    You will be given a sentence and an aspect term. 
    
    Your task is to paraphrase the sentence so that:
    - The meaning and sentiment with regards to the aspect term "{aspect}" remain unchanged.
    - Any domain-specific words are replaced with generic language.
    - The word "{aspect}" MUST appear exactly as given, with no changes.

    Test Sample:
    - Sentence: {sentence}
    - Aspect: {aspect}

    Output:
    Return ONLY the paraphrased sentence. Do NOT add extra text or formatting.
    Make sure the word "{aspect}" present exactly as given in the paraphrased sentence.
    """

    try:
        enforce_rate_limit(request_times=request_times, MAX_REQUESTS_PER_MINUTE=MAX_REQUESTS_PER_MINUTE, REQUEST_WINDOW=REQUEST_WINDOW)
        response = get_response_with_correction(prompt, client, model_name, aspect, 1, i)
        return response.strip()
    except Exception as e:
        print(f" \n [Error] Paraphrasing failed: {e}")
        return ""


def transform_and_cache(data, cache_path, model_name, api_key):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
    #     print(f"[Cache] Loaded paraphrased data from {cache_path}")
        
    #     with open(cache_path, "r") as f:
    #         loaded_f = json.load(f)
    #         if "{}" not in loaded_f:
    #             return loaded_f
            
        data = load_json_data(cache_path)
        

    print(f"[Transforming] Paraphrasing data and caching to {cache_path}")
    client = Groq(api_key=api_key)     

    for i in tqdm(range(len(data))):
        sample = data[i]
        if "paraphrased_text" not in sample or sample["paraphrased_text"] == "{}":
            sentence = sample["text"]
            aspect = sample["aspect"]
            sample["paraphrased_text"]= paraphrase_sentence(sentence, aspect, client, model_name, i)

            if aspect not in sample["paraphrased_text"]:
                paraphrased_sentence = sample["paraphrased_text"]
                print(f"Sentence at index {i} is invalid, the aspect was: {aspect} and the paraphrased sentence was: {paraphrased_sentence}]")
                sample["paraphrased_text"] = "{}"
        
            with open(cache_path, "w") as f:
                    json.dump(data, f, indent=2)

       

    return data

# if you want to transform a seperate path without doing the classification, run the main function in this file
if __name__ == "__main__":
    MAX_REQUESTS_PER_MINUTE = 30
    REQUEST_WINDOW = 60
    request_times = deque()

    load_dotenv() 
    key_groq = os.getenv("GROQ_API_KEY")

    model = "llama3"
    train_domains = ["laptop"]
    test_domains = []
    for train_domain in train_domains:
        year = 2019 if train_domain == "book" else 2014
        train_file_path = f"data_out/{train_domain}/raw_data_{train_domain}_train_{year}.txt"
        train_data = load_txt_data(train_file_path)
        test_data = transform_and_cache(
            data=train_data,
            cache_path=f"cache/{model}_paraphrased_train_{train_domain}.json",
            model_name=model,
            api_key=key_groq
        )
    for test_domain in test_domains:
        year = 2019 if test_domain == "book" else 2014
        test_file_path = f"data_out/{test_domain}/raw_data_{test_domain}_test_{year}.txt"
        test_data = load_txt_data(test_file_path)
       
        test_data = transform_and_cache(
            data=test_data,
            cache_path=f"cache/{model}_paraphrased_test_{test_domain}.json",
            model_name=model,
            api_key=key_groq
        ) 
       