import os
import json
import time
import unicodedata
import re

from collections import deque
from tqdm import tqdm


from dotenv import load_dotenv
from groq import Groq
from utilities import get_response, enforce_rate_limit, load_txt_data, load_json_data

MAX_REQUESTS_PER_MINUTE = 1000
REQUEST_WINDOW = 60
request_times = deque()

DOMAIN_EXAMPLES = {
    "laptop": {
        "dependent": [
            ("The battery drains quickly, but the customer support was helpful.", "The battery drains quickly.", "battery", "negative"),
            ("The screen is vibrant, though the packaging looked cheap.", "The screen is vibrant.", "screen", "positive"),
            ("I had issues with the keyboard layout, although the price was fair.", "I had issues with the keyboard layout.", "keyboard layout", "negative"),
        ],
        "independent": [
            ("The battery drains quickly, but the customer support was helpful.", "The customer support was helpful.", "customer support", "positive"),
            ("The screen is vibrant, though the packaging looked cheap.", "The packaging looked cheap.", "packaging", "negative"),
            ("I had issues with the keyboard layout, although the price was fair.", "The price was fair.", "price", "positive"),
        ]
    },
    "restaurant": {
        "dependent": [
            ("The steak was undercooked, but the ambience was cozy.", "The steak was undercooked.", "steak", "negative"),
            ("The waitress was inattentive, even though the background music was nice.", "The waitress was inattentive.", "waitress", "negative"),
            ("The pasta was perfectly cooked, although the chairs were uncomfortable.", "The pasta was perfectly cooked.", "pasta", "positive"),
        ],
        "independent": [
            ("The steak was undercooked, but the ambience was cozy.", "The ambience was cozy.", "ambience", "positive"),
            ("The waitress was inattentive, even though the background music was nice.", "The background music was nice.", "background music", "positive"),
            ("The pasta was perfectly cooked, although the chairs were uncomfortable.", "The chairs were uncomfortable.", "chairs", "negative"),
        ]
    },
    "book": {
        "dependent": [
            ("The plot was predictable, but I liked the style.", "The plot was predictable.", "plot", "negative"),
            ("The characters were well-developed, although the shipping took long.", "The characters were well-developed.", "characters", "positive"),
            ("The ending was rushed, though the price was reasonable.", "The ending was rushed.", "ending", "negative"),
        ],
        "independent": [
            ("The plot was predictable, but I liked the style.", "I liked the style.", "the style", "positive"),
            ("The characters were well-developed, although the shipping took long.", "The shipping took long.", "shipping", "negative"),
            ("The ending was rushed, though the price was reasonable.", "The price was reasonable.", "price", "positive"),
        ]
    }
}

PROMPT_TEMPLATES = {
    "dependent": """You are transforming sentences for few-shot learning in cross-domain aspect-based sentiment classification.

Given a sentence from the {domain} domain, rewrite it to retain only **domain-specific** content.

Your output must include:
- A domain-dependent aspect, related to the domain: {domain}. Keep the current aspect if it is already domain-dependent.
- A sentiment polarity (positive/negative/neutral) related to that aspect.

Remove domain-independent parts.

Output the transformed sentence, aspect in the new sentence, and polarity with regards to that aspect in a tuple: ("sentence","aspect","polarity").
Make sure this aspect is in the transformed sentence.
Do not include explanations or anything extra.

{examples}

Now transform the following sentence:
{sentence}
Current aspect: {aspect}

""",
"independent": """You are transforming sentences for few-shot learning in cross-domain aspect-based sentiment classification.

Given a sentence from the {domain} domain, rewrite it to retain only domain-independent content.

Your output must include:
- A domain-independent aspect, not related to the domain: {domain}.
- If current aspect is related to the domain: {domain}, change it to be domain-independent.
- A sentiment expression (positive/negative/neutral) related to that aspect.

Remove domain-specific parts.

Output the transformed sentence, aspect in the new sentence, and polarity with regards to that aspect in a tuple: ("sentence","aspect","polarity").
Make sure this aspect is in the transformed sentence.
Do not include explanations or anything extra.

{examples}

Now transform the following sentence:
{sentence}
Current aspect: {aspect}
""",
"basic": """
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
    Make sure the word "{aspect}" is present exactly as given in the paraphrased sentence.
    """
}


def fix_missing_spaces(text):
    # Add a space between a lowercase letter and a digit, if not already there
    return re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', text)

def normalize(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"'s\b", "", text)               
    text = re.sub(r"[\"'`‘’“”]", "", text)         
    text = re.sub(r"\s+", " ", text)
    text = fix_missing_spaces(text)
    return text



def is_aspect_in_response(aspect, response):
    norm_aspect = normalize(aspect)
    norm_response = normalize(response)
    return norm_aspect in norm_response


def get_response_with_correction(prompt, client, model, aspect, max_retries, i):
        
    model_map = {
        "gpt-4o": "gpt-4o-mini",
        "llama3": "llama3-70b-8192",
        "llama4_scout": "meta-llama/llama-4-scout-17b-16e-instruct",
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
    
    
    while not is_aspect_in_response(aspect, response) and max_retries > 0:
        print(f"\n Reprompting for index {i}")
        correction_prompt = f"""
        The previous response was a paraphrase of the sentence, but it did not include the required aspect term "{aspect}".

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

def build_prompt(domain: str, transformation_type: str, sentence: str, aspect) -> str:
    assert domain in DOMAIN_EXAMPLES, f"Unknown domain: {domain}"
    assert transformation_type in ["dependent", "independent", "basic"], "transformation_type must be 'dependent', 'independent', or 'basic'"
    
    examples = DOMAIN_EXAMPLES[domain][transformation_type]
    formatted_examples = "\n".join(
        f"""Example {i+1}:
        Original: {orig}
        Aspect: {aspect}
        Polarity: {polarity}
        Output: {out}""" for i, (orig, out, aspect, polarity) in enumerate(examples)
            )
    
    template = PROMPT_TEMPLATES[transformation_type]
    prompt = template.format(domain=domain, examples=formatted_examples, sentence=sentence, aspect=aspect)
    return prompt





def get_transformation(prompt_version, domain, sentence, aspect, client, model_name, i):
    prompt = build_prompt(domain, transformation_type=prompt_version, sentence=sentence, aspect=aspect)

    try:
        enforce_rate_limit(request_times=request_times, MAX_REQUESTS_PER_MINUTE=MAX_REQUESTS_PER_MINUTE, REQUEST_WINDOW=REQUEST_WINDOW)
        response = get_response_with_correction(prompt, client, model_name, aspect, 0, i)
        return response.strip()
    except Exception as e:
        print(f" \n [Error] Paraphrasing failed: {e}")
        return ""


def transform_and_cache(domain, prompt_version, data, cache_path, model_name, api_key):
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            data = json.load(f)
            complete = True
            for i in range(len(data)):
                sample = data[i]
                if "paraphrased_text" not in sample or sample["paraphrased_text"] == "{}":
                    complete = False
            
            if complete:
                print("[CACHE] Retrieving transformed data...")
                return data


    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
 
    print(f"[Transforming] Paraphrasing data and caching to {cache_path}")
    client = Groq(api_key=api_key)     

    for i in tqdm(range(len(data))):
        sample = data[i]
        if "paraphrased_text" not in sample or sample["paraphrased_text"] == "{}":
            sentence = sample["text"]
            aspect = sample["aspect"]
            sample["paraphrased_text"]= get_transformation(prompt_version, domain, sentence, aspect, client, model_name, i)

            # if not is_aspect_in_response(aspect, sample['paraphrased_text']):
            #     paraphrased_sentence = sample["paraphrased_text"]
            #     print(f"Sentence at index {i} is invalid, the aspect was: {aspect} and the paraphrased sentence was: {paraphrased_sentence}] \n Falling back to the original sentence: {sample['text']}")
            #     sample["paraphrased_text"] = sample['text']
                        
            with open(cache_path, "w") as f:
                    json.dump(data, f, indent=2)
       
    
    return data   



# if you want to transform a seperate path without doing the classification, run the main function in this file
if __name__ == "__main__":
    
    load_dotenv() 
    key_groq = os.getenv("GROQ_API_KEY")
    key_groq_paid = os.getenv("GROQ_PAID_KEY")

    model = "llama4_scout"
    train_domains = ["laptop"]
    test_domains = []
    prompt_versions = ["dependent", "independent"]
    for prompt_version in prompt_versions:
        for train_domain in train_domains:
            year = 2019 if train_domain == "book" else 2014
            train_file_path = f"data_out/{train_domain}/raw_data_{train_domain}_train_{year}.txt"
            train_data = load_txt_data(train_file_path)
            train_data = transform_and_cache(
                domain=train_domain,
                data=train_data,
                prompt_version=prompt_version,
                cache_path=f"cache/{model}/paraphrased/{prompt_version}_train_data_{train_domain}.json",
                model_name=model,
                api_key=key_groq_paid
            )
    for prompt_version in prompt_versions:
        for test_domain in test_domains:
            year = 2019 if test_domain == "book" else 2014
            test_file_path = f"data_out/{test_domain}/raw_data_{test_domain}_test_{year}.txt"
            test_data = load_txt_data(test_file_path)
        
            test_data = transform_and_cache(
                domain=train_domain,
                data=test_data,
                prompt_version=prompt_version
                cache_path=f"cache/{model}/paraphrased/test_data_{test_domain}.json",
                model_name=model,
                api_key=key_groq_paid
            ) 
   
       