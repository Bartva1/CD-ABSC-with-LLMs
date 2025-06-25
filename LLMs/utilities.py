"""
utilities.py

General utility functions for data loading, normalization, LLM interaction, rate limiting,
and experiment configuration for cross-domain aspect-based sentiment classification.

This module is intended for functions that are used in multiple scripts or are general-purpose helpers.
"""


import os
import time
import json
import numpy as np
import re
import google.generativeai as genai
import argparse
import sys

from typing import Deque


def get_directory(
    demo: str, 
    model: str, 
    shot_info: dict
) -> str:
    """
    Returns the directory path for storing results based on demo, model, and shot_info.

    Args:
        demo: The demonstration type.
        model: The model name.
        shot_info: Dictionary with shot information.

    Returns:
        str: The constructed directory path.
    """

    num_shots = shot_info["num_shots"]
    shot_source = shot_info["sources"]
    subdir = f"results/{model}/{demo}/{num_shots}_shot/{'_'.join(sorted(shot_source))}" 
    if num_shots == 0:
        subdir = f"results/{model}/{num_shots}_shot"
    return subdir


def get_output_path(
    source_domain: str, 
    target_domain: str, 
    num_shots: int, 
    subdir: str
) -> str:
    """
    Returns the output file path for results.

    Args:
        source_domain: Source domain name.
        target_domain: Target domain name.
        num_shots: Number of shots.
        subdir: Subdirectory path.

    Returns:
        str: The output file path.
    """
    filepath = os.path.join(subdir, f"results_{source_domain}_{target_domain}.json")
    if num_shots == 0:
        filepath = os.path.join(subdir, f"results_{target_domain}.json")
    return filepath


def get_response(
    prompt: str, 
    client, 
    model: str, 
    key_gemini: str | None = None
) -> str:
    """
    Gets a response from the specified LLM model.

    Args:
        prompt: The prompt to send.
        client: The client object for the model.
        model: The model name.
        key_gemini: API key for Gemini models.

    Returns:
        str: The model's response text.
    """
    model_map = {
        "gpt-4o": "gpt-4o-mini",
        "llama3": "llama3-70b-8192",
        "llama4_scout": "meta-llama/llama-4-scout-17b-16e-instruct",
        "deepseek_llama": "deepseek-r1-distill-llama-70b",
        "gemma": "gemma2-9b-it",
        "gemma3": "gemma-3-27b-it",
        "gemini_flash": "gemini-2.5-flash-preview-05-20",
        "qwen32": "qwen/qwen3-32b",
        "llama4_mav": "meta-llama/llama-4-maverick-17b-128e-instruct"
    }
    
    if model == 'gemma3' or model == 'gemini_flash':
        model = model_map.get(model, model)
        genai.configure(api_key=key_gemini)
        gemini = genai.GenerativeModel(model)
        response = gemini.generate_content(prompt)
        return response.text
    
    model = model_map.get(model, model)
    messages = [{"role": "user", "content": prompt}]
    output = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return output.choices[0].message.content


def enforce_rate_limit(
    request_times: 'Deque[float]', 
    MAX_REQUESTS_PER_MINUTE: int, 
    REQUEST_WINDOW: float
) -> None:
    """
    Enforces a rate limit for API requests.

    Args:
        request_times: Deque of request timestamps.
        MAX_REQUESTS_PER_MINUTE: Maximum allowed requests per window.
        REQUEST_WINDOW: Time window in seconds.
    """
    current_time = time.time()
    while request_times and current_time - request_times[0] > REQUEST_WINDOW:
        request_times.popleft()
    if len(request_times) >= MAX_REQUESTS_PER_MINUTE:
        sleep_time = REQUEST_WINDOW - (current_time - request_times[0]) + 0.1
        time.sleep(sleep_time)
        return enforce_rate_limit(request_times, MAX_REQUESTS_PER_MINUTE, REQUEST_WINDOW)
    request_times.append(current_time)


def load_txt_data(path: str) -> list[dict[str,str]]:
    """
    Loads data from a text file, used for the train and test sets

    Args:
        path: Path to the text file.

    Returns:
        List[Dict[str, str]]: List of data samples.
    """
    samples = []
    with open(path, "r", encoding="latin-1") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) % 3 == 0, "Data format error: lines must be multiples of 3"
   
    for i in range(0, len(lines), 3):
        template = lines[i]
        aspect = lines[i + 1]
        polarity_map = {"1": "Positive", "0": "Neutral", "-1": "Negative"}
        polarity = polarity_map.get(lines[i + 2], "Positive")
        sentence = template.replace("$T$", aspect)
        
        samples.append({
            "text": sentence,
            "template": template,
            "aspect": aspect,
            "polarity": polarity
        })
    return samples


def load_json_data(path: str) -> list[dict[str,str]]:
    """
    Loads data from a JSON file, used for the predictions.

    Args:
        path: Path to the JSON file.

    Returns:
        List[Dict[str, str]]: List of data samples.
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        text = entry.get("text", "{}")
        template = entry.get("template", "{}")
        aspect = entry.get("aspect", "{}")
        polarity = entry.get("polarity", "{}")
        paraphrased_text = entry.get("paraphrased_text", "{}") 

        samples.append({
            "text": text,
            "template": template,
            "aspect": aspect,
            "polarity": polarity,
            "paraphrased_text": paraphrased_text
        })
    return samples




def remove_entries(file_path: str, mask: list[int]) -> None:
    """
    Removes entries from a results JSON file at the indices specified in mask.

    Args:
        file_path: Path to the JSON file.
        mask: List of indices to remove.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
        results = existing_data.get("results", [])
        inference_prompts = existing_data.get("inference_prompts", [])
        metrics = existing_data.get("metrics", [])
        removed_val = False
        for i in mask:
            if i < len(results):
                results[i] = "{}"
                removed_val = True
            if i < len(inference_prompts):
                inference_prompts[i] = "{}"
        new_contents = {"results": results,
                        "inference_prompts": inference_prompts
                        }
        if not removed_val:
            new_contents["metrics"] = metrics
        
        with open(file_path, 'w') as f:
            json.dump(new_contents, f, indent=2)
        print(f"Removed {len(mask)} entries from {file_path}")


def generate_info(
    source_domains: list[str],
    target_domains: list[str],
    demos: list[str],
    models: list[str],
    shot_infos: list[dict],
    indices: list[int]
) -> list[tuple[str,str,str,str,dict]]:
    """
    Generates a list of information tuples based on the specified domains, models, 
    demos, and shot settings.

    Each tuple has the form:
    (source_domain, target_domain, demo_or_simcse, model, shot_info)

    If shot_info indicates zero shots, SimCSE is used and source == target.
    Otherwise, demos are used with different source and target domains.

    Args:
        source_domains: List of source domains.
        target_domains: List of target domains.
        demos: List of demo types.
        models: List of model names.
        shot_infos: List of shot info dicts.
        indices: Indices to use from shot_infos.

    Returns:
        List[Tuple[str, str, str, str, Dict[str, Any]]]: List of info tuples.
    """
    info = []

    # Handle zero-shot SimCSE cases
    for i in indices:
        shot_info = shot_infos[i]
        if shot_info['num_shots'] == 0:
            for model in models:
                for target in target_domains:
                    info.append((target, target, "SimCSE", model, shot_info))

    # Handle non-zero-shot demo cases
    for model in models:
        for demo in demos:
            for source in source_domains:
                for target in target_domains:
                    if source == target:
                        continue
                    for i in indices:
                        shot_info = shot_infos[i]
                        if shot_info['num_shots'] == 0:
                            continue
                        info.append((source, target, demo, model, shot_info))

    return info



def fix_paraphrased_text(entry: dict):
    """
    Fixes and normalizes the 'paraphrased_text' field in a data entry.

    Args:
        entry: The data entry.

    Returns:
        str: The fixed paraphrased text.
    """
    paraphrased = entry.get("paraphrased_text", "").strip()

    # Case 1: Extract all valid tuples and return the last one
    matches = re.findall(r'\(".*?",".*?",".*?"\)', paraphrased)
    if matches:
        return matches[-1]

    # Case 2 or 3: Split by lines and analyze
    lines = [line.strip().strip(',') for line in paraphrased.splitlines() if line.strip()]

    if len(lines) == 3:
        # Case 2: Three-line format
        sentence, aspect, polarity = lines
        return f"(\"{sentence}\",\"{aspect}\",\"{polarity.lower()}\")"
    
    elif len(lines) == 2:
        # Case 3: aspect and polarity on same line, separated by comma
        sentence = lines[0]
        if ',' in lines[1]:
            aspect_part, polarity_part = lines[1].split(',', 1)
            aspect = aspect_part.strip()
            polarity = polarity_part.strip().lower()
            return f"(\"{sentence}\",\"{aspect}\",\"{polarity}\")"

    # Case 4 or fallback: Do nothing
    return paraphrased


def process_json(data: list[dict]):
    """
    Processes a list of data entries, fixing their 'paraphrased_text' fields.

    Args:
        data: List of data entries.

    Returns:
        List[Dict[str, Any]]: The processed data.
    """
    for entry in data:
        entry["paraphrased_text"] = fix_paraphrased_text(entry)
    return data

def default_experiment_args() -> tuple[list[str], list[str], list[str], list[str], list[dict], list[int]]:
    """
    Returns the default experiment arguments, which is all of them

    Returns:
        Tuple: (source_domains, target_domains, demos, models, shot_infos, indices)
    """
    source_domains = ["laptop", "restaurant", "book"]
    target_domains = ["laptop", "restaurant", "book"]
    demos = ["SimCSE"]
    models = ["llama4_scout", "gemma"]
    indices = [0, 1, 2, 3, 4]
    shot_infos = [
        {"num_shots": 6, "sources": ["regular"]},
        {"num_shots": 6, "sources": ["paraphrased"]},
        {"num_shots": 3, "sources": ["paraphrased", "regular"]},
        {"num_shots": 3, "sources": ["independent", "dependent"]},
        {"num_shots": 0, "sources": []}
    ]
    return source_domains, target_domains, demos, models, shot_infos, indices

def parse_experiment_args() ->  tuple[list[str], list[str], list[str], list[str], list[dict], list[int]]:
    """
    Parses experiment arguments from the command line or a config file.

    Returns:
        Tuple: (source_domains, target_domains, demos, models, shot_infos, indices)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to a JSON config file")

    # Optional overrides, in case you want to change a specific argument
    # Comma-separated input expected
    parser.add_argument("--source_domains", type=str, help="e.g., 'restaurant,laptop'")
    parser.add_argument("--target_domains", type=str, help="e.g., 'restaurant,book'")
    parser.add_argument("--demos", type=str, help="e.g., 'SimCSE'")
    parser.add_argument("--models", type=str, help="e.g., 'gemma,llama4_scout'")
    parser.add_argument("--indices", type=str, help="e.g., '0,1,2'")
    parser.add_argument("--shot_infos_path", type=str)

    args = parser.parse_args()

    if len (sys.argv) == 1:
        print("No arguments passed, running all experiments")
        return default_experiment_args()

    config = {}
    if args.config:
        if not os.path.isfile(args.config):
            raise FileNotFoundError(f"Config file {args.config} not found.")
        with open(args.config, "r") as f:
            config = json.load(f)
 
        

    def get(key, default=None, split=True, cast=str):
        val = getattr(args, key)
        if val is not None:
            if split:
                return [cast(x.strip()) for x in val.split(",")]
            else:
                return cast(val)
        
        if key in config:
            conf_val = config[key]
            if isinstance(conf_val, list):
                return [cast(x) for x in conf_val]
            if split:
                return [cast(x.strip()) for x in conf_val.split(",")]
            else:
                return cast(conf_val)
        
        return default

    source_domains = get("source_domains", [], cast=str)
    target_domains = get("target_domains", [], cast=str)
    demos = get("demos", ["SimCSE"], cast=str)
    models = get("models", [], cast=str)
    indices = get("indices", [], cast=int)

    shot_infos_path = get("shot_infos_path", default=None, split=False, cast=str)
    if not shot_infos_path:
        raise ValueError("Missing required `shot_infos_path`.")
    if not os.path.isfile(shot_infos_path):
        raise FileNotFoundError(f"Shot info file {shot_infos_path} not found.")

    with open(shot_infos_path, "r") as f:
        shot_infos = json.load(f)

    return source_domains, target_domains, demos, models, shot_infos, indices


def fix_missing_spaces(text: str) -> str:
    """
    Adds a space between a letter and a digit if not already present.

    Args:
        text: The input string.

    Returns:
        The string with spaces inserted where needed.
    """
    return re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', text)



def normalize(text: str) -> str:
    """
    Normalizes and cleans a string for comparison.

    Args:
        text: The input string.

    Returns:
        The normalized string.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).lower().strip()  # Unicode normalization, lowercase, and strip whitespace
    text = re.sub(r"'s\b", "", text)                           # Remove possessive 's
    text = re.sub(r"[\"'`‘’“”]", "", text)                     # Remove all types of quotes
    text = re.sub(r"\s+", " ", text)                           # Replace multiple spaces with a single space
    text = fix_missing_spaces(text)                            # Add space between letter and digit if missing
    return text


if __name__ == "__main__":
    # Example usage if the llms misbehave with the output for the transformations, only needed when you get errors from output
    domains = ["laptop", "book", "restaurant"]
    for domain in domains:
        with open(f"cache/gemma/dependent/train_data_{domain}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        fixed_data = process_json(data)
        with open(f"cache/gemma/dependent/train_data_{domain}.json", "w", encoding="utf-8") as f:
            json.dump(fixed_data, f, indent=2, ensure_ascii=False)


        with open(f"cache/gemma/independent/train_data_{domain}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        fixed_data = process_json(data)
        with open(f"cache/gemma/independent/train_data_{domain}.json", "w", encoding="utf-8") as f:
            json.dump(fixed_data, f, indent=2, ensure_ascii=False)
