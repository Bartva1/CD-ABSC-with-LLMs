import os
import time
import json
import numpy as np

def get_directory(use_SimCSE: bool, model: str, use_transformation: bool, num_shots: int) -> str:
    demo_text = "SimCSE" if use_SimCSE else "bm25"
    transformation_dir_text = "test_transformation" if use_transformation else "no_test_transformation"
    subdir = f"results/{model}/{demo_text}/{num_shots}_shot/{transformation_dir_text}" 
    if num_shots == 0:
        subdir = f"results/{model}/{num_shots}_shot/{transformation_dir_text}"
    return subdir

def get_output_path(source_domain: str, target_domain: str, model: str, use_transformation: bool, num_shots: int, subdir: str) -> str:
    transformation_text = "_use_transformation" if use_transformation else ""
    filepath = os.path.join(subdir, f"results_{model}_{source_domain}_{target_domain}_{num_shots}_shot{transformation_text}.json")
    if num_shots == 0:
        filepath = os.path.join(subdir, f"results_{model}_{target_domain}_{num_shots}_shot{transformation_text}.json")
    return filepath



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


def enforce_rate_limit(request_times, MAX_REQUESTS_PER_MINUTE, REQUEST_WINDOW):
    current_time = time.time()
    while request_times and current_time - request_times[0] > REQUEST_WINDOW:
        request_times.popleft()
    if len(request_times) >= MAX_REQUESTS_PER_MINUTE:
        sleep_time = REQUEST_WINDOW - (current_time - request_times[0]) + 0.1
        time.sleep(sleep_time)
        return enforce_rate_limit(request_times, MAX_REQUESTS_PER_MINUTE, REQUEST_WINDOW)
    request_times.append(current_time)

    

def load_txt_data(path: str):
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

if __name__ == "__main__":
    subdir = get_directory(use_SimCSE=True, model="gemma", use_transformation=False, num_shots=0)
    file_path = get_output_path(source_domain="laptop", target_domain="restaurant", model="gemma", use_transformation=False, num_shots=0, subdir=subdir)
    mask = np.arange(0,638)
    remove_entries(file_path=file_path, mask=mask)