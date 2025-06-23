import json
import re

input_path = "results/qwen32/SimCSE/6_shot/regular/results_restaurant_laptop.json"
output_path = "results/qwen32/SimCSE/6_shot/regular/processed_results_restaurant_laptop.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)




def extract_last_json(text):
    """Extracts the last JSON object from the text string."""
    matches = re.findall(r'\{.*?\}', text, flags=re.DOTALL)
    for match in reversed(matches):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    return None  # fallback if nothing valid is found

def process_json(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_results = data.get("results", [])
    cleaned_results = []

    for i, entry in enumerate(raw_results):
        parsed = extract_last_json(entry)
        if parsed is not None:
            cleaned_results.append(json.dumps(parsed))
        else:
            print(f"Warning: Could not extract JSON from entry {i}")
            cleaned_results.append(json.dumps({}))  # fallback if parsing fails


    new_data = {
        "results": cleaned_results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2)

    print(f"Processed {len(cleaned_results)} results written to '{output_path}'.")

if __name__ == "__main__":
    process_json(input_path, output_path)
