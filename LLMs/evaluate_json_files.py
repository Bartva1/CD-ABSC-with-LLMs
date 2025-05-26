import json
from sklearn.metrics import accuracy_score, f1_score

from classification import load_txt_data, evaluation

path_file_predictions = "results/bm25/processed_results_deepseek_llama.json" # replace this with the json file you want to evaluate
test_file_path = "data_out/laptop/raw_data_laptop_test_2014.txt" # replace this with the test file you used in testing
with open(path_file_predictions, "r", encoding="utf-8") as file:
    data_raw = json.load(file)
    predictions = data_raw["results"]
    inference_prompts = data_raw["inference_prompts"]
    


results = []
print("hi1")
for i, entry in enumerate(predictions):
    try: 
        results.append(json.loads(entry))
    except json.JSONDecodeError:
        print(f"Failed to parse prediction at index {i}")
        results.append({})
    
print("hi2")
test_data = load_txt_data(test_file_path)
metrics = evaluation(test_data, predictions)


print("hi3")

print("\n\nInference Prompts and Responses:")
for i, prompt in enumerate(inference_prompts):
    print(f"Prompt {i + 1}:\n{prompt}\n")
    print(f"Response {i + 1}:\n{results[i]}\n")

print(json.dumps(metrics, indent=2))



with open(path_file_predictions, "w") as f:
    json.dump({
        "metrics": metrics,
        "results": results,
        "inference_prompts": inference_prompts 
    }, f, indent=2)