import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter, defaultdict
import os


def load_txt_data(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    assert len(lines) % 3 == 0, "Data format error: lines must be multiples of 3"
    for i in range(0, len(lines), 3):
        template = lines[i]
        aspect = lines[i + 1]
        polarity = "Positive"
        if lines[i + 2] == "0":
            polarity = "Neutral"
        elif lines[i + 2] == "-1":
            polarity = "Negative"

        sentence = template.replace("$T$", aspect)
        samples.append({
            "text": sentence,
            "template": template,
            "aspect": aspect,
            "polarity": polarity
        })
    return samples


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data.get("results", [])

    
    if all(isinstance(entry, str) for entry in predictions):
        parsed_results = []
        for entry in predictions:
            try:
                parsed = json.loads(entry)
                parsed_results.append(parsed)
            except json.JSONDecodeError:
                parsed_results.append({})  
        return parsed_results
    elif all(isinstance(entry, dict) for entry in predictions):
        return predictions
    else:
        raise ValueError("Unknown prediction format in JSON file")


def evaluate(test_data, results):
    y_true, y_pred = [], []
    labels = ["Positive", "Negative", "Neutral"]

    for sample, prediction in zip(test_data, results):
        
        aspect = sample["aspect"]
        true_polarity = sample["polarity"].capitalize()
        predicted_polarity = prediction.get(aspect, "").capitalize()

        if predicted_polarity:
            y_true.append(true_polarity)
            y_pred.append(predicted_polarity)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0) * 100,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0) * 100,
        "micro_f1": f1_score(y_true, y_pred, average="micro", labels=labels, zero_division=0) * 100,
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0) * 100,
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0) * 100,
        "micro_precision": precision_score(y_true, y_pred, average="micro", labels=labels, zero_division=0) * 100,
        "micro_recall": recall_score(y_true, y_pred, average="micro", labels=labels, zero_division=0) * 100,
    }

    correct_per_class = defaultdict(int)
    total_per_class = Counter(y_true)

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct_per_class[true] += 1

    per_class_accuracy = {
        label: (correct_per_class[label] / total_per_class[label]) * 100 if total_per_class[label] > 0 else 0.0
        for label in labels
    }

    metrics["per_class_accuracy"] = per_class_accuracy
    return metrics


if __name__ == "__main__":
    useSimCSE = False
    model = "gemma"
    print(f"Model is: {model}")
    is_reasoning = model == "deepseek_llama" or model == "qwen32"
    processed = "processed_" if is_reasoning else ""
    demo = "SimCSE" if useSimCSE else "bm25"
    domain_pairs = [("book_laptop", 3)]
    for domain_pair, shots in domain_pairs:
        test_domain = domain_pair.split("_")[1]
        year = 2019 if test_domain == "book" else 2014
        path_pred = f"results/{model}/{demo}/{shots}_shot/{processed}results_{model}_{domain_pair}_{shots}_shot.json"
        path_ground_truth = f"data_out/{test_domain}/raw_data_{test_domain}_test_{year}.txt"

    
        print(f"Domain pair is: {domain_pair} using {shots} shots")
        test_data = load_txt_data(path_ground_truth)
        pred_data = load_predictions(path_pred)

        print(json.dumps(evaluate(test_data, pred_data), indent=2))
