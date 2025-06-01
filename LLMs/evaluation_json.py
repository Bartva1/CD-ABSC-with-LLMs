import json
import os
from collections import Counter, defaultdict


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

LABELS = ["Positive", "Negative", "Neutral"]

# data loading
def load_txt_data(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
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


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data.get("results", [])

    if all(isinstance(entry, str) for entry in predictions):
        parsed = []
        for entry in predictions:
            try:
                parsed.append(json.loads(entry))
            except json.JSONDecodeError:
                parsed.append({})  
        return parsed
    elif all(isinstance(entry, dict) for entry in predictions):
        return predictions
    else:
        raise ValueError("Unknown prediction format in predictions")


def evaluate(test_data, results):
    y_true, y_pred = [], []


    for sample, pred in zip(test_data, results):
        
        aspect = sample["aspect"]
        true_polarity = sample["polarity"].capitalize()
        predicted_polarity = pred.get(aspect, "").capitalize()

        if predicted_polarity:
            y_true.append(true_polarity)
            y_pred.append(predicted_polarity)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0) * 100,
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0) * 100,
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0) * 100
    }

    correct_per_class = defaultdict(int)
    total_per_class = Counter(y_true)

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct_per_class[true] += 1

    per_class_accuracy = {
        label: (correct_per_class[label] / total_per_class[label]) * 100 if total_per_class[label] > 0 else 0.0
        for label in LABELS
    }

    metrics["per_class_accuracy"] = per_class_accuracy

    return metrics



def aggregate_predictions_to_df(test_data, results):
    rows = []
    for i, (sample, prediction) in enumerate(zip(test_data, results)):
        aspect = sample["aspect"]
        true_label = sample["polarity"].capitalize()
        predicted_label = prediction.get(aspect, "").capitalize()

        rows.append({
            "index": i,
            "text": sample["text"],
            "template": sample["template"],
            "aspect": aspect,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "correct": true_label == predicted_label
        })

    return pd.DataFrame(rows)

def analyze_predictions(df, k=5):
    summary = {}
    for label in df["true_label"].unique():
        cls_df = df[df["true_label"] == label]
        summary[label] = {
            "best": cls_df[cls_df["correct"]].head(k),
            "worst": cls_df[~cls_df["correct"]].head(k)
        }
    return summary

def visualize_misclassified(df, class_name=None, k=10):
    df = df[~df["correct"]]
    if class_name:
        df = df[df["true_label"] == class_name]

    print(df.head(k)[["text", "aspect", "true_label", "predicted_label"]])

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="true_label", hue="predicted_label", palette="Set2")
    plt.title(f"Misclassifications{f' for {class_name}' if class_name else ''}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def evaluate_multiple_predictions(txt_path, json_paths, k=5):
    test_data = load_txt_data(txt_path)
    all_data_frames = []

    for json_path in json_paths:
        print(f"Evaluating: {json_path}")
        predictions = load_predictions(json_path)
        if len(predictions) != len(test_data):
            print(f"Skipping due to length mismatch: {json_path}")
            continue

        df = aggregate_predictions_to_df(test_data, predictions)
        df["source_file"] = os.path.basename(json_path)
        all_data_frames.append(df)

    if not all_data_frames:
        print("No valid predictions.")
        return

    full_df = pd.concat(all_data_frames, ignore_index=True)
    summary = analyze_predictions(full_df, k)

    for cls, data in summary.items():
        print(f"\n=== Class: {cls} ===")
        print("Top-K Best:")
        print(data["best"][["source_file", "text", "aspect", "true_label", "predicted_label"]])
        print("Top-K Worst:")
        print(data["worst"][["source_file", "text", "aspect", "true_label", "predicted_label"]])

    visualize_misclassified(full_df)
    return full_df

if __name__ == "__main__":
    test_info = [("restaurant", "book", "SimCSE", "gemma", ""),
                 ("laptop", "book", "SimCSE", "gemma", ""),
                 ("restaurant", "book", "SimCSE", "llama3", ""),
                 ("laptop", "book", "SimCSE", "llama3", "_use_transformation")]
  

    domain_to_eval_data = defaultdict(lambda: {"ground_truth": "", "json_paths": []})
    result_paths = []
    for info_tuple in test_info:
        train_domain = info_tuple[0]
        test_domain = info_tuple[1]
        demo = info_tuple[2]
        model = info_tuple[3]
        use_transformation = info_tuple[4]
        shots = 3

        year = 2019 if test_domain == "book" else 2014
        path_pred = f"results/{model}/{demo}/{shots}_shot/results_{model}_{train_domain}_{test_domain}_{shots}_shot{use_transformation}.json"
        path_ground_truth = f"data_out/{test_domain}/raw_data_{test_domain}_test_{year}.txt"
        
        domain_to_eval_data[test_domain]["ground_truth"] = path_ground_truth
        domain_to_eval_data[test_domain]["json_paths"].append(path_pred)


        print(f"Source domain is: {train_domain}, target domain is: {test_domain} using model: {model}")
        test_data = load_txt_data(path_ground_truth)
        pred_data = load_predictions(path_pred)

        print(json.dumps(evaluate(test_data, pred_data), indent=2))

    for test_domain, paths in domain_to_eval_data.items():
        print(f"\n Evaluating predictions for test domain: {test_domain}")
        print(evaluate_multiple_predictions(paths["ground_truth"], paths["json_paths"], k=5).head())