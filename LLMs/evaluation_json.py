import json
import os
from collections import Counter, defaultdict
import unicodedata
import re


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import load_txt_data, get_directory, get_output_path

LABELS = ["Positive", "Negative", "Neutral"]



def load_predictions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = data.get("results", [])

  
    if all(isinstance(x, dict) for x in preds):
        return preds

    if all(isinstance(x, str) for x in preds):
        out = []
        for s in preds:
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError:
                m = re.match(r'^\s*\{\s*"(.+?)"\s*:\s*(.+)\s*\}\s*$', s)

                if m:
                    key, val = m.groups()
                    key_fixed = key.replace('"', '\\"')
                    fixed = f'{{"{key_fixed}":{val}}}'
                    try:
                        out.append(json.loads(fixed))
                        continue
                    except json.JSONDecodeError:
                        pass
                out.append({})
        return out
    raise ValueError("Unknown prediction format in 'results'")



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


def normalize(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"[\"'`‘’“”]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def evaluate_multiple_predictions(txt_path, json_paths, key_info, k=5):
    test_data = load_txt_data(txt_path)

    
    base_df = pd.DataFrame({
        "text": [s["text"] for s in test_data],
        "aspect": [s["aspect"] for s in test_data],
        "true_label": [s["polarity"].capitalize() for s in test_data],
    })

    for json_path in json_paths:
        print(f"Evaluating: {json_path}")
        predictions = load_predictions(json_path)
        if len(predictions) != len(test_data):
            print(f"Skipping {json_path} due to length mismatch")
            continue
        
        # Track invalid predictions
        invalid_preds = []

        normalized_predictions = []
        for prediction in predictions:
            normalized_prediction = {
                normalize(key): value.capitalize() for key, value in prediction.items()
            }
            normalized_predictions.append(normalized_prediction)

        for i, (sample, prediction) in enumerate(zip(test_data, normalized_predictions)):
            aspect = normalize(sample["aspect"])
            predicted_label = prediction.get(aspect, "")
            
            if predicted_label not in LABELS:
                invalid_preds.append({
                    "index": i,
                    "aspect": aspect,
                    "prediction": prediction
                })

        if invalid_preds:
            print(f"\n[!] {len(invalid_preds)} invalid predictions found in {json_path}:")
            for entry in invalid_preds:
                print(f"  - Sample #{entry['index']}: aspect='{entry['aspect']}', "
                      f"Prediction: {entry['prediction']}")

        
        idx = json_paths.index(json_path)
        model, train_domain, transformation_text = key_info[idx]
        model_id = f"{model}_{train_domain}_{transformation_text}" 
      
        preds = [
            pred.get(normalize(sample["aspect"]), "").capitalize()
            for sample, pred in zip(test_data, normalized_predictions)
        ]
        base_df[model_id] = preds

   
    model_cols = [col for col in base_df.columns if col not in ("text", "aspect", "true_label")]
    base_df["num_wrong"] = base_df.apply(
        lambda row: sum(row[col] != row["true_label"] for col in model_cols), axis=1
    )

    n_models = len(model_cols)
    base_df["majority_wrong"] = base_df["num_wrong"] > n_models // 2

    # # examples which are misclassified the most often by the selected models
    # worst = base_df[base_df["majority_wrong"]].sort_values("num_wrong", ascending=False).head(k)
    # print("\nMost frequently misclassified examples:")
    # print(worst[["text", "aspect", "true_label", "num_wrong"] + model_cols])

    # confusion matrix heatmap
    wrong = base_df[base_df["majority_wrong"]]
    melted = wrong.melt(id_vars=["text", "aspect", "true_label"], value_vars=model_cols,
                        var_name="model", value_name="predicted")
    melted["correct"] = melted["true_label"] == melted["predicted"]

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pd.crosstab(melted["true_label"], melted["predicted"]),
        annot=True, fmt="d", cmap="Reds"
    )
    plt.title("Confusion Matrix Across Misclassified Examples")
    plt.show()

    return base_df

if __name__ == "__main__":
    # source, target, use_SimCSE, model, use_transformation, shots
    test_info = [("laptop", "book", True, "llama3", False, 3)]
  

    domain_to_eval_data = defaultdict(lambda: {"ground_truth": "", "json_paths": [], "key_info": []})
    result_paths = []
    for train_domain, test_domain, use_SimCSE, model, use_transformation, num_shots in test_info:
        year = 2019 if test_domain == "book" else 2014
        transformation_text = "with_transformation" if use_transformation else "no_transformation"
        path_ground_truth = f"data_out/{test_domain}/raw_data_{test_domain}_test_{year}.txt"

        subdir = get_directory(use_SimCSE=use_SimCSE, model=model, use_transformation=use_transformation, num_shots=num_shots)
        path_pred = get_output_path(source_domain=train_domain, target_domain=test_domain, model=model, use_transformation=use_transformation, num_shots=num_shots, subdir=subdir)
   
        domain_to_eval_data[test_domain]["ground_truth"] = path_ground_truth
        domain_to_eval_data[test_domain]["json_paths"].append(path_pred)
        domain_to_eval_data[test_domain]["key_info"].append((model, train_domain, transformation_text))


        test_data = load_txt_data(path_ground_truth)
        pred_data = load_predictions(path_pred)
        print(f"\nSource domain: {train_domain}, target: {test_domain}, model: {model}, transformation: {use_transformation}")
        print(json.dumps(evaluate(test_data, pred_data), indent=2))

    for test_domain, paths in domain_to_eval_data.items():
        print(f"\n Evaluating predictions for test domain: {test_domain}")
        df = evaluate_multiple_predictions(paths["ground_truth"], paths["json_paths"], paths["key_info"], k=5)
    
        