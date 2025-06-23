import json
import os
from collections import Counter, defaultdict
import unicodedata
import re


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import load_txt_data, get_directory, get_output_path, generate_info, parse_experiment_args

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
                # need to still explain this
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

def compute_performance_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0) * 100,
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0) * 100,
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0) * 100,
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


def plot_confusion_heatmap(base_df, model_cols):
    base_df["num_wrong"] = base_df.apply(
        lambda row: sum(row[col] != row["true_label"] for col in model_cols), axis=1
    )
    n_models = len(model_cols)
    base_df["majority_wrong"] = base_df["num_wrong"] > n_models // 2

    wrong = base_df[base_df["majority_wrong"]].copy()

   
    def most_common_wrong_prediction(row):
        preds = [row[col] for col in model_cols if row[col] != row["true_label"]]
        if preds:
            return Counter(preds).most_common(1)[0][0]
        return None

    wrong["chosen_predicted"] = wrong.apply(most_common_wrong_prediction, axis=1)
    wrong = wrong.dropna(subset=["chosen_predicted"])

    conf_matrix = pd.crosstab(wrong["true_label"], wrong["chosen_predicted"])

    
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds")
    plt.title("Confusion Matrix (One Representative Error per Misclassified Sample)")
    plt.show()



    
def evaluate_multiple_predictions(txt_path, json_paths, key_info):
    test_data = load_txt_data(txt_path)

    base_df = pd.DataFrame({
        "text": [s["text"] for s in test_data],
        "aspect": [s["aspect"] for s in test_data],
        "true_label": [s["polarity"].capitalize() for s in test_data],
    })

    model_metrics = {}

    for idx, json_path in enumerate(json_paths):
        y_true, y_pred = [], []
        print(f"Evaluating: {json_path}")
        predictions = load_predictions(json_path)
        if len(predictions) != len(test_data):
            print(f"Skipping {json_path} due to length mismatch")
            continue

        invalid_preds = []
        normalized_predictions = [
            {normalize(key): value.capitalize() for key, value in pred.items()}
            for pred in predictions
        ]

        for i, (sample, prediction) in enumerate(zip(test_data, normalized_predictions)):
            aspect = normalize(sample["aspect"])
            predicted_label = prediction.get(aspect, "")
            if predicted_label not in LABELS:
                invalid_preds.append({
                    "index": i,
                    "aspect": aspect,
                    "prediction": prediction
                })
            else:
                y_true.append(normalize(sample["polarity"]).capitalize())
                y_pred.append(normalize(predicted_label).capitalize())

        model, train_domain, demo_method, transformation_text = key_info[idx]
        model_id = f"{model}_{train_domain}_{demo_method}_{transformation_text}"

        preds = [
            pred.get(normalize(sample["aspect"]), "").capitalize()
            for sample, pred in zip(test_data, normalized_predictions)
        ]
        base_df[model_id] = preds

        if not invalid_preds:
            metrics = compute_performance_metrics(y_true, y_pred)
            model_metrics[model_id] = metrics
        else:
            print(f"\n[!] {len(invalid_preds)} invalid predictions found in {json_path}:")
            for entry in invalid_preds:
                print(f"  - Sample #{entry['index']}: aspect='{entry['aspect']}', "
                      f"Prediction: {entry['prediction']}")
    
    model_cols = [col for col in base_df.columns if col not in ("text", "aspect", "true_label")]

  
    
    base_df["num_wrong"] = base_df.apply(
        lambda row: sum(row[col] != row["true_label"] for col in model_cols), axis=1
    )
    # uncomment code below for extra check whether prediction are in correct format + overall heatmap
    # plot_confusion_heatmap(base_df, model_cols)

    return base_df, model_metrics


def print_metric_tables(all_metrics):
    metric_names = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    header_fmt = "{:<50}" + "".join(["{:>15}" for _ in metric_names])
    row_fmt = "{:<50}" + "".join(["{:>15.2f}" for _ in metric_names])

    grouped = defaultdict(lambda: defaultdict(list)) 


    for (source, test_domain), model_dict in all_metrics.items():
        for model_id, metrics in model_dict.items():
            parts = model_id.split("_")
            model = parts[0]
            train_domain = parts[1]
            demo_method = parts[2]
            shot_info = "_".join(parts[3:])
            grouped[test_domain][model].append((train_domain,demo_method, shot_info, model_id, metrics))

    # Print results
    for test_domain in sorted(grouped):
        print(f"\n{'=' * 80}")
        print(f"Evaluation Results for Test Domain: {test_domain.upper()}")
        print(f"{'=' * 80}")

        for model in sorted(grouped[test_domain]):
            print(f"\n--- Model: {model} ---")
            rows = grouped[test_domain][model]

            print(header_fmt.format("Model ID", *metric_names))
            print("-" * (60 + 15 * len(metric_names)))

            for train_domain, demo_method, shot_info, model_id, metrics in sorted(rows):
                label = f"({train_domain}, {demo_method}, {shot_info})"
                values = [metrics.get(m, 0.0) for m in metric_names]
                print(row_fmt.format(label, *values))

            class_labels = LABELS 
            print("\nPer-Class Accuracy:")
            header_fmt_class = "{:<50}" + "".join(["{:>10}" for _ in class_labels])
            row_fmt_class = "{:<50}" + "".join(["{:>10.2f}" for _ in class_labels])

            print(header_fmt_class.format("Model ID", *class_labels))
            print("-" * (60 + 15 * len(class_labels)))

            for train_domain, demo_method, shot_info, model_id, metrics in sorted(rows):
                label = f"({train_domain}, {demo_method}, {shot_info})"
                acc_values = [metrics.get("per_class_accuracy", {}).get(label, 0.0) for label in class_labels]
                print(row_fmt_class.format(label, *acc_values))


if __name__ == "__main__":
    
    source_domains, target_domains, demos, models, shot_infos, indices = parse_experiment_args()
    
    test_info = generate_info(
            source_domains=source_domains,
            target_domains=target_domains,
            demos=demos,
            models=models,
            shot_infos=shot_infos,
            indices=indices
        )

    domain_to_eval_data = defaultdict(lambda: {"ground_truth": "", "json_paths": [], "key_info": []})

    for train_domain, test_domain, demo_method, model, shot_info in test_info:
        year = 2019 if test_domain == "book" else 2014
        path_ground_truth = f"data_out/{test_domain}/raw_data_{test_domain}_test_{year}.txt"
        subdir = get_directory(demo=demo_method, model=model, shot_info=shot_info)
        path_pred = get_output_path(source_domain=train_domain, target_domain=test_domain, num_shots=shot_info['num_shots'], subdir=subdir)

        domain_to_eval_data[test_domain]["ground_truth"] = path_ground_truth
        domain_to_eval_data[test_domain]["json_paths"].append(path_pred)
        domain_to_eval_data[test_domain]["key_info"].append(
        (model, train_domain, demo_method, "_".join(shot_info["sources"]))
        )


    all_metrics = {}

    for test_domain, paths in domain_to_eval_data.items():
        print(f"\nEvaluating predictions for test domain: {test_domain}")
        df, model_metrics = evaluate_multiple_predictions(paths["ground_truth"], paths["json_paths"], paths["key_info"])

        for model_id, metrics in model_metrics.items():
            source = model_id.split("_")[1]
            all_metrics.setdefault((source, test_domain), {})[model_id] = metrics

    print_metric_tables(all_metrics)
        