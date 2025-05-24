import json
import xml.etree.ElementTree as ET
import numpy as np
import torch
import os

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline, AutoModel, AutoTokenizer
from collections import defaultdict

load_dotenv()


# Fill these variables yourself
key_openai = os.getenv("OPENAI_API_KEY")  # OpenAI API key
key_deepinfra = ''
train_file_path = ""
test_file_path = ""

use_SimCSE = True  # if true, use SimCSE, otherwise use BM25 for demonstration selection
model_choice = "gpt-4o" # options: "llama3"
num_shots = 3  # number of demonstrations to use

# Load data
train_root = ET.parse(train_file_path).getroot()
test_root = ET.parse(test_file_path).getroot()

train_sentences = train_root.findall(".//sentence")
test_sentences = test_root.findall(".//sentence")
train_corpus = [sentence.find("text").text for sentence in train_sentences]

openai_client_openai = OpenAI(api_key= key_openai)
openai_client_deepinfra = OpenAI(api_key= key_deepinfra, base_url = "https://api.deepinfra.com/v1/openai")

def get_response(prompt, client, model):
    if model == "gpt-3.5":
        model = "gpt-3.5-turbo"
    elif model == "gpt-4o":
        model = "gpt-4o-mini"
    elif model == "llama3":
        model = "meta-llama/Meta-Llama-3-70B-Instruct"
    elif model == "llama4":
        model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    else:
        raise ValueError("Unsupported model type. Please choose from 'gpt-3.5', 'gpt-4o', 'llama3', or 'llama4'.")

    messages = [{"role": "user", "content": prompt}]
    output = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return output.choices[0].message.content


# BM25
def BM25_demonstration_selection(query_sentence, corpus, k):
    bm25 = BM25Okapi([s.lower().split() for s in corpus])
    scores = bm25.get_scores( query_sentence.lower().split())
    return np.argsort(scores)[::-1][:k]

# SimCSE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
SimCSE_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(device)
SimCSE_model.eval()

def precompute_train_embeddings(corpus, tokenizer):
    inputs = tokenizer(corpus, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        return SimCSE_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu()

train_embeddings = precompute_train_embeddings(train_corpus, tokenizer, SimCSE_model)


def SimCSE_demonstration_selection(query_sentence, embeddings, k):
    inputs = tokenizer(query_sentence, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        query_embedding = SimCSE_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu()
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    return np.argsort(similarities)[-k:][::-1]


# Base prompt
base_prompt = """
Instruction:

Please perform the Aspect-Based Sentiment Classification task. Given an aspect in a sentence, assign a sentiment label from ['positive', 'neutral', 'negative'].


Demonstrations:
{demonstrations}

Tested sample:
- Sentence: {sentence}
- Aspects: {aspects}

Output:
Generate the answer in a compact JSON format with no newlines or indentation, containing the following fields:
- {aspects} - string that is one of the polarities ("Positive", "Negative", "Neutral")

Always respond with a valid JSON. Do not invlude any extra characters, symbols, or text in or outside the JSON itself (including backticks, ", /)

"""


# Evaluation metrics
def evaluation(test_sentences, results):

    ground_truth, filtered_ground_truth, filtered_preds = [], [], []
    for sentence in test_sentences:
        opinions = sentence.findall(".//Opinion")
        sentence_truth = {opinion.get("target"): opinion.get("polarity").capitalize() for opinion in opinions}
        ground_truth.append(sentence_truth)


    # Filter out invalid JSON, aspect mismatches
    for truth, predicted in zip(ground_truth, results):

        # Skip if JSON is invalid
        try:
            predicted = json.loads(predicted)
        except json.JSONDecodeError:
            continue

        if list(truth.keys()) == list(pred_dict.keys()):
            filtered_ground_truth.append(truth)
            filtered_preds.append(pred_dict)

    y_true, y_pred = [], []
    for truth, predicted in zip(filtered_ground_truth, filtered_preds):

        for aspect, true_polarity in truth.items():
            predicted_polarity = predicted.get(aspect).capitalize()
            y_true.append(true_polarity.capitalize())
            y_pred.append(predicted_polarity)

    labels = ["Positive", "Negative", "Neutral"]

    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "f1": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0) * 100,
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0) * 100,
    }

results = []

# Main
for sentence in test_sentences:
    sentence_text = sentence.find("text").text
    opinions = sentence.findall(".//Opinion")
    aspects = [opinion.get("target") for opinion in opinions] # Currently I don't do anything with this
   

    if use_SimCSE:
        top_indices = SimCSE_demonstration_selection(sentence_text, train_corpus, train_embeddings, num_shots)
    else:
        top_indices = BM25_demonstration_selection(sentence_text, train_corpus, num_shots)

    demonstrations = []
    for idx in top_indices:
        demo = train_sentences[idx]
        demo_text = demo.find("text").text
        demo_opinions = demo.findall(".//Opinion")
        pair_str = ", ".join(f"{opinion.get('target')} ({opinion.get('polarity')})" for opinion in demo_opinions)
        demonstrations.append(f"Sentence: {demo_text}\nAspects: {pair_str}")
    demonstrations = "\n\n".join(demonstrations)

    # Preparing prompt
    prompt = base_prompt.format(
        demonstrations=demonstrations,
        sentence=sentence_text,
        aspects=", ".join(aspects) if aspects else "None"
    )
    print(prompt)

    try:
        output = get_response(prompt, openai_client_openai, model_choice)
        results.append(output)
    except Exception as e:
        print(f"Error generating response: {e}")
        results.append("{}")  # fallback to empty prediction

metrics = evaluation(test_sentences, results)
print(json.dumps(metrics, indent=2))