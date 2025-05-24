import json
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
from tqdm import tqdm
from groq import Groq 

load_dotenv()


# Fill these variables yourself
key_openai = os.getenv("OPENAI_API_KEY") 
key_groq = os.getenv("GROQ_API_KEY")  

key_deepinfra = ''
train_file_path = r"data_out\restaurant\raw_data_restaurant_train_2014.txt"
test_file_path = r"data_out\laptop\raw_data_laptop_test_2014.txt"

use_SimCSE = False  # if true, use SimCSE, otherwise use BM25 for demonstration selection
model_choice = "llama3" # options: "llama3"
num_shots = 3  # number of demonstrations to use

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

train_data = load_txt_data(train_file_path)
test_data = load_txt_data(test_file_path)
train_corpus = [sample["text"] for sample in train_data]

openai_client_openai = OpenAI(api_key= key_openai)
groq_client = Groq(api_key=key_groq)
openai_client_deepinfra = OpenAI(api_key= key_deepinfra, base_url = "https://api.deepinfra.com/v1/openai")

client = groq_client

def get_response(prompt, client, model):
    if model == "gpt-3.5":
        model = "gpt-3.5-turbo"
    elif model == "gpt-4o":
        model = "gpt-4o-mini"
    elif model == "llama3":
        model = "llama3-70b-8192"
        # model = "meta-llama/Meta-Llama-3-70B-Instruct"
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
if use_SimCSE:
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    SimCSE_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(device)
    SimCSE_model.eval()

def precompute_train_embeddings(corpus):
    inputs = tokenizer(corpus, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        return SimCSE_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu()

if use_SimCSE:
    train_embeddings = precompute_train_embeddings(train_corpus)


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

    y_true, y_pred = [], []

    for sample, prediction in zip(test_data, results):
        try:
            pred = json.loads(prediction)
        except json.JSONDecodeError:
            continue

        aspect = sample["aspect"]
        true_polarity = sample["polarity"]
        predicted_polarity = pred.get(aspect, "").capitalize()
        if predicted_polarity:
            y_true.append(true_polarity)
            y_pred.append(predicted_polarity)
    labels = ["Positive", "Negative", "Neutral"]
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "f1": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0) * 100,
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0) * 100,
    }

results = []

# Main
for sample in tqdm(test_data):
    sentence_text = sample["text"]
    template = sample["template"]
    aspects = sample["aspect"]
   

    if use_SimCSE:
        top_indices = SimCSE_demonstration_selection(sentence_text, train_embeddings, num_shots)
    else:
        top_indices = BM25_demonstration_selection(sentence_text, train_corpus, num_shots)

    demonstrations = []
    for idx in top_indices:
        demo = train_data[idx]
        demonstrations.append(f"Sentence: {demo['template']}\nAspects: {demo['aspect']} ({demo['polarity']})")
    demonstrations = "\n\n".join(demonstrations)


    # Preparing prompt
    prompt = base_prompt.format(
        demonstrations=demonstrations,
        sentence=sentence_text,
        aspects=aspects
    )
    # print(prompt)

    try:
        output = get_response(prompt, client, model_choice)
        results.append(output)
    except Exception as e:
        print(f"Error generating response: {e}")
        results.append("{}")  

metrics = evaluation(test_data, results)
print(json.dumps(metrics, indent=2))