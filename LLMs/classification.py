import json
import numpy as np
import torch
import os
import time


from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize
from collections import defaultdict, deque
from tqdm import tqdm
from groq import Groq 
from transformers import AutoTokenizer, AutoModel
from transform_data import transform_and_cache
from utilities import get_directory, get_output_path, get_response, enforce_rate_limit, load_txt_data, generate_info


# List of things to do:
# 1. Reproduce the code from the other papers to see if we get same results
# 2. 



def BM25_demonstration_selection(query_sentence, corpus, k):
    bm25 = BM25Okapi([s.lower().split() for s in corpus])
    scores = bm25.get_scores( query_sentence.lower().split())
    return np.argsort(scores)[::-1][:k]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_simcse_model():
    model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model

def compute_embeddings(texts, tokenizer, model, batch_size=32):
    print("Computing SimCSE embeddings...")
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            embeddings = outputs.pooler_output
            all_embeddings.append(embeddings.cpu())
    return normalize(torch.cat(all_embeddings, dim=0).numpy())

def SimCSE_demonstration_selection(query_sentence, embeddings, tokenizer, model, corpus, k):
    inputs = tokenizer([query_sentence], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        query_embedding = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu().numpy()
        query_embedding = normalize(query_embedding)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    return np.argsort(similarities)[-k:][::-1]


def evaluation(test_data, results):
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

def is_prediction_misaligned(prediction_str, test_sample):
    try:
        prediction = json.loads(prediction_str)
    except json.JSONDecodeError:
        return True 

    aspect = test_sample.get("aspect")
    if not isinstance(prediction, dict):
        return True

    return aspect not in prediction

def load_existing_results(filepath, n):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            existing = json.load(f)
        return existing.get("results", ["{}"]*n), existing.get("inference_prompts", [])
    return ["{}"] * n, []

def load_data_and_embeddings(train_path, test_path, method, model, domain, key_groq, tokenizer=None, sim_model=None, use_paraphrase=False):
    train_data = load_txt_data(train_path)
    test_data = load_txt_data(test_path)
  
    
    if use_paraphrase:
        paraphrased_train_data = transform_and_cache(
                data=train_data,
                cache_path=f"cache/{model}/paraphrased/train_data_{domain}.json",
                model_name=model,
                api_key=key_groq
            )
    else:
        paraphrased_train_data = None
    
    embeddings, paraphrased_embeddings = None, None
    corpus = [d["text"] for d in train_data]

    if method == "SimCSE":
        assert tokenizer is not None and sim_model is not None
        embed_path = f"cache/regular_embeddings/embeddings_{domain}.npy"
        embeddings = np.load(embed_path) if os.path.exists(embed_path) else compute_and_cache_embeddings(corpus, tokenizer, sim_model, embed_path)

        if use_paraphrase:
            paraphrased_corpus = [d["paraphrased_text"] for d in paraphrased_train_data]
            embed_path_extra = f"cache/{model}/embeddings/embeddings_{domain}_transformed.npy"
            paraphrased_embeddings = np.load(embed_path_extra) if os.path.exists(embed_path_extra) else compute_and_cache_embeddings(paraphrased_corpus, tokenizer, sim_model, embed_path_extra)


    return train_data, test_data, embeddings, paraphrased_train_data, paraphrased_embeddings

def compute_and_cache_embeddings(corpus, tokenizer, model, path):
    embeddings = compute_embeddings(corpus, tokenizer, model)
    np.save(path, embeddings)
    return embeddings

def select_demonstration_indices(sentence, method, num_shots, train_corpus, train_embeddings, paraphrased_train_corpus, paraphrased_train_embeddings,  tokenizer=None, sim_model=None):
    demo_indices = {}
    if method == "SimCSE":
        assert tokenizer is not None and sim_model is not None
        if train_embeddings is not None:
            demo_indices["regular"] = SimCSE_demonstration_selection(sentence, train_embeddings, tokenizer, sim_model, train_corpus, num_shots)
        if paraphrased_train_embeddings is not None:
            demo_indices["paraphrased"] = SimCSE_demonstration_selection(sentence, paraphrased_train_embeddings, tokenizer, sim_model, paraphrased_train_corpus, num_shots)
    else:
        if train_corpus:
            demo_indices["regular"] = BM25_demonstration_selection(sentence, train_corpus, num_shots)
        if paraphrased_train_corpus:
            demo_indices["paraphrased"] = BM25_demonstration_selection(sentence, paraphrased_train_corpus, num_shots)
    return demo_indices

def format_demonstrations(indices, dataset, label):
    demos = []
    for idx in indices:
        data = dataset[idx]
        text = data.get("paraphrased_text", data["text"])
        demos.append(f"Sentence: {text}\nAspects: {data['aspect']} ({data['polarity']})")
    title = "Domain-invariant Demonstrations" if label == "paraphrased" else "Demonstrations"
    return f"\n{title}:\n" + "\n\n".join(demos)

def generate_prompt(sentence, aspects, demo_block, extra_demo_block):
    instruction = """
    Please perform the Aspect-Based Sentiment Classification task. Given an aspect in a sentence, assign a sentiment label from ['positive', 'negative', 'neutral'].
    """
    prompt = f"""{instruction}
    {demo_block}
    {extra_demo_block}
    Tested sample:
    - Original Sentence: {sentence}
    - Aspects: {aspects}
    \n
    Output:
    Generate the answer in a compact JSON format with no newlines or indentation, containing the following fields:
    - {aspects} - string that is one of the polarities ("Positive", "Negative", "Neutral")

    Always respond with a valid JSON. Do not include any extra characters, symbols, or text in or outside the JSON itself (including backticks, ", /)
    """
    return prompt



def main():
    load_dotenv()
    simcse_tokenizer, simcse_model = load_simcse_model()
    key_openai = os.getenv("OPENAI_API_KEY") 
    key_groq = os.getenv("GROQ_API_KEY")  
    key_groq_paid = os.getenv("GROQ_PAID_KEY")

    shot_infos = [{"num_shots": 6, "sources": ["regular"]},
                  {"num_shots": 6, "sources": ["paraphrased"]},
                  {"num_shots": 3, "sources": ["paraphrased", "regular"]},
                  {"num_shots": 0, "sources": []}]
    

    test_info = generate_info(
        source_domains=["laptop"],
        target_domains=["book"],
        demo="SimCSE",
        model="llama4_scout",
        shot_infos=shot_infos,
        indices=[3]
    )
   
    
    for (train_domain, test_domain, demo_method, model_choice, shot_info) in tqdm(test_info):
        year_train = 2019 if train_domain == "book" else 2014
        year_test = 2019 if test_domain == "book" else 2014
        train_path = f"data_out/{train_domain}/raw_data_{train_domain}_train_{year_train}.txt"
        test_path = f"data_out/{test_domain}/raw_data_{test_domain}_test_{year_test}.txt"

        shot_explanation = "" if shot_info["num_shots"] == 0 else f"shots from sources: {', '.join(shot_info['sources'])}"

        print(f"\nRunning {demo_method} with {shot_explanation} ({shot_info['num_shots']}) on {train_domain}→{test_domain}, model: {model_choice}")  

        subdir = get_directory(demo=demo_method, model=model_choice, shot_source=shot_info["sources"], num_shots=shot_info["num_shots"])
        filepath = get_output_path(source_domain=train_domain, target_domain=test_domain, num_shots=shot_info['num_shots'], subdir=subdir)
        os.makedirs(subdir, exist_ok=True)
       
        results, inference_prompts = load_existing_results(filepath, len(load_txt_data(test_path)))

        include_paraphrased = "paraphrased" in shot_info["sources"]
        include_regular = "regular" in shot_info["sources"]
       
      
        train_data, test_data, train_embeddings, paraphrased_train_data, paraphrased_train_embeddings = load_data_and_embeddings(
            train_path, test_path, demo_method, model_choice, train_domain, key_groq_paid, tokenizer=simcse_tokenizer, sim_model=simcse_model, use_paraphrase=include_paraphrased
        )
        with open("cache/llama4_scout/paraphrased/test_data_book.json", "r") as f:
            test_data = json.load(f)

       
   
       
        openai_client = OpenAI(api_key=key_openai)
        groq_client = Groq(api_key=key_groq)
        groq_client_paid = Groq(api_key=key_groq_paid)
        client = groq_client_paid

        MAX_REQUESTS_PER_MINUTE = 1000
        REQUEST_WINDOW = 60  
        request_times = deque()

    
        for i in tqdm(range(len(test_data))):
            if results[i].strip() != "{}":
                continue
            sample = test_data[i]

            demo_indices = select_demonstration_indices(
            sentence=sample["text"],
            method=demo_method,
            num_shots=shot_info["num_shots"],
            train_corpus=[d["text"] for d in train_data] if include_regular else None,
            train_embeddings=train_embeddings if include_regular else None,
            paraphrased_train_corpus=[d["paraphrased_text"] for d in paraphrased_train_data] if include_paraphrased else None,
            paraphrased_train_embeddings=paraphrased_train_embeddings if include_paraphrased else None,
            tokenizer=simcse_tokenizer, sim_model=simcse_model
        )


            demo_block = format_demonstrations(demo_indices["regular"], train_data, "regular") if include_regular else ""
            extra_demo_block = format_demonstrations(demo_indices["paraphrased"], paraphrased_train_data, "paraphrased") if include_paraphrased else ""
            prompt = generate_prompt(sample["paraphrased_text"], sample["aspect"], demo_block, extra_demo_block)
            
            try:
                enforce_rate_limit(request_times, MAX_REQUESTS_PER_MINUTE, REQUEST_WINDOW)
                output = get_response(prompt, client, model_choice)
            except Exception as e:
                print(f"Error generating response: {e}")
                output = "{}"

            results[i] = output 

            if len(inference_prompts) < 10:
                inference_prompts.append(prompt)

            with open(filepath, 'w') as f:
                json.dump({"results": results, "inference_prompts": inference_prompts}, f, indent=2)

        metrics = evaluation(test_data, results)
        print(json.dumps(metrics, indent=2))
        with open(filepath, 'w') as f:
            json.dump({"metrics": metrics, "results": results, "inference_prompts": inference_prompts}, f, indent=2)



if __name__ == "__main__":
    main()