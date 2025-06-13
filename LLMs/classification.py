import json
import numpy as np
import torch
import os
import time
import ast


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



def BM25_demonstration_selection(query_sentence, corpus):
    bm25 = BM25Okapi([s.lower().split() for s in corpus])
    scores = bm25.get_scores( query_sentence.lower().split())
    return np.argsort(scores)[::-1]

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

def SimCSE_demonstration_selection(query_sentence, embeddings, tokenizer, model):
    inputs = tokenizer([query_sentence], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        query_embedding = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu().numpy()
        query_embedding = normalize(query_embedding)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    return np.argsort(similarities)[-k:]

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

def top_k(sorted_indices: list[int], dataset, k:int):
    selected = []
    seen_pairs = set()
    for idx in sorted_indices:
        data = dataset[idx]
        text = data.get("paraphrased_text", data["text"])
        aspect = data['aspect']
        key = (text, aspect)
        if key not in seen_pairs:
            seen_pairs.add(key)
            selected.append(idx)
        if len(selected) == k:
            break
    return selected

def select_demonstration_indices(sentence, method, num_shots, train_data, train_embeddings, paraphrased_train_data, paraphrased_train_embeddings,  tokenizer=None, sim_model=None):
    demo_indices = {}
    train_corpus=[d["text"] for d in train_data] if train_data is not None else None,
    paraphrased_train_corpus=[d["paraphrased_text"] for d in paraphrased_train_data] if paraphrased_train_data is not None else None,
    if method == "SimCSE":
        assert tokenizer is not None and sim_model is not None
        if train_embeddings is not None:
            sorted_indices = SimCSE_demonstration_selection(sentence, train_embeddings, tokenizer, sim_model)
            demo_indices["regular"] = top_k(sorted_indices, train_data, num_shots)

        if paraphrased_train_embeddings is not None:
            sorted_indices = SimCSE_demonstration_selection(sentence, paraphrased_train_embeddings, tokenizer, sim_model)
            demo_indices["paraphrased"] = top_k(sorted_indices, paraphrased_train_data, num_shots)
    else:
        if train_corpus:
            sorted_indices = BM25_demonstration_selection(sentence, train_corpus)
            demo_indices["regular"] = top_k(sorted_indices, train_data, num_shots)
        if paraphrased_train_corpus:
            sorted_indices =  BM25_demonstration_selection(sentence, paraphrased_train_corpus)
            demo_indices["paraphrased"] = top_k(sorted_indices, paraphrased_train_data, num_shots)
    return demo_indices

def format_demonstrations(indices, dataset, label, is_tuple:bool, skip_title=False):
    demos = []
    for idx in indices:
        data = dataset[idx]
        # text = data.get("paraphrased_text", data["text"])
        # aspect = data['aspect']
        # polarity = data['polarity']
        arr = data['paraphrased_text'].split(',')
        text, aspect, polarity = arr[0][2:-1], arr[1][1:-1], arr[2][1:-2]
        demos.append(f"Sentence: {text}\nAspects: {aspect} ({polarity})")
    if skip_title:
        return "\n" + "\n\n".join(demos)
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

def load_dependent_independent_sources(base_path, domain, tokenizer, model):
    dep_path = os.path.join(base_path, f"dependent_train_data_{domain}_200.json")
    indep_path = os.path.join(base_path, f"independent_train_data_{domain}_200.json")

    with open(dep_path, "r") as f:
        dep_data = json.load(f)
    with open(indep_path, "r") as f:
        indep_data = json.load(f)

    dep_corpus = [d["paraphrased_text"].split(',')[0][2:-1] for d in dep_data]
    indep_corpus = [d["paraphrased_text"].split(',')[0][2:-1] for d in indep_data]
    

    dep_embed_path = os.path.join(base_path, f"../embeddings/dependent_embeddings_{domain}.npy")
    indep_embed_path = os.path.join(base_path, f"../embeddings/independent_embeddings_{domain}.npy")

    dep_embeddings = (
        np.load(dep_embed_path)
        if os.path.exists(dep_embed_path)
        else compute_and_cache_embeddings(dep_corpus, tokenizer, model, dep_embed_path)
    )
    indep_embeddings = (
        np.load(indep_embed_path)
        if os.path.exists(indep_embed_path)
        else compute_and_cache_embeddings(indep_corpus, tokenizer, model, indep_embed_path)
    )

    return dep_data, indep_data, dep_embeddings, indep_embeddings


def main():
    load_dotenv()
    simcse_tokenizer, simcse_model = load_simcse_model()
    key_openai = os.getenv("OPENAI_API_KEY") 
    key_groq = os.getenv("GROQ_API_KEY")  
    key_groq_paid = os.getenv("GROQ_PAID_KEY")

    shot_infos = [{"num_shots": 6, "sources": ["regular"]},
                  {"num_shots": 6, "sources": ["paraphrased"]},
                  {"num_shots": 3, "sources": ["paraphrased", "regular"]},
                  {"num_shots": 3, "sources": ["independent", "dependent"]},
                  {"num_shots": 0, "sources": []}]
    

    test_info = generate_info(
        source_domains=["laptop"],
        target_domains=["book", "laptop", "restaurant"],
        demos=["SimCSE"],
        models=["llama4_scout"],
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

        subdir = get_directory(demo=demo_method, model=model_choice, shot_info=shot_info)
        filepath = get_output_path(source_domain=train_domain, target_domain=test_domain, num_shots=shot_info['num_shots'], subdir=subdir)
        os.makedirs(subdir, exist_ok=True)
       
        results, inference_prompts = load_existing_results(filepath, len(load_txt_data(test_path)))

        include_paraphrased = "paraphrased" in shot_info["sources"]
        include_regular = "regular" in shot_info["sources"]
       
      
        train_data, test_data, train_embeddings, paraphrased_train_data, paraphrased_train_embeddings = load_data_and_embeddings(
            train_path, test_path, demo_method, model_choice, train_domain, key_groq_paid, tokenizer=simcse_tokenizer, sim_model=simcse_model, use_paraphrase=include_paraphrased
        )

       
        base_cache_path = "cache/llama4_scout/paraphrased"
        dependent_data, independent_data, dependent_embeds, independent_embeds = load_dependent_independent_sources(base_cache_path, train_domain, simcse_tokenizer, simcse_model)

       
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
            sentence = sample["text"]
            aspect = sample["aspect"]

            demo_indices = select_demonstration_indices(
            sentence=sample["text"],
            method=demo_method,
            num_shots=shot_info["num_shots"],
            train_dataset = ,
            train_embeddings=train_embeddings if include_regular else None,
            paraphrased_train_dataset = ,
            paraphrased_train_embeddings=paraphrased_train_embeddings if include_paraphrased else None,
            tokenizer=simcse_tokenizer, sim_model=simcse_model
        )   

            dep_indices = SimCSE_demonstration_selection(sentence, dependent_embeds, simcse_tokenizer, simcse_model, [d["paraphrased_text"] for d in dependent_data], shot_info["num_shots"])
            indep_indices = SimCSE_demonstration_selection(sentence, independent_embeds, simcse_tokenizer, simcse_model, [d["paraphrased_text"] for d in independent_data], shot_info["num_shots"])

            dep_block = format_demonstrations(dep_indices, dependent_data, "dependent", is_tuple=True)
            indep_block = format_demonstrations(indep_indices, independent_data, "independent", is_tuple=True, skip_title=True)

            prompt = generate_prompt(sentence, aspect, dep_block, indep_block)
           
            # demo_block = format_demonstrations(demo_indices["regular"], train_data, "regular") if include_regular else ""
            # extra_demo_block = format_demonstrations(demo_indices["paraphrased"], paraphrased_train_data, "paraphrased") if include_paraphrased else ""
            # prompt = generate_prompt(sample["text"], sample["aspect"], demo_block, extra_demo_block)
            
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