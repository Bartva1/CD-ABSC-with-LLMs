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
    return np.argsort(similarities)[::-1]

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

def load_data_and_embeddings(train_path, test_path, demo_method, model, domain, key_groq, tokenizer=None, sim_model=None, use_paraphrase=False):
    train_data = load_txt_data(train_path)
    test_data = load_txt_data(test_path)
  
    
    if use_paraphrase:
        paraphrased_train_data = transform_and_cache(
                domain=domain,
                prompt_version='basic',
                data=train_data,
                cache_path=f"cache/{model}/paraphrased/train_data_{domain}.json",
                model_name=model,
                api_key=key_groq
            )
    else:
        paraphrased_train_data = None
    
    embeddings, paraphrased_embeddings = None, None
    corpus = [d["text"] for d in train_data]

    if demo_method == "SimCSE":
        assert tokenizer is not None and sim_model is not None
        embed_path = f"cache/regular_embeddings/embeddings_{domain}.npy"
        embeddings = np.load(embed_path) if os.path.exists(embed_path) else compute_and_cache_embeddings(corpus, tokenizer, sim_model, embed_path)

        if use_paraphrase:
            paraphrased_corpus = [d["paraphrased_text"] for d in paraphrased_train_data]
            embed_path_extra = f"cache/{model}/embeddings/basic_transformation/embeddings_{domain}_transformed.npy"
            paraphrased_embeddings = np.load(embed_path_extra) if os.path.exists(embed_path_extra) else compute_and_cache_embeddings(paraphrased_corpus, tokenizer, sim_model, embed_path_extra)


    return train_data, test_data, embeddings, paraphrased_train_data, paraphrased_embeddings

def compute_and_cache_embeddings(corpus, tokenizer, model, path):
    embeddings = compute_embeddings(corpus, tokenizer, model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)
    return embeddings

def top_k(sorted_indices: list[int], dataset: list, k: int, sources:str):
    selected = []
    seen_pairs = set()
    for idx in sorted_indices:
        data = dataset[idx]
        if "dependent" in sources or "independent" in sources:
            try:
                arr = data['paraphrased_text'].split(',')
                text, aspect = arr[0].strip()[2:-1], arr[1].strip()[1:-1]
            except Exception as e:
                print(f"Error during parsing (in)dependent sentence: {e}")
                print(f"Sentence which failed to parse: {arr}")
                exit()

        else:
            text, aspect = data.get("paraphrased_text", data["text"]), data['aspect']
     
        key = (text, aspect)
        if key not in seen_pairs:
            seen_pairs.add(key)
            selected.append(idx)
        if len(selected) == k:
            break
    return selected

def select_demonstration_indices(sentence, method, num_shots, datasets, embeddings, sources, tokenizer=None, sim_model=None):
    demo_indices = {}
    for version, dataset in datasets.items():
        if dataset is None or version not in embeddings:
            continue

        corpus = [d.get("paraphrased_text", d["text"]) for d in dataset]
        if "dependent" in version:
            corpus = [d['paraphrased_text'].split(',')[0][2:-1] for d in dataset]
        embedding = embeddings[version]

        if method == "SimCSE":
            assert tokenizer is not None and sim_model is not None
            sorted_indices = SimCSE_demonstration_selection(sentence, embedding, tokenizer, sim_model)
        else:
            sorted_indices = BM25_demonstration_selection(sentence, corpus)

        demo_indices[version] = top_k(sorted_indices, dataset, num_shots, sources)
    return demo_indices
   

def format_demonstrations(indices: list[int], dataset, label, is_tuple:bool, skip_title=False):
    demos = []
    for idx in indices:
        data = dataset[idx]
        if is_tuple:
            try:
                arr = data['paraphrased_text'].split(',')
                # print(arr)
                text, aspect, polarity = arr[0].strip()[2:-1], arr[1].strip()[1:-1], arr[2].strip()[1:-2]
            except Exception as e:
                print(f"Error during parsing (in)dependent sentence: {e}")
                print(f"Sentence which failed to parse: {arr}")
                exit()
        else:
            text, aspect, polarity = data.get("paraphrased_text", data["text"]), data['aspect'], data['polarity']
        
        demos.append(f"Sentence: {text}\nAspects: {aspect} ({polarity})")
    if skip_title:
        return "\n" + "\n\n".join(demos)
    title = "Domain-invariant Demonstrations" if label == "paraphrased" else "Demonstrations"
    return f"\n{title}:\n" + "\n\n".join(demos)

def generate_prompt(sentence, aspects, demo_block):
    instruction = """
    Please perform the Aspect-Based Sentiment Classification task. Given an aspect in a sentence, assign a sentiment label from ['positive', 'negative', 'neutral'].
    """
    prompt = f"""{instruction}
    {demo_block}
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

def load_dependent_independent_sources(train_data, demo_method, model, domain, key_groq, tokenizer, sim_model):
    dep_data = transform_and_cache(
        domain=domain,
        prompt_version="dependent",
        data = train_data,
        cache_path=f"cache/{model}/dependent/train_data_{domain}.json",
        model_name=model,
        api_key=key_groq
    )

    indep_data = transform_and_cache(
        domain=domain,
        prompt_version="independent",
        data = train_data,
        cache_path=f"cache/{model}/independent/train_data_{domain}.json",
        model_name=model,
        api_key=key_groq
    )
   
    dep_corpus = [d["paraphrased_text"].split(',')[0][2:-1] for d in dep_data]
    indep_corpus = [d["paraphrased_text"].split(',')[0][2:-1] for d in indep_data]
    

    dep_embeddings, indep_embeddings = None, None

    if demo_method == "SimCSE":
        assert tokenizer is not None and sim_model is not None
        embed_path_dep = f"cache/{model}/embeddings/dependent/embeddings_{domain}.npy"
        dep_embeddings = np.load(embed_path_dep) if os.path.exists(embed_path_dep) else compute_and_cache_embeddings(dep_corpus, tokenizer, sim_model, embed_path_dep)

        embed_path_indep = f"cache/{model}/embeddings/independent/embeddings_{domain}.npy"
        indep_embeddings = np.load(embed_path_indep) if os.path.exists(embed_path_indep) else compute_and_cache_embeddings(indep_corpus, tokenizer, sim_model, embed_path_indep)

    return dep_data, indep_data, dep_embeddings, indep_embeddings


def main():
    load_dotenv()
    simcse_tokenizer, simcse_model = load_simcse_model()
    key_openai = os.getenv("OPENAI_API_KEY") 
    key_groq = os.getenv("GROQ_API_KEY")  
    key_groq_paid = os.getenv("GROQ_PAID_KEY")
    key_gemini = os.getenv("GEMINI_KEY")

    shot_infos = [{"num_shots": 6, "sources": ["regular"]},
                  {"num_shots": 6, "sources": ["paraphrased"]},
                  {"num_shots": 3, "sources": ["paraphrased", "regular"]},
                  {"num_shots": 3, "sources": ["independent", "dependent"]},
                  {"num_shots": 0, "sources": []}]
    
    test_info = generate_info(
            source_domains=["restaurant"],
            target_domains=["laptop" ],
            demos=["SimCSE"],
            models=["llama4_scout"],
            shot_infos=shot_infos,
            indices=[0,3]
        )
   
    
    for (train_domain, test_domain, demo_method, model_choice, shot_info) in tqdm(test_info):
        # making paths for the original train and test data
        year_train = 2019 if train_domain == "book" else 2014
        year_test = 2019 if test_domain == "book" else 2014
        train_path = f"data_out/{train_domain}/raw_data_{train_domain}_train_{year_train}.txt"
        test_path = f"data_out/{test_domain}/raw_data_{test_domain}_test_{year_test}.txt"
            
        # explanation for the current run
        shot_explanation = "" if shot_info["num_shots"] == 0 else f"shots from sources: {', '.join(shot_info['sources'])}"
        print(f"\nRunning {demo_method} with {shot_explanation} ({shot_info['num_shots']}) on {train_domain}→{test_domain}, model: {model_choice}")  

        # output path
        subdir = get_directory(demo=demo_method, model=model_choice, shot_info=shot_info)
        filepath = get_output_path(source_domain=train_domain, target_domain=test_domain, num_shots=shot_info['num_shots'], subdir=subdir)
        os.makedirs(subdir, exist_ok=True)
       
        # loading cached results
        results, inference_prompts = load_existing_results(filepath, len(load_txt_data(test_path)))

        include_paraphrased = "paraphrased" in shot_info["sources"]

        # loading/creating train data, test data, and paraphrased version + embeddings
        train_data, test_data, train_embeddings, paraphrased_train_data, paraphrased_train_embeddings = load_data_and_embeddings(
            train_path, test_path, demo_method, model_choice, train_domain, key_groq_paid, simcse_tokenizer, simcse_model, include_paraphrased
        ) 

        # creating transformed data for domain-dependent and domain-independent
        if "dependent" in shot_info["sources"]:
            dependent_train_data, independent_train_data, dependent_train_embeddings, independent_train_embeddings = load_dependent_independent_sources(
                train_data, demo_method, model_choice, train_domain, key_groq_paid, simcse_tokenizer, simcse_model
            )
        else:
             dependent_train_data, independent_train_data, dependent_train_embeddings, independent_train_embeddings = None, None, None, None


       
        openai_client = OpenAI(api_key=key_openai)
        groq_client = Groq(api_key=key_groq)
        groq_client_paid = Groq(api_key=key_groq_paid)
        client = groq_client_paid

        MAX_REQUESTS_PER_MINUTE = 1000
        REQUEST_WINDOW = 60 
        request_times = deque()

        datasets = {
            "regular": train_data if "regular" in shot_info["sources"] else None,
            "paraphrased": paraphrased_train_data if "paraphrased" in shot_info["sources"] else None,
            "dependent": dependent_train_data if "dependent" in shot_info["sources"] else None,
            "independent": independent_train_data if "independent" in shot_info["sources"] else None
        }

        all_embeddings = {
            "regular": train_embeddings,
            "paraphrased": paraphrased_train_embeddings,
            "dependent": dependent_train_embeddings,
            "independent": independent_train_embeddings
        }
        while len(results) < len(test_data):
            results.append("{}")
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
            datasets = datasets,
            embeddings=all_embeddings,
            tokenizer=simcse_tokenizer, sim_model=simcse_model,
            sources=shot_info['sources']
            )   

            # might opt to refactor this, as it is a bit lengthy
            demo_block_parts = []

            if "regular" in shot_info["sources"] and "regular" in demo_indices:
                demo_block_parts.append(
                    format_demonstrations(demo_indices["regular"], train_data, "regular", is_tuple=False)
                )

            if "paraphrased" in shot_info["sources"] and "paraphrased" in demo_indices:
                demo_block_parts.append(
                    format_demonstrations(demo_indices["paraphrased"], paraphrased_train_data, "paraphrased", is_tuple=False)
                )

            if "dependent" in shot_info["sources"] and "dependent" in demo_indices:
                demo_block_parts.append(
                    format_demonstrations(demo_indices["dependent"], dependent_train_data, "dependent", is_tuple=True)
                )

            if "independent" in shot_info["sources"] and "independent" in demo_indices:
                demo_block_parts.append(
                    format_demonstrations(demo_indices["independent"], independent_train_data, "independent", is_tuple=True, skip_title=True)
                )

            demo_block_combined = "\n".join(demo_block_parts)
            prompt = generate_prompt(sentence, aspect, demo_block_combined)
           
        
            try:
                enforce_rate_limit(request_times, MAX_REQUESTS_PER_MINUTE, REQUEST_WINDOW)
                if model_choice == 'gemma3' or model_choice == 'gemini_flash':
                    output = get_response(prompt, client, model_choice, key_gemini)
                else: 
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
            json.dump({"metrics": metrics, "results": results}, f, indent=2)



if __name__ == "__main__":
    main()