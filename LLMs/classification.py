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
from utilities import get_directory, get_output_path, get_response, enforce_rate_limit, load_txt_data


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


def main():
    load_dotenv()
    key_openai = os.getenv("OPENAI_API_KEY") 
    key_groq = os.getenv("GROQ_API_KEY")  
    # (source_domain, target_domain, use_SimCSE, Model, use_transformation, shots)
    test_info = [("laptop", "book", True, "gemma", True, 3),
                 ("laptop", "book", True, "llama3", True, 3)]
      


    for input_tuple in tqdm(test_info):
        train_domain = input_tuple[0]
        test_domain = input_tuple[1]
        year_train = 2019 if train_domain == "book" else 2014
        year_test = 2019 if test_domain == "book" else 2014
       
        train_file_path = f"data_out/{train_domain}/raw_data_{train_domain}_train_{year_train}.txt"
        test_file_path = f"data_out/{test_domain}/raw_data_{test_domain}_test_{year_test}.txt"

        use_SimCSE = input_tuple[2] # if true, use SimCSE, otherwise use BM25 for demonstration selection
        model_choice = input_tuple[3] # options: "llama3", "llama4", "deepseek_llama", "gemma", "qwen32"
        use_transformation = input_tuple[4]
        num_shots = input_tuple[5]  # number of demonstrations to use

        if num_shots == 0:
            print(f"\n Processing target domain: {test_domain}, using model: {model_choice}, using transformation? {use_transformation}, num_shots: {num_shots}")
        else:
            print(f"\n Processing source_domain: {train_domain}, target domain: {test_domain}, using model: {model_choice}, using SimCSE? {use_SimCSE,} using transformation? {use_transformation}, num_shots: {num_shots}")
        
        subdir = get_directory(use_SimCSE=use_SimCSE, model=model_choice, use_transformation=use_transformation, num_shots=num_shots)
        filepath = get_output_path(source_domain=train_domain, target_domain=test_domain, model=model_choice, use_transformation=use_transformation, num_shots=num_shots, subdir=subdir)
        os.makedirs(subdir, exist_ok=True)
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                existing_data = json.load(f)
            results = existing_data.get("results", [])
            inference_prompts = existing_data.get("inference_prompts", [])
        else:
            results = []
            inference_prompts = []

        # These depend on the limit of the API
        MAX_REQUESTS_PER_MINUTE = 30
        REQUEST_WINDOW = 60  
        request_times = deque()


        train_data = load_txt_data(train_file_path)
        test_data = load_txt_data(test_file_path)
       
        # for i, (res, sample) in enumerate(zip(results, test_data)):
        #     if is_prediction_misaligned(res, sample):
        #         print(f"Misaligned prediction at index {i}: {res} (aspect: {sample['aspect']})")
        # print("stop")
        # exit()
        
        if use_transformation:
            extra_train_data = transform_and_cache(
                data=train_data,
                cache_path=f"cache/{model_choice}_paraphrased_train_{train_domain}.json",
                model_name=model_choice,
                api_key=key_groq
            )
            train_extra_corpus = [sample["paraphrased_text"] for sample in extra_train_data]

            # test_data = transform_and_cache(
            #     data=test_data,
            #     cache_path=f"cache/paraphrased_test_{test_domain}.json",
            #     model_name=model_choice,
            #     api_key=key_groq
            # )

        train_corpus = [sample["text"] for sample in train_data]
        

        if use_SimCSE:
            tokenizer, simcse_model = load_simcse_model()
            os.makedirs("cache", exist_ok=True)
            embed_path = f"cache/embeddings_{train_domain}_simcse.npy"
            if os.path.exists(embed_path):
                print("Loading cached SimCSE embeddings...")
                train_embeddings = np.load(embed_path)
            else:
                print("Computing SimCSE embeddings...")
                train_embeddings = compute_embeddings(train_corpus, tokenizer, simcse_model)
                np.save(embed_path, train_embeddings)

            if use_transformation:
                embed_extra_path = f"cache/embeddings_{train_domain}_transformed_simcse.npy"
                if os.path.exists(embed_extra_path):
                    print("Loading extra cached SimCSE embeddings...")
                    train_extra_embeddings = np.load(embed_extra_path)
                else:
                    print("Computing extra SimCSE embeddings...")
                    train_extra_embeddings = compute_embeddings(train_extra_corpus, tokenizer, simcse_model)
                    np.save(embed_extra_path, train_extra_embeddings)

        openai_client_openai = OpenAI(api_key= key_openai)
        groq_client = Groq(api_key=key_groq)
        client = groq_client

     
        while len(results) < len(test_data):
            results.append("{}")

        for i in tqdm(range(len(test_data))):
            if results[i].strip() != "{}":
                continue

            sample = test_data[i]

            sentence_text = sample["text"]
            template = sample["template"]
            aspects = sample["aspect"]
        

            if use_SimCSE:
                top_indices = SimCSE_demonstration_selection(sentence_text, train_embeddings, tokenizer, simcse_model, train_corpus, num_shots)
                if use_transformation:
                    top_extra_indices = SimCSE_demonstration_selection(sentence_text, train_extra_embeddings, tokenizer, simcse_model, train_extra_corpus, num_shots)
            else:
                top_indices = BM25_demonstration_selection(sentence_text, train_corpus, num_shots)
                if use_transformation:
                    top_extra_indices = BM25_demonstration_selection(sentence_text, train_extra_corpus, num_shots)

            demonstrations = []
            for idx in top_indices:
                demo = train_data[idx]
                demonstrations.append(f"Sentence: {demo['template']}\nAspects: {demo['aspect']} ({demo['polarity']})")
            demonstrations = "\n\n".join(demonstrations)

            if use_transformation:
                extra_demonstrations = []
                for idx in top_extra_indices:
                    demo = extra_train_data[idx]
                    extra_demonstrations.append(f"Sentence: {demo['paraphrased_text']}\nAspects: {demo['aspect']} ({demo['polarity']})")
                extra_demonstrations = "\n\n".join(extra_demonstrations)

            if num_shots > 0:
                demo_block = f"\nDemonstrations:\n{demonstrations}\n"
                if use_transformation:
                    extra_demo_block = f"\nDomain-invariant Demonstrations:\n{extra_demonstrations}\n"
                else:
                    extra_demo_block = ""
            else:
                demo_block = ""
                extra_demo_block = ""


            instruction = """
            Please perform the Aspect-Based Sentiment Classification task. Given an aspect in a sentence, assign a sentiment label from ['positive', 'negative', 'neutral'].
            """

            base_prompt = instruction + """
            {demo_block}
            {extra_demo_block}
            Tested sample:
            - Original Sentence: {sentence}
            - Aspects: {aspects}

            Output:
            Generate the answer in a compact JSON format with no newlines or indentation, containing the following fields:
            - {aspects} - string that is one of the polarities ("Positive", "Negative", "Neutral")

            Always respond with a valid JSON. Do not include any extra characters, symbols, or text in or outside the JSON itself (including backticks, ", /)
            """

            prompt = base_prompt.format(
                demo_block=demo_block,
                sentence=sentence_text,
                aspects=aspects,
                extra_demo_block = extra_demo_block
            )

            if i == 0:
                print(prompt)

            try:
                enforce_rate_limit(request_times=request_times, MAX_REQUESTS_PER_MINUTE=MAX_REQUESTS_PER_MINUTE, REQUEST_WINDOW=REQUEST_WINDOW)
                output = get_response(prompt, client, model_choice)
            except Exception as e:
                print(f"Error generating response: {e}")
                output = "{}"  

            
            results[i] = output
         
            if len(inference_prompts) < 10:
                inference_prompts.append(prompt)

            with open(filepath, 'w') as f:
                json.dump({
                    "results": results,
                    "inference_prompts": inference_prompts
                }, f, indent=2)

        metrics = evaluation(test_data, results)
        print(json.dumps(metrics, indent=2))
        with open(filepath, 'w') as f:
            json.dump({
                "metrics": metrics,
                "results": results,
                "inference_prompts": inference_prompts
            }, f, indent=2)



if __name__ == "__main__":
    main()