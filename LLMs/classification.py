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



def enforce_rate_limit(request_times, MAX_REQUESTS_PER_MINUTE, REQUEST_WINDOW):
    current_time = time.time()
    while request_times and current_time - request_times[0] > REQUEST_WINDOW:
        request_times.popleft()
    if len(request_times) >= MAX_REQUESTS_PER_MINUTE:
        sleep_time = REQUEST_WINDOW - (current_time - request_times[0]) + 0.1
        time.sleep(sleep_time)
        return enforce_rate_limit(request_times, MAX_REQUESTS_PER_MINUTE, REQUEST_WINDOW)
    request_times.append(current_time)
   


def load_txt_data(path):
    samples = []
    with open(path, "r", encoding='latin-1') as f:
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





def get_response(prompt, client, model):
    model_map = {
        "gpt-4o": "gpt-4o-mini",
        "llama3": "llama3-70b-8192",
        "deepseek_llama": "deepseek-r1-distill-llama-70b",
        "gemma": "gemma2-9b-it",
        "qwen32": "qwen-qwq-32b",
        "llama4_mav": "meta-llama/llama-4-maverick-17b-128e-instruct"
    }
    model = model_map.get(model, model)
    messages = [{"role": "user", "content": prompt}]
    output = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return output.choices[0].message.content


# BM25
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



def main():
    load_dotenv()
    key_openai = os.getenv("OPENAI_API_KEY") 
    key_groq = os.getenv("GROQ_API_KEY")  
    source_domain_target_domain_use_SimCSE_model = [("restaurant", "laptop", True, "llama3", True),
                                                    ("restaurant", "book", True, "llama3", True)]
    
    for input_tuple in tqdm(source_domain_target_domain_use_SimCSE_model):
        train_domain = input_tuple[0]
        test_domain = input_tuple[1]
        year_train = 2019 if train_domain == "book" else 2014
        year_test = 2019 if test_domain == "book" else 2014
       

        train_file_path = f"data_out/{train_domain}/raw_data_{train_domain}_train_{year_train}.txt"
        test_file_path = f"data_out/{test_domain}/raw_data_{test_domain}_test_{year_test}.txt"

        use_SimCSE = input_tuple[2] # if true, use SimCSE, otherwise use BM25 for demonstration selection
        model_choice = input_tuple[3] # options: "llama3", "llama4", "deepseek_llama", "gemma", "qwen32"
        use_transformation = input_tuple[4]
        num_shots = 3  # number of demonstrations to use
        print(f"Processing source domain: {train_domain}, target domain: {test_domain}, using SimCSE: {use_SimCSE}, num shots: {num_shots}")

        # These depend on the limit of the API
        MAX_REQUESTS_PER_MINUTE = 30
        REQUEST_WINDOW = 60  
        request_times = deque()


        train_data = load_txt_data(train_file_path)
        test_data = load_txt_data(test_file_path)
       

        if use_transformation:
            # train_data = transform_and_cache(
            #     data=train_data,
            #     cache_path=f"cache/paraphrased_train_{train_domain}.json",
            #     model_name=model_choice,
            #     api_key=key_groq
            # )

            test_data = transform_and_cache(
                data=test_data,
                cache_path=f"cache/paraphrased_test_{test_domain}.json",
                model_name=model_choice,
                api_key=key_groq
            )

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

        openai_client_openai = OpenAI(api_key= key_openai)
        groq_client = Groq(api_key=key_groq)
        client = groq_client

        results = []
        inference_prompts = []

        special_indices = [1, 2, 3]

        for i, sample in tqdm(enumerate(test_data)):
            # if i not in special_indices: 
            #     continue
            sentence_text = sample["text"]
            template = sample["template"]
            aspects = sample["aspect"]
        

            if use_SimCSE:
                top_indices = SimCSE_demonstration_selection(sentence_text, train_embeddings, tokenizer, simcse_model, train_corpus, num_shots)
            else:
                top_indices = BM25_demonstration_selection(sentence_text, train_corpus, num_shots)

            demonstrations = []
            for idx in top_indices:
                demo = train_data[idx]
                demonstrations.append(f"Sentence: {demo['template']}\nAspects: {demo['aspect']} ({demo['polarity']})")
            demonstrations = "\n\n".join(demonstrations)
          

            if num_shots > 0:
                demo_block = f"\nDemonstrations:\n{demonstrations}\n"
            else:
                demo_block = ""

            if use_transformation and "paraphrased_text" in sample:
                paraphrased_block = f"- Paraphrased Sentence: {sample['paraphrased_text']}\n"
            else:
                paraphrased_block = ""


            instruction = """
            Please perform the Aspect-Based Sentiment Classification task. Given an aspect in a sentence, assign a sentiment label from ['positive', 'neutral', 'negative'].
            {paraphrased_notice}
            """


            if use_transformation and "paraphrased_text" in sample:
                paraphrased_notice = "You are provided with both the original sentence and a paraphrased version. Consider both when making your prediction."
            else:
                paraphrased_notice = ""

            base_prompt = instruction + """
            {demo_block}
            Tested sample:
            - Original Sentence: {sentence}
            {paraphrased_block}
            - Aspects: {aspects}

            Output:
            Generate the answer in a compact JSON format with no newlines or indentation, containing the following fields:
            - {aspects} - string that is one of the polarities ("Positive", "Negative", "Neutral")

            Always respond with a valid JSON. Do not include any extra characters, symbols, or text in or outside the JSON itself (including backticks, ", /)
            """

            paraphrased_block = f"- Paraphrased Sentence: {sample['paraphrased_text']}\n" if use_transformation and "paraphrased_text" in sample else ""

            prompt = base_prompt.format(
                demo_block=demo_block,
                sentence=sentence_text,
                paraphrased_block=paraphrased_block,
                aspects=aspects,
                paraphrased_notice=paraphrased_notice
            )

            try:
                enforce_rate_limit(request_times=request_times, MAX_REQUESTS_PER_MINUTE=MAX_REQUESTS_PER_MINUTE, REQUEST_WINDOW=REQUEST_WINDOW)
                output = get_response(prompt, client, model_choice)
                results.append(output)
                if len(inference_prompts) < 10:
                    inference_prompts.append(prompt)
            except Exception as e:
                print(f"Error generating response: {e}")
                results.append("{}")  

        metrics = evaluation(test_data, results)



        print("\n\nInference Prompts and Responses:")
        for i, prompt in enumerate(inference_prompts):
            print(f"Prompt {i + 1}:\n{prompt}\n")
            print(f"Response {i + 1}:\n{results[i]}\n")

        print(json.dumps(metrics, indent=2))

        subdir = f"results/{model_choice}/SimCSE/{num_shots}_shot" if use_SimCSE else f"results/{model_choice}/bm25/{num_shots}_shot"
        os.makedirs(subdir, exist_ok=True)
        filepath = os.path.join(subdir, f"results_{model_choice}_{train_domain}_{test_domain}_{num_shots}_shot_with_transformation.json")

        with open(filepath, "w") as f:
            json.dump({
                "metrics": metrics,
                "results": results,
                "inference_prompts": inference_prompts
            }, f, indent=2)


if __name__ == "__main__":
    main()