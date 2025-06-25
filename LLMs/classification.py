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
from utilities import get_directory, get_output_path, get_response, enforce_rate_limit, load_txt_data, generate_info, parse_experiment_args



def BM25_demonstration_selection(query_sentence: str, corpus: list[str]) -> np.ndarray:
    """
    Rank the corpus containing the train sentences based on bm25 similarity to the query sentence.

    Args:
        query_sentence: The test sentence to match against the corpus.
        corpus: A list of sentences representing the train set

    Returns:
        An array of indices sorted by relevance in descending order.
    """
    
    bm25 = BM25Okapi([s.lower().split() for s in corpus])
    scores = bm25.get_scores( query_sentence.lower().split())
    return np.argsort(scores)[::-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_simcse_model() -> tuple[AutoTokenizer, AutoModel]:
    """
    Load the pretrained SimCSE model and tokenizer onto the appropriate device.

    Returns:
        A tuple containing the tokenizer and model.
    """
    model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model

def compute_embeddings(
        texts: list[str],
        tokenizer: AutoTokenizer,
        model: AutoModel,
        batch_size: int = 32
) -> np.ndarray:
    """
    Compute and normalize SimCSE embeddings for a list of texts.

    Args:
        texts: List of input texts.
        tokenizer: Tokenizer compatible with the model.
        model: SimCSE model.
        batch_size: Batch size for embedding computation.

    Returns:
        A numpy array of SimCSE embeddings.
    """

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

def SimCSE_demonstration_selection(
    query_sentence: str, 
    embeddings: np.ndarray, 
    tokenizer: AutoTokenizer, 
    model: AutoModel
) -> np.ndarray:
    """
    Rank the dataset based on cosine similarity between query and corpus embeddings.

    Args:
        query_sentence: The test sentence for which to find similar examples.
        embeddings: Embeddings of the corpus.
        tokenizer: Tokenizer for the SimCSE model.
        model: SimCSE model.

    Returns:
        Indices of sorted examples by descending similarity.
    """
    inputs = tokenizer([query_sentence], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        query_embedding = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu().numpy()
        query_embedding = normalize(query_embedding)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    return np.argsort(similarities)[::-1]

def evaluation(test_data: list[dict], results: list[str]) -> dict[str,float]:
    """
    Evaluate predictions against the ground truth using accuracy and F1 metrics.

    Args:
        test_data: Ground truth data samples.
        results: JSON string responses from the model.

    Returns:
        Dictionary containing accuracy, weighted F1, and macro F1 scores.
    """
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


def load_existing_results(filepath: str, n: int) -> tuple[list[str], list[str]]:
    """
    Load previously saved results if they exist.

    Args:
        filepath: Path to the results file.
        n: Number of samples in the results file

    Returns:
        Tuple of results and prompts.
    """

    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            existing = json.load(f)
        return existing.get("results", ["{}"]*n), existing.get("inference_prompts", [])
    return ["{}"] * n, []

def load_data_and_embeddings(
    train_path: str,
    test_path: str,
    demo_method: str, 
    model: str, 
    domain: str, 
    key_groq: str, 
    tokenizer: AutoTokenizer | None = None, 
    sim_model: AutoModel | None = None, 
    use_paraphrase: bool = False
) -> tuple[list[dict], list[dict], np.ndarray | None, list[dict] | None, np.ndarray | None]:
    
    """
    Load the train and test datasets, and optionally compute or load embeddings
    and paraphrased versions of the data depending on the demonstration method.

    Args:
        train_path: File path to the training dataset (.txt).
        test_path: File path to the test dataset (.txt).
        demo_method: Method for selecting demonstrations ('SimCSE' or 'bm25').
        model: Name of the LLM used.
        domain: source domain used.
        key_groq: API key for accessing LLMs.
        tokenizer: Tokenizer instance for SimCSE, if applicable.
        sim_model: SimCSE model, if applicable.
        use_paraphrase: Whether to use AP-DI demonstrations.

    Returns:
        A tuple:
        - train_data: The loaded training dataset.
        - test_data: The loaded test dataset.
        - embeddings: Numpy array of SimCSE embeddings (if demo_method == 'SimCSE').
        - paraphrased_data: AP-DI training data (if applicable).
        - paraphrased_embeddings: Embeddings for AP-DI data (if applicable).
    """

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

def compute_and_cache_embeddings(
    corpus: list[str],
    tokenizer: AutoTokenizer, 
    model: AutoModel, 
    path: str
) -> np.ndarray:
    """
    Compute embeddings for a given corpus using the provided tokenizer and SimCSE model,
    save them to a file, and return the embeddings.

    Args:
        corpus: List of sentences.
        tokenizer: Tokenizer for the embedding model.
        model: Pretrained model used to compute embeddings.
        path: File path to save the computed embeddings (.npy).

    Returns:
        Numpy array of computed embeddings.
    """

    embeddings = compute_embeddings(corpus, tokenizer, model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)
    return embeddings

def top_k(
    sorted_indices: list[int], 
    dataset: list[dict], 
    k: int, 
    sources: str
) -> list[int]:
    """
    Select the top-k unique (text, aspect) pairs based on provided sorted indices.

    Args:
        sorted_indices: Ranked indices of dataset entries based on either bm25 or SimCSE.
        dataset: List of examples, possibly with transformed sentences.
        k: Number of top elements to select.
        sources: Indicates whether parsing should consider transformed sentences (e.g., 'dependent' or 'independent').

    Returns:
        List of selected top-k unique indices.
    """
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

def select_demonstration_indices(
    sentence: str, 
    method: str, 
    num_shots: int, 
    datasets: dict[str,list[dict]] | None, 
    embeddings: dict[str, np.ndarray], 
    sources: str, 
    tokenizer: AutoTokenizer | None = None, 
    sim_model: AutoModel | None = None
) -> dict[str, list[int]]:
    """
    Select demonstration indices using SimCSE or bm25 based similarity.

    Args:
        sentence: Target sentence for which to select demonstrations.
        method: Similarity method ('SimCSE' or 'bm25').
        num_shots: Number of demonstrations to select from each data set.
        datasets: Dictionary mapping dataset versions to data lists.
        embeddings: Dictionary of version -> embeddings.
        sources: String indicating from which source domains the demonstrations are taken.
        tokenizer: Tokenizer for SimCSE (if used).
        sim_model: Model for SimCSE (if used).

    Returns:
        Dictionary mapping version name to selected indices.
    """
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
   

def format_demonstrations(
    indices: list[int], 
    dataset: list[dict], 
    label: str, 
    is_tuple: bool, 
    skip_title: bool = False
) -> str:
    """
    Format selected demonstration examples into a string block for prompt use.

    Args:
        indices: Indices of selected examples from the dataset.
        dataset: Source data set.
        label: Label for the demo block title.
        is_tuple: Whether input format is a tuple-style paraphrased sentence.
        skip_title: Whether to omit the title in the output.

    Returns:
        A formatted string containing demonstrations.
    """
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

def generate_prompt(
    sentence: str, 
    aspects: str, 
    demo_block: str
) -> str:
    """
    Construct a prompt for classification including instruction, demonstrations,
    and test sentence.

    Args:
        sentence: Sentence containing the aspect which is classified.
        aspects: Comma-separated aspects in the sentence that is classified.
        demo_block: Formatted demonstration examples.

    Returns:
        The complete prompt string.
    """

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

def load_dependent_independent_sources(
    train_data: list[dict], 
    demo_method: str, 
    model: str, 
    domain: str, 
    key_groq: str, 
    tokenizer: AutoTokenizer | None, 
    sim_model: AutoModel | None
) -> tuple[list[dict], list[dict], np.ndarray | None, np.ndarray | None]:
    
    """
    Load or generate AV-DI and AV-DD versions of the train data.
    If using SimCSE, compute or load cached embeddings for each version.

    Args:
        train_data: Original training dataset.
        demo_method: Demonstration selection method ('SimCSE' or 'bm25').
        model: Model name .
        domain: Dataset source domain name.
        key_groq: API key for the LLM.
        tokenizer: Tokenizer for SimCSE model.
        sim_model: SimCSE model.

    Returns:
        Tuple of dependent data, independent data, and optionally their embeddings.
    """
    
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
    # Loading models and keys
    load_dotenv()
    simcse_tokenizer, simcse_model = load_simcse_model()
    key_openai = os.getenv("OPENAI_KEY") 
    key_groq = os.getenv("GROQ_KEY")
    key_gemini = os.getenv("GEMINI_KEY")

    # Parse experiment arguments and generate prediction info
    source_domains, target_domains, demos, models, shot_infos, indices = parse_experiment_args()
    test_info = generate_info(
        source_domains=source_domains,
        target_domains=target_domains,
        demos=demos,
        models=models,
        shot_infos=shot_infos,
        indices=indices
    )
   
    # Loop over all experiments, use tqdm to show progress
    for (train_domain, test_domain, demo_method, model_choice, shot_info) in tqdm(test_info):
        # Making paths for the original train and test data
        year_train = 2019 if train_domain == "book" else 2014
        year_test = 2019 if test_domain == "book" else 2014
        train_path = f"data_out/{train_domain}/raw_data_{train_domain}_train_{year_train}.txt"
        test_path = f"data_out/{test_domain}/raw_data_{test_domain}_test_{year_test}.txt"
            
        # Explanation for the current run
        shot_explanation = "" if shot_info["num_shots"] == 0 else f"shots from sources: {', '.join(shot_info['sources'])}"
        print(f"\nRunning {demo_method} with {shot_explanation} ({shot_info['num_shots']}) on {train_domain}→{test_domain}, model: {model_choice}")  

        # Output path
        subdir = get_directory(demo=demo_method, model=model_choice, shot_info=shot_info)
        filepath = get_output_path(source_domain=train_domain, target_domain=test_domain, num_shots=shot_info['num_shots'], subdir=subdir)
        os.makedirs(subdir, exist_ok=True)
       
        # Loading cached results
        results, inference_prompts = load_existing_results(filepath, len(load_txt_data(test_path)))

        include_paraphrased = "paraphrased" in shot_info["sources"]

        # Loading/creating train data, test data, and paraphrased version + embeddings
        train_data, test_data, train_embeddings, paraphrased_train_data, paraphrased_train_embeddings = load_data_and_embeddings(
            train_path, test_path, demo_method, model_choice, train_domain, key_groq, simcse_tokenizer, simcse_model, include_paraphrased
        ) 

        # Creating transformed data for domain-dependent and domain-independent
        if "dependent" in shot_info["sources"]:
            dependent_train_data, independent_train_data, dependent_train_embeddings, independent_train_embeddings = load_dependent_independent_sources(
                train_data, demo_method, model_choice, train_domain, key_groq, simcse_tokenizer, simcse_model
            )
        else:
             dependent_train_data, independent_train_data, dependent_train_embeddings, independent_train_embeddings = None, None, None, None


        # Creating client + Rate limit settings
        openai_client = OpenAI(api_key=key_openai)
        groq_client = Groq(api_key=key_groq)
        client = groq_client

        MAX_REQUESTS_PER_MINUTE = 1000
        REQUEST_WINDOW = 60 
        request_times = deque()

        # Configure data sets used in current experiment
        datasets = {
            "regular": train_data if "regular" in shot_info["sources"] else None,
            "paraphrased": paraphrased_train_data if "paraphrased" in shot_info["sources"] else None,
            "dependent": dependent_train_data if "dependent" in shot_info["sources"] else None,
            "independent": independent_train_data if "independent" in shot_info["sources"] else None
        }

        # Configure embeddings used in current experiment
        all_embeddings = {
            "regular": train_embeddings,
            "paraphrased": paraphrased_train_embeddings,
            "dependent": dependent_train_embeddings,
            "independent": independent_train_embeddings
        }
        # Creating space for where the predictions will come, used to see if a prediction is missing
        while len(results) < len(test_data):
            results.append("{}")

        # Prediction loop
        for i in tqdm(range(len(test_data))):
            # If there is already a prediction, go to the next sample
            if results[i].strip() != "{}":
                continue
            # Info for the current test sample
            sample = test_data[i]
            sentence = sample["text"]
            aspect = sample["aspect"]

            # Get the indices for the most similar samples to the test samples
            demo_indices = select_demonstration_indices(
                sentence=sample["text"],
                method=demo_method,
                num_shots=shot_info["num_shots"],
                datasets = datasets,
                embeddings=all_embeddings,
                tokenizer=simcse_tokenizer, sim_model=simcse_model,
                sources=shot_info['sources']
            )   

            # Information for what goes into the prompt
            FORMAT_CONFIG = {
                "regular":      {"data": train_data, "is_tuple": False, "skip_title": False},
                "paraphrased":  {"data": paraphrased_train_data, "is_tuple": False, "skip_title": False},
                "dependent":    {"data": dependent_train_data, "is_tuple": True,  "skip_title": False},
                "independent":  {"data": independent_train_data, "is_tuple": True,  "skip_title": True}
            }

            # Storage for the demonstrations that go into the prompt
            demo_block_parts = []

            # Getting the demonstrations from the source domains used (e.g. raw, transformations, etc.)
            for source_type in shot_info["sources"]:
                if source_type in demo_indices and source_type in FORMAT_CONFIG:
                    config = FORMAT_CONFIG[source_type]
                    demo_block_parts.append(
                        format_demonstrations(
                            demo_indices[source_type],
                            config["data"],
                            source_type,
                            is_tuple=config["is_tuple"],
                            skip_title=config["skip_title"]
                        )
                    )
            demo_block_combined = "\n".join(demo_block_parts)
            prompt = generate_prompt(sentence, aspect, demo_block_combined)
           
            # Prompt the model used to get a classification for the test sample
            try:
                enforce_rate_limit(request_times, MAX_REQUESTS_PER_MINUTE, REQUEST_WINDOW)
                if model_choice == 'gemma3' or model_choice == 'gemini_flash':
                    output = get_response(prompt, client, model_choice, key_gemini)
                else: 
                    output = get_response(prompt, client, model_choice)
            except Exception as e:
                print(f"Error generating response: {e}")
                output = "{}"
            
            # Store the prediction and the prompt that was used
            results[i] = output 
            inference_prompts.append(prompt)

            # Write the results to the output file
            with open(filepath, 'w') as f:
                json.dump({"results": results, "inference_prompts": inference_prompts}, f, indent=2)

        # Add basic evaluation metrics to the file, not complete as some prediction might be faulty, also does not show all metrics
        metrics = evaluation(test_data, results)
        print(json.dumps(metrics, indent=2))
        with open(filepath, 'w') as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()