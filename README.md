# Improving-Cross-Domain-ABSC-with-LLMs-and-Domain-Invariant-Transformations

This repository contains the code and data for the research paper:

**"Improving Cross-Domain Aspect-Based Sentiment Classification with Large Language Models and Domain-Invariant Transformations"**

The study explores how large language models (LLMs) can be used to improve cross-domain Aspect-Based Sentiment Classification (ABSC) tasks by transforming source domain samples into domain-invariant representations.

---
## Repository Structure
```
📦Improving-Cross-Domain-ABSC-with-LLMs-and-Domain-Invariant-Transformations
 ┣ 📂books
 ┃ ┗ 📜book_reviews_2019.xml
 ┣ 📂data_processing
 ┃ ┣ 📜data_book_hotel.py
 ┃ ┣ 📜data_rest_lapt.py
 ┃ ┣ 📜get_data_stats.py
 ┃ ┣ 📜load_data.py
 ┃ ┗ 📜raw_data.py
 ┣ 📂LLMs
 ┃ ┣ 📜classification.py
 ┃ ┣ 📜evaluation_json.py
 ┃ ┣ 📜transform_data.py
 ┃ ┗ 📜utilities.py
 ┣ 📂 Replication
 ┃ ┣ 📂dawm
 ┃ ┣ 📂lcr
 ┃ ┣ 📜config.py
 ┃ ┣ 📜load_data.py
 ┃ ┗ 📜save_data.py
 ┣ 📂SemEval2014
 ┃ ┣ 📜laptop_test_2014.xml
 ┃ ┣ 📜laptop_train_2014.xml
 ┃ ┣ 📜restaurant_test_2014.xml
 ┃ ┗ 📜restaurant_train_2014.xml
 ┣ 📜LICENSE
 ┗ 📜README.md
```

## Requirements
The required libraries can be found in requirements.txt

## How to Run

To get the main results, follow these steps in a terminal.

1. Get the required libraries
``` console
pip install -r requirements.txt
```

2. Create Train/Test Sets
``` console
python data_processing/raw_data.py
```

3. (Optional) View Dataset Statistics 

 ``` console
 python data_processing/get_data_stats.py
```
4. Set up API keys 
To run the experiments, you need to provide API keys for the LLM services. In this project, these are loaded from a .env file in the project root.
For more information on how to get an API key using groq, go to the bottom of the repository.

- create a .env file in the root directory (same level as Improving-Cross-Domain-ABSC-with-LLMs-and-Domain-Invariant-Transformations).
- add the following keys to the file:
```env
OPENAI_KEY=your_openai_key_here
GROQ_KEY=your_groq_key_here
GEMINI_KEY=your_gemini_key_here
 ```
- the client/keys you want to use can be adjusted in LLMs/transformation.py and LLMs/classification.py.
  
5. Run ABSC experiments with LLMs

``` console
python LLMs/classification.py
```
6. Get the performance measures
``` console
python LLMs/evaluation_json.py
```

7. (Optional) Run individual experiment by using config file, currently config_file.json runs all experiments.
``` console
python LLMs/classification.py --config configs/config_file.json
```
8. (Optional) Run experiments for certain domains/models
The example below runs code for 
- Source domain: laptop (not used)
- Target domain: laptop
- Demonstration strategy: SimCSE (not used)
- Models: Llama4-Scout-17B-16E-Instruct
- Shot info: 0-shot

```console
python LLMs/evaluation_json.py --config configs/config_file.json --source_domains laptop --target_domains laptop --demos SimCSE --models llama4_scout --indices 4 --shot_infos_path configs/shot_infos.json
```



## Description of Key Files

**data_processing/** (adapted from https://github.com/FvdKnaap/DAWM-LCR-Rot-hop-plus-plus)
- raw_data.py: Creates the train and test sets.
- load_data.py: Helper functions for loading the datasets.
- get_data_stats.py: Prints statistics for the processed datasets.
- data_book_hotel.py: Book-specific data creation logic.
- data_rest_lapt.py: Restaurant and laptop data preparation logic.

**LLMs/**
- classification.py: Main script for ABSC prediction using LLMs.
- transform_data.py: Transforms source domain samples using LLMs to improve cross-domain generalization.
- evaluation_json.py: Evaluates predictions to get performance measures.
- utilities.py: Helper functions + functions for post-processing.
- preview_test_info.py: Prints out all the different combinations used in the experiments, more info on the structure of these configurations can be found in the script itself.

## How to Run Replication Code
In this study we also replicate the study: "Domain-Adaptive Aspect-Based Sentiment Classification Using Masked Domain-Specific Words and Target Position-Aware Attention". All the files are adapted from: https://github.com/FvdKnaap/DAWM-LCR-Rot-hop-plus-plus. Here are the steps to follow to get the results from the benchmark models, which are the LCR-Rot-hop++ model and DAWM-Rot-hop++ model.
1. Follow the instruction to run the main code, to get the folder data_out.
2. Open the folder called "Replication", and copy the data_out folder into this folder.
3. Run save_data.py to create BERT embeddings which are used
4. Run lcr/lcr_rot_train.py to get the results for the LCR-Rot-hop++ model
5. Run dawm/bertmasker_lcr_train_cross.py to get the results for the DAWM-LCR-Rot-hop++

## Getting an API key from groq
1. Go to https://groq.com.
2. Click on DEV CONSOLE at the top right of the page.
3. Create an account.
4. Go to API Keys at the top of the page.
5. Click on Create API Key.
6. Store this API key somewhere secure.
  

