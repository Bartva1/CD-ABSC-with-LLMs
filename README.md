# An-LLM-Based-Approach-for-Cross-Domain-ABSC

This repository contains the code and data for the research paper:

**"An LLM-Based Approach for Cross-Domain Aspect-Based Sentiment Classification"**

The study explores how large language models (LLMs) can be used to improve cross-domain ABSC tasks by transforming source domain samples into domain-invariant representations.

---
## Repository Structure
```
📦An-LLM-Based-Approach-for-Cross-Domain-ABSC
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
 ┣ 📂SemEval2014
 ┃ ┣ 📜laptop_test_2014.xml
 ┃ ┣ 📜laptop_train_2014.xml
 ┃ ┣ 📜restaurant_test_2014.xml
 ┃ ┗ 📜restaurant_train_2014.xml
 ┣ 📜LICENSE
 ┗ 📜README.md
```

## Requirements
``` console
torch==2.7.1
nltk==3.9.1
numpy==1.26.4
pandas==2.0.3
scikit-learn==1.3.2
python-dotenv==1.1.1
openai==1.91.0
groq==0.28.0
google-generativeai==0.8.5
rank-bm25==0.2.2
tqdm==4.66.4
transformers==4.41.2
matplotlib==3.8.4
seaborn==0.13.2

```

## How to Run
# Optional set-up using conda in VSCode, this is one way to ensure the dependencies between the libraries are correct:
- Press ctrl + shift + p, to select an interpreter and select conda with python 3.12
- Ensure the virtual environment is activated
- Run the following in the terminal (can be opened by ctrl + shift + `)
``` console
 pip install torch==2.7.1 nltk==3.9.1 numpy==1.26.4 pandas==2.0.3 scikit-learn==1.3.2 python-dotenv==1.1.1 openai==1.91.0 groq==0.28.0 google-generativeai==0.8.5 rank-bm25==0.2.2 tqdm==4.66.4 transformers==4.41.2 matplotlib==3.8.4 seaborn==0.13.2
 ```

# Executing the code
Open the terminal.

1. Create Train/Test Sets
``` console
  python data_processing/raw_data.py
```

2. View Dataset Statistics (Optional)

 ``` console
  python data_processing/get_data_stats.py
```
 3. Set up API keys 
To run the experiments, you need to provide API keys for the LLM services. In this project, these are loaded from a .env file in the project root.
For more information on how to get an API key using groq, go to the bottom of the repository.

- create a .env file in the root directory (same level as An-LLM-Based-Approach-for-Cross-Domain-ABSC).
- add the following keys to the file:
```env
  OPENAI_KEY=your_openai_key_here
  GROQ_KEY=your_groq_key_here
  GEMINI_KEY=your_gemini_key_here
 ```
- the client/keys you want to use can be adjusted in LLMs/transformation.py and LLMs/classification.py.
  
4. Run ABSC experiments with LLMs

``` console
  python LLMs/classification.py
```
5. Get the performance measures
``` console
python LLMs/evaluation_json.py
```

Optional: run individual experiment by using config file, currently config_file.json runs all experiments.
``` console
python LLMs/classification.py --config configs/config_file.json
```
Or only for certain domains/models: 
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


## Getting an API key from groq
1. Go to https://groq.com.
2. Click on DEV CONSOLE at the top right of the page.
3. Create an account.
4. Go to API Keys at the top of the page.
5. Click on Create API Key.
6. Store this API key somewhere secure.
  

