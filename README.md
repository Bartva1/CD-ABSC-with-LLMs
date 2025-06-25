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
Install required packages:


## How to Run
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
python LLMs/evaluation.py
```

Optional: run individual experiment by using config file, currently config_file.json runs all experiments.
``` console
python LLMs/classification.py --config configs/config_file.json
```
Or only for certain domains/models, the example below runs code for the laptop-book source-target pair, using SimCSE for the demonstrations, predictions are created by both models, and for the demonstration configuration, they use the 6-shot raw source domain, 6-shot Hybrid (Aspect-Preserved Domain-Independent + raw source domain), and 0-shot. 

```console
python classification.py \
  --source_domains laptop \
  --target_domains book \
  --demos SimCSE \
  --models gemma,llama4_scout \
  --indices 0,1,4 \
  --shot_infos_path configs/shot_infos.json
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
  

