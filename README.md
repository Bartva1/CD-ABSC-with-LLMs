# An-LLM-Based-Approach-for-Cross-Domain-ABSC

Code for the paper `An LLM-Based Approach for Cross-Domain Aspect-Based Sentiment Classifcation'

To run the code:

- run raw_data.py for the specified domains to process the XML reviews and create train and test sets
- run get_data_stats.py for the statistics on the data sets after they have been created by running raw_data.py
- run LLMs/classification.py for the ABSC task using LLMs
- run LLMs/evaluation.json.py to get the performance metrics for a certain output file


Other files:

- data_book_hotel.py is a helper file used to get the data sets for books
- data_rest_lapt.py is a helper file used to get the data sets for restaurants and laptops
- load_data.py is a helper file used to get the data sets
- kl_divergence.py to calculate kl divergence and correlation with accuracy results
- LLMs/preprocess_incontext_outputs.py is used to format the output from reasoning models into the correct format
- LLMs/add_metrics_to_json.py can be used to also add the metrics to these files after the format is corrected
- LLMs/transform_data.py is used to transform data using LLMs to be domain invariant

