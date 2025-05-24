# DAWM-LCR-Rot-hop-
Code for the paper `Domain-Adaptive Aspect-Based Sentiment Classification using Masked Domain-Specific Words and Target Position-Aware Attention'

To run the code:

- run raw_data.py for the specified domains to process the XML reviews
- run concat.py for the specified domains to concatenate the existing train and test datasets
- run split_sample.py to create train and test datasets
- run save_data.py to create embeddings and save all necessary variables to load them later
- run file_val.py for a specified file to do hyperparameter tuning
- run file_train.py to train the model

Other files:

- config.py is there for general variable values
- kl_divergence.py to calculate kl divergence and correlation with accuracy results
