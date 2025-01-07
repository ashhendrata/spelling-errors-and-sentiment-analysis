import nltk
import random
import pandas as pd
import math
import os
import numpy as np
from BigramModel import BigramModel
from nltk.tokenize import word_tokenize
import random
import requests  # For downloading GloVe
import zipfile  # For extracting GloVe

random.seed(17)

def setup_glove_vocab():
    """Downloads and processes GloVe vocab"""

    vocab_path = 'vocab/glove_vocab.txt'
    if os.path.exists(vocab_path):
        return vocab_path # If alr exists
            
    os.makedirs('vocab', exist_ok=True)
    
    # Download
    zip_path = 'vocab/glove.6B.zip'
    if not os.path.exists(zip_path):
        print("Downloading GloVe vectors...")
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        response = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
    
    # Processing!!
    print("Extracting vocabulary...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        glove_file = 'glove.6B.50d.txt'
        zip_ref.extract(glove_file, 'vocab')
        
        # Create vocab file :)
        vocab = set()
        with open(f'vocab/{glove_file}', 'r', encoding='utf-8') as f:
            for line in f:
                word = line.split()[0]
                vocab.add(word)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for word in sorted(vocab):
                f.write(word + '\n')
    
    return vocab_path

def get_prior(data: dict) -> dict:
    """ Returns the per-label priors
    """
    line_counts = {}  
    total = 0

    # Line count
    for label, path in data.items():
        file = open(path, 'r')
        lines = file.readlines()
        num_lines = len(lines)
        file.close()
        line_counts[label] = num_lines
        total += num_lines

    # For each class
    res_dict = {}
    for label, count in line_counts.items():
        proportion = count / total
        res_dict[label] = proportion
    
    return res_dict


def get_likelihood(model_mapping: dict, input_file: str, true_label: str, prob_output: str = None) -> pd.DataFrame:
    """ 
    Parameters:
        model_mapping: Dictionary where keys are classes and values are the models trained on the classes
        input_file: the file models should be evaluated on
        true_label: the label of the file that models are evaluated on
        prob_output: Optional path to save token-by-token probabilities
    Returns:
        A Dataframe with the following columns including ikelihood of a sequence is the sum of log probability across all the words in the sequence
    """
    sequence_results = []
    per_token_probs = []
    
    f = open(input_file, 'r')
    text_lines = f.readlines() 
    f.close()

    # Likelihood for each model
    for curr_class, curr_model in model_mapping.items():
        for seq_id, curr_line in enumerate(text_lines):
            tokens = curr_line.strip().split()
            total_prob = 0.0

            # Bigram prob
            for pos in range(1, len(tokens)):
                prev_token = tokens[pos-1]
                curr_token = tokens[pos]
                current_prob = curr_model.getBigramProb((prev_token, curr_token), curr_model.smooth)

                # Sum log for small values
                if current_prob > 0:
                    total_prob += math.log(current_prob)
    
                per_token_probs.append({
                   'sentid': seq_id,
                   'token': curr_token,
                   'prev_token': prev_token, 
                   'bigram': f"{prev_token} {curr_token}",
                   'position': pos,
                   'prob': current_prob,
                   'model (pos/neg)': curr_class,
                   'label': true_label,
                   'token_length': len(curr_token),
                   'sequence_length': len(tokens)
                })
    
            sequence_results.append({
                'sentid': seq_id,
                'model': curr_class,
                'likelihood': total_prob,
                'class': true_label,
                'sequence_length': len(tokens),
                'text': curr_line.strip()
            })
    
    pd.DataFrame(per_token_probs).to_csv(prob_output, sep='\t', index=False)
    
    return pd.DataFrame(sequence_results)



def get_posterior(likelihood_df: pd.DataFrame, prior_dict: dict) -> pd.DataFrame:
    """Calculate posterior probabilities for each sequence
    """
    result_df = pd.DataFrame(likelihood_df)

    # log of priors
    result_df['prior'] = result_df['model'].apply(lambda x: math.log(prior_dict[x]))

    # Combine liklihood and prior
    result_df['posterior'] = result_df['likelihood'] + result_df['prior']
    return result_df


def classify(posterior):
    """
    Parameters:
        Dataframe with posterior probabilities

    Returns: 
        Dataframe where each sentence id is associated with a prediction
    """

    wide_df = posterior.pivot(index=['sentid'], columns=['model'], values='posterior').reset_index()

    predictions = []

    for index, row in wide_df.iterrows():
        if row['positive'] > row['negative']:
            predictions.append('positive')
        else:
            predictions.append('negative')

    wide_df['pred'] = predictions

    result_df = wide_df[['sentid', 'negative', 'positive', 'pred']]

    return result_df


def calc_accuracy(models, eval_dict, prior_dict, k):
    """ Parameters:
        models: keys are classes, values are models trained on data from the class
        eval_dict: keys are classes, values are fpaths to evaluation data where the correct label is the class associated with the key
        prior_dict: keys are classes, values are prior probabilties of the classes
    Returns:
        Float which is the accuracy of the predictions across all classes"""
    
    os.makedirs('results', exist_ok=True)
    
    num_samples = 0
    num_correct = 0
   
    for actual_class, eval_filepath in eval_dict.items():
        # token probabilities!! 
        likelihood_results = get_likelihood(
            models, 
            eval_filepath, 
            actual_class, 
            prob_output=f'predictions/{actual_class}_{k}.tsv'
        )
    
        # posterior and classify
        posterior_df = get_posterior(likelihood_results, prior_dict)
        predictions = classify(posterior_df)
    
        # counting
        if len(predictions) > 0:
            class_correct = (predictions['pred'] == actual_class).sum()
            num_samples += len(predictions)
            num_correct += class_correct
    
    if num_samples == 0:
        return 0.0
    else:
        return num_correct / num_samples
    
def main():
    vocab_path = setup_glove_vocab()
    training = '/home/ahendrata/NLPScholar/nlp_final/processed_data/all_reviews_train.txt'
    
    # For each review type
    review_types = ['imdb', 'industrial', 'clothing']
    k = 0.01  # Best smoothing value based on trying
    
    for review_type in review_types:
        print(f"\n---- {review_type}_reviews ----")
        
        # For each degree of error
        for degree in range(11):  # 0 to 10
            # Set up paths for this degree
            testing = f'/home/ahendrata/NLPScholar/nlp_final/processed_data/{review_type}_reviews/{review_type}_reviews_test_degree_{degree}.txt'
            
            eval_files = {
                'positive': testing.replace('.txt', '_positive.txt'),
                'negative': testing.replace('.txt', '_negative.txt')
            }
            
            class_fname = {
                'positive': training.replace('.txt', '_positive.txt'),
                'negative': training.replace('.txt', '_negative.txt')
            }
            
            class_priors = get_prior(class_fname)
            print(class_priors)

            # Models
            models_dict = {
                "positive": BigramModel(
                    tokenizer=word_tokenize, 
                    vocab_fname=vocab_path, 
                    train_fpath=class_fname['positive'], 
                    mark_ends=True, 
                    smooth=f'add-{k}'
                ),
                "negative": BigramModel(
                    tokenizer=word_tokenize, 
                    vocab_fname=vocab_path, 
                    train_fpath=class_fname['negative'], 
                    mark_ends=True, 
                    smooth=f'add-{k}'
                )
            }
            
            # Calculate accuracies
            acc = calc_accuracy(models_dict, eval_files, class_priors, k)
            print(f"Accuracy for degree {degree}: {acc:.4f}")

'''
    # testing which smoothing value should be used: 0.01 had the highest accuracy
    smooth_values = [0.0, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0]
    res = []
    
    for k in smooth_values:
        # models for both pos and neg
        models_dict = {
            "positive": BigramModel(
                tokenizer=word_tokenize, 
                vocab_fname=vocab_path, 
                train_fpath=class_fname['positive'], 
                mark_ends=True, 
                smooth=f'add-{k}'
            ),
            "negative": BigramModel(
                tokenizer=word_tokenize, 
                vocab_fname=vocab_path, 
                train_fpath=class_fname['negative'], 
                mark_ends=True, 
                smooth=f'add-{k}'
            )
        }
    
        # actual accuracy calcuations
        acc = calc_accuracy(models_dict, eval_files, class_priors, k)
        res.append((k, acc))
        print(f"Accuracy for {k}: {acc:.4f}")
'''
    
main()
    
    
