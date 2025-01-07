import pandas as pd
import json
import random
import os

def format_movie_reviews_for_evaluate(fname: str) -> pd.DataFrame: 
    """ Format the movie reviews data
    """
    data = []

    with open(fname, 'r', encoding='utf-8') as file:
        
        df = pd.read_csv(file)

        for idx, row in df.iterrows():
            text = row['review'].replace('<br />', ' ').strip()
            sentiment = row['sentiment']

            data.append({
                'id': idx,
                'text': text,
                'sentiment': sentiment
            })

    formatted_df = pd.DataFrame(data)
    return formatted_df


def format_json_reviews_for_evaluate(fname: str) -> pd.DataFrame: 
    """ Format the reviews data for (1) clothing, shoes, and jewelry and (2) industrial and scientific 
    """
    data = []
    try:
        with open(fname, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                line = line.strip()
                if not line:
                    continue
                try:
                    review = json.loads(line)
                    review_text = review.get("reviewText")
                    overall = review.get("overall")

                    # Don't if reviewText is missing (no review atached)
                    if not review_text:
                        continue

                    # Sentiment based on overall rating
                    if overall in [1.0, 2.0]:
                        sentiment = "negative"
                    elif overall == 3.0:
                        sentiment = "neutral"
                    elif overall in [4.0, 5.0]:
                        sentiment = "positive"
                    else:
                        continue

                    data.append({
                        "id": idx,
                        "text": review_text.strip(),
                        "sentiment": sentiment
                    })
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {idx + 1}: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"File not found: {fname}")
        return pd.DataFrame(columns=['id', 'text', 'sentiment'])
    
    formatted_df = pd.DataFrame(data)
    return formatted_df

'''
def format_financial_news_for_evaluate(fname: str) -> pd.DataFrame:
    """ Format the financial news data"""
    df = pd.read_csv(fname, header=None, names=["sentiment", "text"], encoding='latin-1')

    data = []

    for idx, row in df.iterrows():
        sentiment = row['sentiment'].strip().lower()
        text = row['text'].strip()

        # Skip rows with missing text or sentiment
        if not sentiment or not text:
            continue

        data.append({
            "id": idx,
            "text": text,
            "sentiment": sentiment
        })

    formatted_df = pd.DataFrame(data)
    return formatted_df 
'''


def inject_spelling_errors(formatted_df: pd.DataFrame, degree: int) -> pd.DataFrame:
    """
    Inject spelling errors into the `text` column
    
    Error types include:
    - Character substitution
    - Transposition
    - Omission
    - Duplication
    - Scrambling
    """
    def introduce_error(word, severity):
        if len(word) < 2:
            return word

        # Types explained in paper
        error_type = random.choice(["substitution", "transposition", "omission", "duplication", "scramble"])
        
        if error_type == "substitution":
            idx = random.randint(0, len(word) - 1)
            replacement = random.choice("abcdefghijklmnopqrstuvwxyz")
            word = word[:idx] + replacement + word[idx + 1:]
        
        elif error_type == "transposition" and len(word) > 2:
            idx = random.randint(0, len(word) - 2)
            word = word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]
        
        elif error_type == "omission":
            idx = random.randint(0, len(word) - 1)
            word = word[:idx] + word[idx + 1:]
        
        elif error_type == "duplication":
            idx = random.randint(0, len(word) - 1)
            word = word[:idx] + word[idx] * 2 + word[idx + 1:]
        
        elif error_type == "scramble" and severity == 3:
            word = ''.join(random.sample(word, len(word)))
        
        # For extreme, apply multiple modifications
        if severity == 3 and random.random() < 0.5:
            word = introduce_error(word, severity)
        elif severity == 2 and random.random() < 0.2:
            word = introduce_error(word, severity)
        
        return word

    def add_errors_to_text(text, degree):
        words = text.split()
        
        if degree == 0:
            num_words_to_modify = 0  # 0%
        elif degree == 1:
            num_words_to_modify = max(1, int(len(words) * 0.1))  # 10%
        elif degree == 2:
            num_words_to_modify = max(1, int(len(words) * 0.2))  # 20%
        elif degree == 3:
            num_words_to_modify = max(1, int(len(words) * 0.3))  # 30%
        elif degree == 4:
            num_words_to_modify = max(1, int(len(words) * 0.4))  # 40%
        elif degree == 5:
            num_words_to_modify = max(1, int(len(words) * 0.5))  # 50%
        elif degree == 6:
            num_words_to_modify = max(1, int(len(words) * 0.6))  # 60%
        elif degree == 7:
            num_words_to_modify = max(1, int(len(words) * 0.7))  # 70%
        elif degree == 8:
            num_words_to_modify = max(1, int(len(words) * 0.8))  # 80%
        elif degree == 9:
            num_words_to_modify = max(1, int(len(words) * 0.9))  # 90%
        else:  # degree 10
            num_words_to_modify = len(words)  # 100%

        # Note:
        # Degree controls % of words modified
        # Severity controls how badly each word is modified
        if degree <= 3:
            severity = 1
        elif degree <= 7:
            severity = 2
        else:
            severity = 3
            
        for _ in range(num_words_to_modify):
            idx = random.randint(0, len(words) - 1)
            words[idx] = introduce_error(words[idx], severity)
        
        return " ".join(words)

    modified_df = formatted_df.copy()
    modified_df["text"] = modified_df["text"].apply(lambda x: add_errors_to_text(x, degree))
    return modified_df


def create_all_reviews(movie_df: pd.DataFrame, industrial_df: pd.DataFrame, clothing_df: pd.DataFrame, output_path: str):
    """
    Combines all reviews from different sources into a single file.
    Only includes positive and negative sentiments.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Process and add source
    def process_df(df, source):
        processed = df[df['sentiment'].isin(['positive', 'negative'])][['text', 'sentiment']]
        processed['source'] = source
        return processed
    
    all_dfs = [
        process_df(movie_df, 'movie'),
        process_df(industrial_df, 'industrial'),
        process_df(clothing_df, 'clothing')
    ]
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save
    combined_df.to_csv(output_path, sep='\t', index=False, header=False)

def create_train_test_splits(input_path: str, train_path: str, test_path: str):
    """
    Creates balanced train and test splits from the combined reviews file
    Takes exactly 20,000 reviews from each category (10,000 positive, 10,000 negative) for training, and another 20,000 (10,000 positive, 10,000 negative) for testing
    """

    df = pd.read_csv(input_path, sep='\t', names=['text', 'sentiment', 'source'])
    
    train_dfs = []
    test_dfs = []
    
    # Process each source separately to ensure equal
    for source in ['movie', 'industrial', 'clothing']:
        source_df = df[df['source'] == source].copy()
        
        # Split
        pos_df = source_df[source_df['sentiment'] == 'positive']
        neg_df = source_df[source_df['sentiment'] == 'negative']
        
        # Shuffle
        pos_df = pos_df.sample(frac=1, random_state=42).reset_index(drop=True)
        neg_df = neg_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Training
        train_pos = pos_df[:10000]
        train_neg = neg_df[:10000]
        
        # Testing
        test_pos = pos_df[10000:20000]
        test_neg = neg_df[10000:20000]
        
        # Combine positive and negative
        train_dfs.append(pd.concat([train_pos, train_neg]))
        test_dfs.append(pd.concat([test_pos, test_neg]))
    
    # Combine and shuffle the final
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Text and sentiment columns only
    train_df[['text', 'sentiment']].to_csv(train_path, sep='\t', index=False, header=False)
    test_df[['text', 'sentiment']].to_csv(test_path, sep='\t', index=False, header=False)

def split_by_sentiment(input_path: str, pos_output: str, neg_output: str):
    """
    Splits a review file into positive and negative sentiment files
    """
    df = pd.read_csv(input_path, sep='\t', names=['text', 'sentiment'])
    
    # Positive reviews
    pos_df = df[df['sentiment'] == 'positive']
    pos_df.to_csv(pos_output, sep='\t', index=False, header=False)
    
    # Negative reviews
    neg_df = df[df['sentiment'] == 'negative']
    neg_df.to_csv(neg_output, sep='\t', index=False, header=False)

def process_all_data(movie_df: pd.DataFrame, industrial_df: pd.DataFrame, hotel_df: pd.DataFrame):
    """
    Function to create all required files
    """
    os.makedirs('processed_data', exist_ok=True)
    
    # Paths
    all_reviews_path = 'processed_data/all_reviews.txt'
    train_path = 'processed_data/all_reviews_train.txt'
    test_path = 'processed_data/all_reviews_test.txt'
    train_pos_path = 'processed_data/all_reviews_train_positive.txt'
    train_neg_path = 'processed_data/all_reviews_train_negative.txt'
    test_pos_path = 'processed_data/all_reviews_test_positive.txt'
    test_neg_path = 'processed_data/all_reviews_test_negative.txt'
    
    # Combined file
    create_all_reviews(movie_df, industrial_df, hotel_df, all_reviews_path)
    
    # Train/test splits
    create_train_test_splits(all_reviews_path, train_path, test_path)
    
    # Split train and test by sentiment
    split_by_sentiment(train_path, train_pos_path, train_neg_path)
    split_by_sentiment(test_path, test_pos_path, test_neg_path)
    
def create_error_test_files(df: pd.DataFrame, source_name: str):
    """
    Creates test files with different degrees of spelling errors for a specific source
    """
    output_dir = f'processed_data/{source_name}_reviews'
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for positive and negative sentiments only
    df = df[df['sentiment'].isin(['positive', 'negative'])]
    
    # Test set
    test_df = df.iloc[20000:40000].copy()
    
    # Files for each degree of error
    for degree in range(11):  # 0 to 10
        # Injecting spelling errors
        error_df = inject_spelling_errors(test_df, degree)
        
        # Save
        output_path = f'{output_dir}/{source_name}_reviews_test_degree_{degree}.txt'
        error_df[['text', 'sentiment']].to_csv(output_path, sep='\t', index=False, header=False)
        pos_df = error_df[error_df['sentiment'] == 'positive']
        neg_df = error_df[error_df['sentiment'] == 'negative']
        
        pos_path = f'{output_dir}/{source_name}_reviews_test_degree_{degree}_positive.txt'
        neg_path = f'{output_dir}/{source_name}_reviews_test_degree_{degree}_negative.txt'
        
        pos_df[['text', 'sentiment']].to_csv(pos_path, sep='\t', index=False, header=False)
        neg_df[['text', 'sentiment']].to_csv(neg_path, sep='\t', index=False, header=False)


if __name__ == "__main__":
    # Movie reviews
    movie_df = format_movie_reviews_for_evaluate("final_data/imdb_movie_reviews.csv")
    
    # Equipment reviews
    industrial_df = format_json_reviews_for_evaluate("final_data/industrial_and_scientific_reviews.json")
    
    # Apparel reviews
    clothing_df = format_json_reviews_for_evaluate("final_data/clothing_shoes_jewelry_reviews.json")
    
    # Main data files
    process_all_data(movie_df, industrial_df, clothing_df)
    
    # Error test files for each source
    create_error_test_files(movie_df, 'imdb')
    create_error_test_files(industrial_df, 'industrial')
    create_error_test_files(clothing_df, 'clothing')
    
    # print("we did itttt")


'''
    # -------- preprocessing --------
    
    # testing formatting functions
    print("final_data/imdb_movie_reviews.csv\n")
    test_file = "final_data/imdb_movie_reviews.csv"
    formatted_df = format_movie_reviews_for_evaluate(test_file)
    print(formatted_df.head())

    print("final_data/industrial_and_scientific_reviews.json\n")
    test_file2 = "final_data/industrial_and_scientific_reviews.json"
    formatted_df2 = format_json_reviews_for_evaluate(test_file2)
    print(formatted_df2.head())

    print("final_data/clothing_shoes_jewelry_reviews.json\n")
    test_file2 = "final_data/industrial_and_scientific_reviews.json"
    formatted_df2 = format_json_reviews_for_evaluate(test_file2)
    print(formatted_df2.head())

    for degree in range(11):  # 0 to 10
        modified_df = inject_spelling_errors(formatted_df, degree)
        print(f"\nDegree {degree}:")
        print(modified_df.head(1))

    # print("final_data/financial_news_sentiment.csv\n")
    # test_file3 = "final_data/financial_news_sentiment.csv"
    # formatted_df3 = format_financial_news_for_evaluate(test_file3)
    # print(formatted_df3.head())
'''




