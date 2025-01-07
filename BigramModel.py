import math
import csv
import nltk
import pandas
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def nltk_tokenizer(text):
    return word_tokenize(text)


from collections import defaultdict

class BigramModel:
    def __init__(self, tokenizer, vocab_fname: str, train_fpath: dict, mark_ends: bool, smooth = 'add-0.1'):
        # Initialize model
        self.tokenizer = tokenizer
        self.mark_ends = mark_ends

        # Vocab from file
        self.vocab = self.getVocab(vocab_fname)

        # Tokenize training data
        self.train_dat= self.loadData(tokenizer, train_fpath, mark_ends)

        # Frequencies
        self.bigram_freqs = self.getBigramFreqs()
        #print(self.bigram_freqs)
        self.unigram_freqs = self.getUnigramFreqs()
        #print(self.unigram_freqs)

        self.smooth = smooth


    def getVocab(self, vocab_fname):
        """
        Args: 
            vocab_fname: filepath to vocab file. Each line has a new vocab item

        Returns: 
            A set of all the vocabulary items in the file plus three additional tokens: 
                - [UNK] : to represent words in the text not in the vocab
                - [BOS] : to represent the beginning of sentences. 
                - [EOS] : to represent the end of sentences. 
        """
        with open(vocab_fname, 'r') as f:
            dat = f.read().split('\n')
    
        vocab = set(dat)
        # Tokens added to mark unknown words and start/end of sentences
        vocab.update(['[UNK]', '[BOS]', '[EOS]'])
        return vocab


    def loadData(self, tokenizer, textfname, mark_ends: bool) -> list:
        """
        tokenizer: a function to tokenize a sequence into words.
        textfname: fpath with data.  
        mark_ends: indicates whether sequences should start with [BOS] and end with [EOS]
        Returns: 
        A list of lists where each sublist consists of tokens from each sentence. 
        """
        with open(textfname, 'r', encoding='utf-8') as f:
            text = f.read()

        sentences = nltk.sent_tokenize(text)

        tokenized_sentences = []
        for sentence in sentences:
            tokens = [token.lower() for token in tokenizer(sentence)]
            if mark_ends:
                tokens = ['[BOS]'] + tokens + ['[EOS]']
            else:
                tokens = tokenizer(tokens)

            tokenized_sentences.append(tokens)

        return tokenized_sentences



    def getBigramFreqs(self):
        """
        Args: object
        Returns: 
            dictionary with all bigrams that occur in the text along with frequencies. 
            Each key should be a tuple of strings of the format (first_token, second_token). 
        """
        bigram_freqs = defaultdict(int)
        
        for sentence in self.train_dat:
            # [UNK] for unknown!
            sentence = [token if token in self.vocab else '[UNK]' for token in sentence]
                
            
            for i in range(len(sentence) - 1):
                # Count bigrams
                bigram = (sentence[i], sentence[i+1])
                bigram_freqs[bigram] += 1

        return bigram_freqs

    def getUnigramFreqs(self) -> dict:
        """
        Args: object
        Returns: 
            dictionary with all unigrams that occur in the text along with frequencies. 

        """
        unigram_freqs = defaultdict(int)
        
        for sentence in self.train_dat:
            for word in sentence:
                # Tokens vs regular
                if word[0] != '[':
                    unigram_freqs[word.lower()] += 1
                else:
                    unigram_freqs[word] += 1

        return unigram_freqs

    def getBigramProb(self, bigram, smooth):
        """
        Args:
            bigram: the tuple of the bigram you want the prob of
            smooth: MLE (no smoothing), add-k where you add k to all bigram counts. Returns -1 if invalid smooth is entered.  

        Returns:
            float with prob. 
            Return -1.0 if invalid smoothing value is entered. 
        """

        word1, word2 = bigram

        #print(smooth)
        if smooth == 'MLE':
            # Likelihood estimation
            bigram_count = self.bigram_freqs.get(bigram, 0)
            unigram_count = self.unigram_freqs.get(word1, 0)
            #print(bigram_count / unigram_count)
            return bigram_count / unigram_count
        elif smooth.startswith('add-'):
            # For unseen
            k = float(smooth.split('-')[1])
            #print(k)
            bigram_count = self.bigram_freqs.get(bigram, 0)
            unigram_count = self.unigram_freqs.get(word1, 0)
            vocab_size = len(self.vocab)

            # Smoothed prob
            numerator = bigram_count + k
            denominator = unigram_count + k * vocab_size
            #print(numerator / denominator)
            return numerator / denominator if denominator > 0 else 0
        else:
            #print(-1.0)
            return -1.0


    def evaluate(self, eval_fpath, result_fpath, smooth=None):
        """
        Args: 
            eval_fpath: the path to the evaluation data
            result_fpath: path where predictions will be saved
            smooth: smoothing to be applied

        Output: 
            Creates a file with five columns: 
                sentid: id of the sequence 
                word: second word of the bigram (e.g., if bigram is 'kitten had', word would be had)
                wordpos: position of the word in the sentence
                prob: P(word | prev word)
                surp: - log_2 P(word | prev word)
        """
        smooth = self.smooth
        with open(eval_fpath, 'r', encoding='utf-8') as f:
            text = f.read()

        sentences = nltk.sent_tokenize(text)

        tokenized_sentences = []
        for sentence in sentences:
            tokens = [token.lower() for token in self.tokenizer(sentence)]
            if self.mark_ends:
                tokens = ['[BOS]'] + tokens + ['[EOS]']
            tokenized_sentences.append(tokens)

        # Prob and surp for bigrams
        res = []
        idx = 0
        for tokens in tokenized_sentences:
            for i in range(len(tokens)-1):
                bigram = (tokens[i],tokens[i+1])
                prob = self.getBigramProb(bigram, self.smooth)
                if prob <= 0:
                    # for zero prob
                    suprisal = float('inf')
                else:
                    surprisal = -math.log2(prob)

                res.append({
                    'sentid': idx,
                    'word': tokens[i],
                    'wordpos': i,
                    'prob': prob,
                    'surp': surprisal
                })
            idx += 1
        df = pandas.DataFrame(res)
        df.to_csv(result_fpath, sep='\t', index=False)
                

