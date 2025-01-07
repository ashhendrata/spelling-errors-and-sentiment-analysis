# Instructions
The data files and processed data were to large to push. However, the following instructions are sufficient to replicate the paper.

## Step 1: Download the Required Datasets

1. [Reviews from the following sections: "Industrial and Scientific" & "Clothing Shoes and Jewelry"](https://nijianmo.github.io/amazon/index.html)
   
2. [Movie reviews](https://ai.stanford.edu/~amaas/data/sentiment/)

## Step 2: Preprocess the Reviews

Run the preprocessing script to prepare the datasets for analysis:

```bash
python preprocession_reviews.py
```

## Step 3: Train and Test the Model

```bash
python train_and_test.py
```
