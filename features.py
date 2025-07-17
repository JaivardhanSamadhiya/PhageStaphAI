import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def compute_kmer_features(sequences, k=3):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k))
    X = vectorizer.fit_transform(sequences)
    kmers = vectorizer.get_feature_names_out()
    return X.toarray(), kmers

if __name__ == "__main__":
    df = pd.read_csv("phage_records.csv")
    X, kmers = compute_kmer_features(df['Sequence'])
    kmer_df = pd.DataFrame(X, columns=[f"kmer_{i}" for i in kmers])
    df_features = pd.concat([df[['Phage_ID', 'Host']], kmer_df], axis=1)
    df_features.to_csv("phage_features.csv", index=False)
