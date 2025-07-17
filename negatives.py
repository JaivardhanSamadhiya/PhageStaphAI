from Bio import Entrez
import pandas as pd

Entrez.email = "jaisamadhiya@gmail.com"

def search_phages(query, max_records=100):
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=max_records, retmode="xml")
    record = Entrez.read(handle)
    return record["IdList"]

def fetch_positive_pairs(ids):
    pairs = []
    for phage_id in ids:
        handle = Entrez.efetch(db="nucleotide", id=phage_id, retmode="xml")
        record = Entrez.read(handle)
        phage_name = record[0].get("GBSeq_locus", "Unknown")
        for feature in record[0].get("GBSeq_feature-table", []):
            if feature.get("GBFeature_key") == "source":
                for qualifier in feature.get("GBFeature_quals", []):
                    if qualifier.get("GBQualifier_name") == "host":
                        host = qualifier.get("GBQualifier_value")
                        pairs.append((phage_name, host, 1))  # 1 = positive
    return pairs

# Example: search for all Staphylococcus phages
phage_ids = search_phages("bacteriophage", max_records=100)
positive_pairs = fetch_positive_pairs(phage_ids)
df_pos = pd.DataFrame(positive_pairs, columns=["Phage", "Host", "label"])

import numpy as np

# Step 1: Get unique phages and hosts
phages = df_pos['Phage'].unique()
hosts = df_pos['Host'].unique()

# Step 2: Build all possible pairs
all_pairs = [(p, h) for p in phages for h in hosts]

# Step 3: Remove known positive pairs
positive_set = set(zip(df_pos['Phage'], df_pos['Host']))
neg_pairs = [ (p, h) for (p, h) in all_pairs if (p, h) not in positive_set ]

# Step 4: Sample negatives to match number of positives
np.random.seed(42)
neg_sample_indices = np.random.choice(len(neg_pairs), size=len(df_pos), replace=False)
neg_pairs_sampled = [neg_pairs[i] for i in neg_sample_indices]

# Step 5: Create DataFrame for negatives
df_neg = pd.DataFrame(neg_pairs_sampled, columns=["Phage", "Host"])
df_neg['label'] = 0
df_all = pd.concat([df_pos, df_neg], ignore_index=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
df_all.to_csv("phage_host_interactions.csv", index=False)
