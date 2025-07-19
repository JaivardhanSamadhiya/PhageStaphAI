from Bio import Entrez, SeqIO
import pandas as pd
import os

Entrez.email = "jaisamadhiya@gmail.com"
query = "bacteriophage"       # Or more specific search
max_records = 50              # Adjust for your purposes

# A. Search NCBI for IDs
search_handle = Entrez.esearch(db="nucleotide", term=query, retmax=max_records, retmode="xml")
search_record = Entrez.read(search_handle)
phage_ids = search_record["IdList"]

# B. Fetch and parse data for each ID
results = []
for pid in phage_ids:
    try:
        # Try to fetch FASTA for sequence
        fasta_handle = Entrez.efetch(db="nucleotide", id=pid, rettype="fasta", retmode="text")
        seq_records = list(SeqIO.parse(fasta_handle, "fasta"))
        sequence = str(seq_records[0].seq) if seq_records else ""
        
        # Fetch GenBank XML for host field
        gb_handle = Entrez.efetch(db="nucleotide", id=pid, retmode="xml")
        gb_record = Entrez.read(gb_handle)
        phage_name = gb_record[0].get("GBSeq_locus", pid)
        host = "Unknown"
        for feature in gb_record[0].get("GBSeq_feature-table", []):
            if feature.get("GBFeature_key") == "source":
                for qualifier in feature.get("GBFeature_quals", []):
                    if qualifier.get("GBQualifier_name") == "host":
                        host = qualifier.get("GBQualifier_value")
        results.append((phage_name, host, sequence))
    except Exception as e:
        print(f"Error with {pid}: {e}")

df = pd.DataFrame(results, columns=["Phage_ID", "Host", "Sequence"])
df.to_csv("phage_records.csv", index=False)
print(df.head())
print(f"Total records: {len(df)}")
print(f"Number with sequence: {df['Sequence'].apply(bool).sum()}")
