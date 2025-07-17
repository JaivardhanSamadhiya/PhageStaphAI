from Bio import Entrez
import pandas as pd

def search_phages(query, max_records=100):
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=max_records, retmode="xml")
    record = Entrez.read(handle)
    return record["IdList"]

def fetch_phage_records(ids):
    records = []
    for phage_id in ids:
        handle = Entrez.efetch(db="nucleotide", id=phage_id, retmode="xml")
        record = Entrez.read(handle)
        phage_name = record[0].get("GBSeq_locus", "Unknown")
        seq = record[0].get("GBSeq_sequence", "")
        host = "Unknown"
        for feature in record[0].get("GBSeq_feature-table", []):
            if feature.get("GBFeature_key") == "source":
                for qualifier in feature.get("GBFeature_quals", []):
                    if qualifier.get("GBQualifier_name") == "host":
                        host = qualifier.get("GBQualifier_value")
        records.append((phage_name, host, seq))
    return records

if __name__ == "__main__":
    Entrez.email = "jaisamadhiya@gmail.com"  # Set your email
    query = "bacteriophage"  # Customize your query
    phage_ids = search_phages(query, max_records=100)
    records = fetch_phage_records(phage_ids)
    df = pd.DataFrame(records, columns=["Phage_ID", "Host", "Sequence"])
    df.to_csv("phage_records.csv", index=False)
