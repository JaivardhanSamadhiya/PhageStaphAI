import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset_optimized.csv')
host_counts = df['Host_ID'].value_counts()
host_counts.plot.pie(autopct='%1.1f%%', startangle=90, figsize=(6,6))
plt.title('Host Diversity by ID')
plt.ylabel('')
plt.tight_layout()
plt.show()
