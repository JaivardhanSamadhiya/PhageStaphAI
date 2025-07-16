import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset_optimized.csv')

# Pie chart for Phage diversity by ID
phage_counts = df['Phage_ID'].value_counts()
phage_counts.plot.pie(autopct='%1.1f%%', startangle=90, figsize=(6,6))
plt.title('Phage Diversity by ID')
plt.ylabel('')
plt.tight_layout()
plt.show()
