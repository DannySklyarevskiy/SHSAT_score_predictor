import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a DataFrame
df = pd.read_csv("data.csv")

# Drop non-numeric columns
df_numeric = df.select_dtypes(include=['number'])

# Calculate correlations for numerical columns
correlations_numeric = df_numeric.corr()["shsat_total_score"].drop("shsat_total_score")

# Plot correlation graphs for numerical columns
for column in correlations_numeric.index:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=column, y="shsat_total_score")
    plt.title(f"Correlation between {column} and SHSAT Total Score")
    plt.xlabel(column)
    plt.ylabel("SHSAT Total Score")
    plt.grid(True)
    plt.show()

# Plot box plots for categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x=column, y="shsat_total_score")
    plt.title(f"Distribution of {column} with respect to SHSAT Total Score")
    plt.xlabel(column)
    plt.ylabel("SHSAT Total Score")
    plt.grid(True)
    plt.show()
