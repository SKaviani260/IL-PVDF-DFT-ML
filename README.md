# IL-PVDF-DFT-ML

# Benchmarking Machine Learning Models for Band Gap Prediction
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example performance dictionary
results = {
    'RandomForest': {'R2 Score': 0.91, 'MAE': 0.12, 'RMSE': 0.18},
    'SVR': {'R2 Score': 0.84, 'MAE': 0.19, 'RMSE': 0.25},
    'GradientBoosting': {'R2 Score': 0.89, 'MAE': 0.14, 'RMSE': 0.21},
    'KNN': {'R2 Score': 0.78, 'MAE': 0.22, 'RMSE': 0.29}
}

# Prepare DataFrame
df_results = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})

# Set minimal style (no grid)
sns.set(style="white", font_scale=1.3)

# Define a function to format all three plots consistently
def plot_metric(metric, palette, ylabel):
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x='Model', y=metric, data=df_results, palette=palette, edgecolor='black')
    plt.title(f'Model Comparison: {metric}', fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel('')

    # Remove grid and add clean outer frame
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.show()

# Plot each metric
plot_metric('R2 Score', 'viridis', 'R²')
plot_metric('MAE', 'magma', 'MAE (eV)')
plot_metric('RMSE', 'coolwarm', 'RMSE (eV)')

# Descriptor Analysis and Band Gap Distribution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
df = pd.read_excel("Dataset.xlsx")

# Rename for convenience
df.rename(columns={
    'EH(eV)': 'HOMO',
    'EL(eV)': 'LUMO',
    'Eg (eV)': 'BandGap',
    'IP (eV)': 'IP',
    'EA (eV)': 'EA'
}, inplace=True)

# Feature and target
features = ['HOMO', 'LUMO', 'IP', 'EA']
target = 'BandGap'

X = df[features]
y = df[target]

plt.figure(figsize=(8, 6))
sns.heatmap(df[features + ['BandGap']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Quantum Descriptors")
plt.show()


plt.figure(figsize=(7, 5))
sns.histplot(df['BandGap'], kde=True, bins=15, color='skyblue')
plt.title('Distribution of Band Gap')
plt.xlabel('Band Gap (eV)')
plt.ylabel('Frequency')
plt.show()
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['HOMO'], df['EA'], df['BandGap'], c=df['BandGap'], cmap='viridis')
ax.set_xlabel('HOMO')
ax.set_ylabel('EA')
ax.set_zlabel('Band Gap')
ax.set_title('3D View of Band Gap by HOMO & EA')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.xlabel("Actual Band Gap")
plt.ylabel("Predicted Band Gap")
plt.title("Actual vs Predicted Band Gap (Random Forest)")
plt.grid(False)
plt.show()

# Influence of Cation Class on Electronic Properties
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("Dataset.xlsx")

# Rename columns for consistency
df.rename(columns={
    'EH(eV)': 'HOMO',
    'EL(eV)': 'LUMO',
    'Eg (eV)': 'BandGap',
    'IP (eV)': 'IP',
    'EA (eV)': 'EA'
}, inplace=True)

# Map detailed cations to general cation group
cation_map = {
    'imidazolium': 'Imidazolium',
    'pyrrolidinium': 'Pyrrolidinium',
    'piperidinium': 'Piperidinium',
    'ammonium': 'Ammonium',
    'sulfonium': 'Sulfonium',
    'phosphonium': 'Phosphonium',
    'pyridinium': 'Pyridinium',
    'pyrazolium': 'Pyrazolium',
    'guanidinium': 'Guanidinium',
    'morpholinium': 'Morpholinium',
    'thiazolium': 'Thiazolium',
    # Add more if necessary
}

def classify_cation(name):
    name = name.lower()
    for key in cation_map:
        if key in name:
            return cation_map[key]
    return 'Other'

# Apply classification
df['Cation_group'] = df['Cation'].apply(classify_cation)

# Keep major cation groups only
top_groups = df['Cation_group'].value_counts().nlargest(10).index
df_top = df[df['Cation_group'].isin(top_groups)]

# Set plot style
sns.set(style="whitegrid", font_scale=1.2)

# Plotting function for boxplots by cation group
def plot_box_by_group(y, ylabel):
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df_top, x='Cation_group', y=y, palette='Set2')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.title(f'{ylabel} Distribution by Cation Group')
    plt.xlabel('Cation Group')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

# Generate each figure
plot_box_by_group('BandGap', 'Band Gap (eV)')         # Fig. 4a
plot_box_by_group('EA', 'Electron Affinity (eV)')     # Fig. 4b
plot_box_by_group('IP', 'Ionization Potential (eV)')  # Fig. 4c
plot_box_by_group('HOMO', 'HOMO Energy (eV)')         # Fig. 4d
plot_box_by_group('LUMO', 'LUMO Energy (eV)')         # Fig. 4e

# Correlation of Band Gap with Frontier Orbital and Quantum Descriptors
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid", font_scale=1.2)

# Scatter plot: Band Gap vs IP
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='IP', y='BandGap', hue='Cation_group', palette='tab10', s=80)

plt.title("Band Gap vs Ionization Potential (IP)")
plt.xlabel("Ionization Potential (eV)")
plt.ylabel("Band Gap (eV)")
plt.legend(title='Cation Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Minimal plot style without grid
sns.set(style="white", font_scale=1.2)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x='IP',
    y='BandGap',
    hue='Cation_group',
    palette='tab10',
    s=80,
    edgecolor='black'
)

plt.title("Band Gap vs Ionization Potential (IP)", fontsize=14)
plt.xlabel("Ionization Potential (eV)", fontsize=12)
plt.ylabel("Band Gap (eV)", fontsize=12)
plt.legend(title='Cation Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Use white style (no grid)
sns.set(style="white", font_scale=1.2)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x='LUMO',
    y='BandGap',
    hue='Cation_group',
    palette='Set2',
    s=80,
    edgecolor='black'
)

plt.title("Band Gap vs LUMO Energy", fontsize=14)
plt.xlabel("LUMO Energy (eV)", fontsize=12)
plt.ylabel("Band Gap (eV)", fontsize=12)
plt.legend(title='Cation Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Set style: no internal grid, clean white background
sns.set(style="white", font_scale=1.2)

# Create scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x='HOMO',
    y='BandGap',
    hue='Cation_group',
    palette='Dark2',
    s=80,
    edgecolor='black'
)

# Outer box only, no internal grid
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
ax.grid(False)

# Labels and formatting
plt.title("Band Gap vs HOMO Energy", fontsize=14)
plt.xlabel("HOMO Energy (eV)", fontsize=12)
plt.ylabel("Band Gap (eV)", fontsize=12)

# Legend formatting
plt.legend(title='Cation Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Set clean white background, no grid
sns.set(style="white", font_scale=1.2)

# Create the scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x='EA',
    y='BandGap',
    hue='Cation_group',
    palette='Set1',
    s=80,
    edgecolor='black'
)

# Draw outer box only
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
ax.grid(False)

# Labels and title
plt.title("Band Gap vs Electron Affinity (EA)", fontsize=14)
plt.xlabel("Electron Affinity (eV)", fontsize=12)
plt.ylabel("Band Gap (eV)", fontsize=12)

# Legend
plt.legend(title='Cation Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Feature Importance Analysis Across ML Models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sample data
features = ['EH', 'EL', 'Eg', 'IP', 'EA']
rf_importance = [0.25, 0.15, 0.30, 0.20, 0.10]
gb_importance = [0.22, 0.18, 0.28, 0.21, 0.11]
svr_importance = [0.20, 0.20, 0.25, 0.25, 0.10]
knn_importance = [0.18, 0.22, 0.26, 0.19, 0.15]

# Prepare figure
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Fig. 6. Feature importance comparison across ML models", fontsize=14)

# Panel (a)
sns.barplot(x=features, y=rf_importance, palette='Blues_d', ax=axs[0, 0])
axs[0, 0].set_title("(a) Random Forest")
axs[0, 0].set_ylabel("Importance")
axs[0, 0].grid(False)

# Panel (b)
sns.barplot(x=features, y=gb_importance, palette='Greens_d', ax=axs[0, 1])
axs[0, 1].set_title("(b) Gradient Boosting")
axs[0, 1].set_ylabel("")
axs[0, 1].grid(False)

# Panel (c)
sns.barplot(x=features, y=svr_importance, palette='Oranges_d', ax=axs[1, 0])
axs[1, 0].set_title("(c) SVR (approx.)")
axs[1, 0].set_ylabel("Importance")
axs[1, 0].grid(False)

# Panel (d)
sns.barplot(x=features, y=knn_importance, palette='Purples_d', ax=axs[1, 1])
axs[1, 1].set_title("(d) KNN (dummy importance)")
axs[1, 1].set_ylabel("")
axs[1, 1].grid(False)

# Layout adjustments
for ax in axs.flat:
    ax.set_xlabel("Features")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



