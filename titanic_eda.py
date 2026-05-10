# Titanic Dataset - Exploratory Data Analysis
# Author: Muhammad Abdullah
# Tools: Python, Pandas, Matplotlib, Seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. LOAD DATA ──────────────────────────────────────────
df = pd.read_csv('titanic.csv')

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nBasic Statistics:")
print(df.describe())

# ── 2. DATA CLEANING ──────────────────────────────────────
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ── 3. SURVIVAL RATE OVERVIEW ─────────────────────────────
print("\nOverall Survival Rate:")
print(df['Survived'].value_counts(normalize=True) * 100)

# ── 4. VISUALIZATIONS ─────────────────────────────────────
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Titanic EDA - Muhammad Abdullah', fontsize=18, fontweight='bold')

# Chart 1 - Survival Count
sns.countplot(x='Survived', data=df, palette='Set2', ax=axes[0,0])
axes[0,0].set_title('Survival Count')
axes[0,0].set_xticklabels(['Did Not Survive', 'Survived'])

# Chart 2 - Survival by Gender
sns.countplot(x='Sex', hue='Survived', data=df, palette='Set1', ax=axes[0,1])
axes[0,1].set_title('Survival by Gender')

# Chart 3 - Survival by Passenger Class
sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set2', ax=axes[0,2])
axes[0,2].set_title('Survival by Passenger Class')

# Chart 4 - Age Distribution
sns.histplot(df['Age'], bins=30, kde=True, color='steelblue', ax=axes[1,0])
axes[1,0].set_title('Age Distribution')

# Chart 5 - Fare Distribution
sns.histplot(df['Fare'], bins=30, kde=True, color='coral', ax=axes[1,1])
axes[1,1].set_title('Fare Distribution')

# Chart 6 - Survival by Age (Boxplot)
sns.boxplot(x='Survived', y='Age', data=df, palette='Set3', ax=axes[1,2])
axes[1,2].set_title('Age vs Survival')
axes[1,2].set_xticklabels(['Did Not Survive', 'Survived'])

plt.tight_layout()
plt.savefig('titanic_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved as titanic_eda.png!")
print("\nEDA Complete! Key Insights:")
print("- Female survival rate much higher than male")
print("- 1st class passengers had better survival chances")
print("- Younger passengers slightly more likely to survive")