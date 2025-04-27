import pandas as pd
import matplotlib.pyplot as plt

class TitanicEDA:
    """
    A class for performing basic exploratory data analysis on the Titanic dataset.
    """
    def __init__(self, filepath: str):
        """
        Initialize with the path to the Titanic CSV file.
        """
        self.filepath = filepath
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the Titanic dataset into a pandas DataFrame.
        """
        self.df = pd.read_csv(self.filepath)
        return self.df

    def summary_statistics(self) -> pd.DataFrame:
        """
        Generate summary statistics for the dataset.
        Returns a DataFrame with descriptive statistics.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.df.describe(include='all')

    def plot_survival_by_feature(self, feature: str, output_path: str = None) -> None:
        """
        Plot and save the survival rate grouped by a given feature.

        Parameters:
        - feature: Column name to group by (e.g., 'Pclass', 'Sex').
        - output_path: Optional filepath to save the plot. If None, defaults to 'survival_by_<feature>.png'.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        grouped = self.df.groupby(feature)['Survived'].mean()
        plt.figure(figsize=(8, 6))
        grouped.plot(kind='bar')
        plt.title(f'Survival Rate by {feature}')
        plt.ylabel('Survival Rate')
        plt.xlabel(feature)
        plt.ylim(0, 1)
        plt.tight_layout()

        if output_path is None:
            output_path = f'survival_by_{feature.lower()}.png'
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot: {output_path}")

    def plot_survival_by_age(self, bins: list = None, output_path: str = None) -> None:
        """
        Plot and save survival rate by age groups.

        Parameters:
        - bins: List of bin edges for age grouping. If None, uses default bins.
        - output_path: Optional filepath to save the plot. If None, defaults to 'survival_by_age.png'.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        age_df = self.df.dropna(subset=['Age']).copy()
        if bins is None:
            bins = [0, 12, 18, 35, 60, 80]
        age_df['AgeGroup'] = pd.cut(age_df['Age'], bins)
        grouped = age_df.groupby('AgeGroup')['Survived'].mean()

        plt.figure(figsize=(10, 6))
        grouped.plot(kind='bar')
        plt.title('Survival Rate by Age Group')
        plt.ylabel('Survival Rate')
        plt.xlabel('Age Group')
        plt.ylim(0, 1)
        plt.tight_layout()

        if output_path is None:
            output_path = 'survival_by_age.png'
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot: {output_path}")

eda = TitanicEDA('train.csv')
df = eda.load_data()
print(eda.summary_statistics())
eda.plot_survival_by_feature('Pclass')
eda.plot_survival_by_feature('Sex')
eda.plot_survival_by_age()