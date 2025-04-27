import pandas as pd
import matplotlib.pyplot as plt
import time


def timing_decorator(func):
    """
    Decorator to measure execution time of methods.
    Prints the function name and elapsed time.
    """
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} executed in {end - start:.4f} seconds")
        return result
    return wrapper


class SalesDataProcessor:
    """
    Class to process and visualize Supermarket Sales data.
    """

    def __init__(self, filepath: str):
        """
        Load sales data CSV. Expects a 'Date' column parseable to datetime.
        """
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'], format='%m/%d/%Y')
        else:
            raise ValueError("Expected 'Date' column in the dataset")

    @timing_decorator
    def summary_statistics(self) -> pd.DataFrame:
        """
        Return descriptive statistics for numeric columns.
        """
        return self.df.describe()

    @timing_decorator
    def plot_sales_over_time(
        self,
        date_col: str = 'Date',
        sales_col: str = 'Total',
        output_path: str = 'sales_over_time.png'
    ) -> None:
        """
        Plot and save total sales over time aggregated by date.

        :param date_col: Column name for dates
        :param sales_col: Column name for sales values
        :param output_path: Filepath to save the plot
        """
        # Aggregate sales by date
        sales_ts = self.df.groupby(date_col)[sales_col].sum()

        plt.figure(figsize=(10, 6))
        sales_ts.plot()
        plt.title('Total Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Total Sales')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot: {output_path}")

processor = SalesDataProcessor('supermarket_sales.csv')
stats = processor.summary_statistics()
processor.plot_sales_over_time()
