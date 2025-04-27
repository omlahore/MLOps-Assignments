import pandas as pd

class ChunkIterator:
    """
    Iterator to stream a CSV dataset in chunks and compute basic stats for each chunk.
    """
    def __init__(self, filepath: str, chunksize: int = 1000):
        """
        Initialize the iterator with a CSV file path and chunk size.

        :param filepath: Path to the CSV file
        :param chunksize: Number of rows per chunk
        """
        self.filepath = filepath
        self.chunksize = chunksize
        self._iterator = pd.read_csv(self.filepath, chunksize=self.chunksize, iterator=True)

    def __iter__(self):
        return self

    def __next__(self):
        """
        Read the next chunk and return a tuple of (chunk DataFrame, stats DataFrame).
        Raises StopIteration when no more data.
        """
        chunk = next(self._iterator)  # may raise StopIteration
        stats = self.calculate_stats(chunk)
        return chunk, stats

    def calculate_stats(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and return basic statistics (count, mean, std, min, 25%, 50%, 75%, max) for numeric columns in the chunk.

        :param chunk: DataFrame chunk
        :return: DataFrame of descriptive statistics
        """
        return chunk.describe()

iterator = ChunkIterator('Mall_Customers.csv', chunksize=200)
for i, (chunk_df, chunk_stats) in enumerate(iterator):
    print(f"Chunk {i+1} statistics:")
    print(chunk_stats)
