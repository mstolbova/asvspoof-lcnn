import pandas as pd


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(
            index=keys,
            columns=["total", "counts", "average"],
            dtype=float,
        )
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data.loc[:, col] = 0.0

    def update(self, key, value, n=1):
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        total = self._data.loc[key, "total"]
        counts = self._data.loc[key, "counts"]
        self._data.loc[key, "average"] = total / counts

    def avg(self, key):
        return self._data.loc[key, "average"]

    def result(self):
        return dict(self._data["average"])

    def keys(self):
        return self._data.index



