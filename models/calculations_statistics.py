import numpy as np
import pandas as pd

data150 = pd.read_csv('c.txt', sep=" ", names=["p", "c"])
print(data150.head())

data50 = pd.read_csv('c50.txt', sep=" ", names=["p", "c"])
print(data50.head())