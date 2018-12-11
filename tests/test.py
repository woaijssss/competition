
import pandas as pd
from pandas import DataFrame
import numpy as np
import time

lst = [1, 2, 3, 4, 5, 6, 7, 8]
L = []

df_new = DataFrame()

names = ['a', 'b', 'c', 'e', 'f', 'g', 'h', 'i']
df = DataFrame(np.array(L), columns=names)

for i in range(0, 7770):
	L.append(lst)
#df = DataFrame(np.array(L))

ladd = L.copy()

t1 = time.time()
for i in range(0, 1):
	for j in range(0, len(ladd)):
		L.append(ladd[j])

li = []
for i in range(0, len(df)):
	li.append(list(df.loc[i]))

t2 = time.time()
print(df)

print(t2-t1)