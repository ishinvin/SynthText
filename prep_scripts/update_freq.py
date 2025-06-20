import os
import pickle
from collections import Counter

cnt = 0
filename = os.path.abspath(os.path.join(os.getcwd(), 'data/newsgroup/khmer_words.txt'))
output_filename = os.path.abspath(os.path.join(os.getcwd(), 'data/models/char_freq.cp'))
with open(filename, encoding='utf-8') as f:
    c = Counter()
    for x in f:
        c += Counter(x.strip())
        cnt += len(x.strip())

print(f"Total characters: {cnt}")

for key in sorted(c):
    c[key] = float(c[key]) / cnt
    print(key, c[key])

d = dict(c)
with open(output_filename, 'wb') as f:
    pickle.dump(d, f)