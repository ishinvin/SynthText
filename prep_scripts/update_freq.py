from collections import Counter
import pickle
import os
import unicodedata

def is_khmer(ch):
    """Return True if character is Khmer script, numeral, or diacritic."""
    code = ord(ch)
    return (
        (0x1780 <= code <= 0x17FF) or  # Basic Khmer
        (0x19E0 <= code <= 0x19FF) or  # Khmer Symbols (includes numerals)
        (0x200C <= code <= 0x200D)     # ZWJ and ZWNJ (used in shaping)
    )

cnt = 0
c = Counter()
filename = os.path.abspath(os.path.join(os.getcwd(), 'data/newsgroup/newsgroup.txt'))
output_filename = os.path.abspath(os.path.join(os.getcwd(), 'data/models/char_freq.cp'))

with open(filename, encoding='utf-8') as f:
    for x in f:
        normalized_line = unicodedata.normalize('NFC', x.strip())
        khmer_chars = [ch for ch in normalized_line if is_khmer(ch)]
        c += Counter(khmer_chars)
        cnt += len(khmer_chars)

print(f"Total Khmer-related characters: {cnt}")

for key in sorted(c):
    c[key] = float(c[key]) / cnt
    print(repr(key), c[key])

d = dict(c)
with open(output_filename, 'wb') as f:
    pickle.dump(d, f)
