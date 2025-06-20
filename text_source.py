import numpy as np

class TextSource(object):
    """
    Provides text for words
    """
    def __init__(self, min_nchar, fn):
        """
        fn : path to file containing text data.
        """
        self.min_nchar = min_nchar
        
        with open(fn,'r') as f:
            self.txt = [l.strip() for l in f.readlines()]

    def is_good(self, txt, f=0.35):
        return [ (len(l)> self.min_nchar) for l in txt ]
    
    def sample_word(self,nline_max,nchar_max,niter=100):
        rand_word = self.txt[np.random.choice(len(self.txt))]         

        iter = 0
        while iter < niter and (not self.is_good([rand_word])[0] or len(rand_word)>nchar_max):
            rand_word = self.txt[np.random.choice(len(self.txt))]
            iter += 1

        if not self.is_good([rand_word])[0] or len(rand_word)>nchar_max:
            return []
        else:
            return rand_word