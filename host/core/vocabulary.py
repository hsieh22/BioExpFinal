import random

class Word:
    def __init__(self, word_id, english, chinese):
        self.id = word_id
        self.english = english
        self.chinese = chinese
        self.state_probs = {
            'relax': 0.0,
            'focus': 0.0,
            'memory': 0.0,
            'stress': 0.0
        }
        self.test_history = []

    def __repr__(self): 
        # define a string representation for the Word object
        return f"[{self.id}] {self.english} ({self.chinese})"

class VocabularyManager:
    def __init__(self, filepath):
        self.words = []
        self.filepath = filepath
      
    def load(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # skip empty lines and comments
                if '\t' in line:
                    eng, ch = line.split('\t')
                elif ',' in line:
                    eng, ch = line.split(',', 1)
                else:
                    raise ValueError(f"cannot parse line: {line}")
                self.words.append(Word(i + 1, eng.strip(), ch.strip()))
        print(f"📚 successfully loaded vocabulary, total : {len(self.words)} vocabularies")

    def shuffle_words(self):
        random.shuffle(self.words)

    def get_word(self, idx):
        if idx < 0 or idx >= len(self.words):
            raise IndexError("Index out of range")
        return self.words[idx]

    def get_all(self):
        return self.words
    
    def size(self):
        return len(self.words)
