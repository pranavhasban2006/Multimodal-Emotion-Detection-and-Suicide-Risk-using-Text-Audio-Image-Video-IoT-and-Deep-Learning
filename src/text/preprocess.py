import os
import re
import json
import string
import numpy as np
from typing import Dict, List, Tuple, Any


class TextPreprocessor:
    def __init__(self, vocab_path: str = "models/vocab.json", max_seq_len: int = 50):
        self.vocab_path = vocab_path
        self.max_seq_len = max_seq_len
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.load_vocabulary()

        # Clinical linguistic lexicons
        self.first_person_pronouns = {"i", "me", "my", "myself", "mine"}
        self.crisis_lexicon = {
            "kill", "suicide", "die", "hopeless", "end", "death", "pain", "goodbye",
            "cutting", "overdose", "hanging", "depressed", "worthless", "tired", "bleeding"
        }

    def load_vocabulary(self):
        """Loads vocabulary mapping from disk if present."""
        if os.path.exists(self.vocab_path):
            try:
                with open(self.vocab_path, "r", encoding="utf-8") as f:
                    self.vocab = json.load(f)
                print(f"[TextPreprocessor] Loaded vocabulary of size {len(self.vocab)} from {self.vocab_path}")
            except Exception as e:
                print(f"[TextPreprocessor] Error loading vocab: {e}. Using empty default.")

    def save_vocabulary(self):
        """Saves current vocabulary mapping to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(self.vocab_path)), exist_ok=True)
        try:
            with open(self.vocab_path, "w", encoding="utf-8") as f:
                json.dump(self.vocab, f, indent=4)
            print(f"[TextPreprocessor] Saved vocabulary of size {len(self.vocab)} to {self.vocab_path}")
        except Exception as e:
            print(f"[TextPreprocessor] Error saving vocab: {e}")

    def build_vocabulary(self, corpus: List[str], max_vocab_size: int = 5000):
        """Builds vocabulary mapping from a training corpus."""
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        word_counts = {}
        for text in corpus:
            cleaned = self.anonymize_and_clean(text)
            words = cleaned.lower().split()
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1

        # Sort by frequency and truncate
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for w, _ in sorted_words[:max_vocab_size]:
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)
        
        self.save_vocabulary()

    def anonymize_and_clean(self, text: str) -> str:
        """
        Strips usernames, handles, subreddit links, and URLs to ensure PII anonymization before disk storage.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Strip URLs
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        # Strip Reddit handles/subreddits e.g. /u/username, r/SuicideWatch
        text = re.sub(r'/?u/\w+', '[USER]', text)
        text = re.sub(r'/?r/\w+', '[SUBREDDIT]', text)
        # Strip general social handles (@username)
        text = re.sub(r'@\w+', '[HANDLE]', text)
        
        return text.strip()

    def extract_linguistic_features(self, raw_text: str) -> Dict[str, float]:
        """
        Extracts clinically-informed structural and lexical linguistic indicators:
        - Affect Intensity: Rates of exclamation marks (!), question marks (?), and ellipses (...)
        - Cognitive Focus: First-person singular pronoun rate (associated with self-focus & depression)
        - Crisis Intensity: Word matches from the suicide crisis lexicon
        """
        if not raw_text:
            return {
                "exclamation_rate": 0.0, "question_rate": 0.0, "ellipsis_rate": 0.0,
                "first_person_rate": 0.0, "crisis_lexicon_rate": 0.0
            }

        total_chars = len(raw_text)
        words = raw_text.lower().split()
        total_words = len(words)

        if total_words == 0:
            total_words = 1
        if total_chars == 0:
            total_chars = 1

        # Count punctuations
        exclamations = raw_text.count('!')
        questions = raw_text.count('?')
        # Match dot sequences representing hesitation/trailing thoughts e.g. "..." or "...."
        ellipses = len(re.findall(r'\.{3,}', raw_text))

        # Pronouns & lexicon counts
        first_person_count = sum(1 for w in words if w in self.first_person_pronouns)
        crisis_count = sum(1 for w in words if w in self.crisis_lexicon)

        return {
            "exclamation_rate": float(exclamations / total_chars),
            "question_rate": float(questions / total_chars),
            "ellipsis_rate": float(ellipses / total_chars),
            "first_person_rate": float(first_person_count / total_words),
            "crisis_lexicon_rate": float(crisis_count / total_words)
        }

    def text_to_sequence(self, text: str) -> np.ndarray:
        """
        Converts text into standard fixed-length token-ID padded sequences.
        Used as the input vector layer for the Bi-LSTM model.
        """
        cleaned = self.anonymize_and_clean(text).lower()
        # Strip standard punctuation for indexing, retaining clean words
        translator = str.maketrans('', '', string.punctuation)
        words = cleaned.translate(translator).split()

        sequence = [self.vocab.get(w, self.vocab["<UNK>"]) for w in words]
        
        # Apply padding or truncation
        if len(sequence) < self.max_seq_len:
            sequence = sequence + [self.vocab["<PAD>"]] * (self.max_seq_len - len(sequence))
        else:
            sequence = sequence[:self.max_seq_len]
            
        return np.array(sequence, dtype=np.int32)

    def text_to_hashing_features(self, text: str) -> np.ndarray:
        """
        Implements a deterministic L2-normalized hashing-trick vectorizer.
        Used as the input features layer for the Random Forest baseline model.
        """
        cleaned = self.anonymize_and_clean(text).lower()
        translator = str.maketrans('', '', string.punctuation)
        words = cleaned.translate(translator).split()

        features = np.zeros(50)
        for w in words:
            idx = abs(hash(w)) % 50
            features[idx] += 1.0
            
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features

    @staticmethod
    def train_test_split_by_user(df_posts: List[Dict[str, Any]], test_size: float = 0.2, random_seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Performs user-level train/test splitting over a social post dataset.
        Guarantees that posts belonging to the same user ID are never split across 
        training and testing splits, preventing data leakage and metric inflation.
        """
        np.random.seed(random_seed)
        
        # Group posts by user ID
        user_to_posts = {}
        for post in df_posts:
            user_id = post.get("user_id", "anonymous_user")
            if user_id not in user_to_posts:
                user_to_posts[user_id] = []
            user_to_posts[user_id].append(post)

        unique_users = list(user_to_posts.keys())
        np.random.shuffle(unique_users)

        split_idx = int(len(unique_users) * (1 - test_size))
        train_users = set(unique_users[:split_idx])

        train_set = []
        test_set = []

        for user_id, posts in user_to_posts.items():
            if user_id in train_users:
                train_set.extend(posts)
            else:
                test_set.extend(posts)

        print(f"[Split] Users - Train: {len(train_users)}, Test: {len(unique_users) - len(train_users)}")
        print(f"[Split] Posts - Train: {len(train_set)}, Test: {len(test_set)}")
        
        return train_set, test_set
