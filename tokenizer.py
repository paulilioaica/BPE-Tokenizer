class Tokenizer:
    def __init__(self, num_merges=100):
        self.vocab = {}
        self.merge_history = []
        self.num_merges = num_merges
        self.token_to_id = {}
        self.id_to_token = {}
        self.trained = False

    def init_vocab(self, words):
        vocab = {}
        for word in words:
            for char in word:
                vocab[char] = vocab.get(char, 0) + 1
        return vocab

    def get_symbol_freq(self, all_words):
        pairs = {}
        for word_tokens in all_words:
            tokens = word_tokens.split()
            for i in range(len(tokens) - 1):
                if tokens[i] == '|' or tokens[i + 1] == '|':
                    continue
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def merge_vocab(self, pair):
        token_a, token_b = pair
        merged_token = token_a + token_b
        self.vocab[merged_token] = 0

    def merge_words(self, all_words, pair):
        token_a, token_b = pair
        merged = token_a + token_b
        pattern = f"{token_a} {token_b}"
        new_words = [word.replace(pattern, merged) for word in all_words]
        return new_words

    def fit(self, text):
        words = text.split()
        all_words = [" ".join(list(word)) for word in words]
        self.vocab = self.init_vocab(words)

        self.merge_lookup = {}

        print("Starting training process...")
        for i in range(self.num_merges):
            pairs = self.get_symbol_freq(all_words)
            if not pairs:
                break

            best_pair, pair_freq = max(pairs.items(), key=lambda x: x[1])

            if best_pair in self.merge_lookup:
                continue

            print(f"Iteration {i + 1}: Merging pair {best_pair} with frequency {pair_freq}")

            self.merge_vocab(best_pair)
            all_words = self.merge_words(all_words, best_pair)
            self.merge_lookup[best_pair] = ''.join(best_pair)

        # Initialize token-to-id mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab.keys(), start=1)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.token_to_id['<UNK>'] = 0
        self.id_to_token[0] = '<UNK>'
        self.trained = True
        print("Training completed.")

    def tokenize(self, text):
        if not self.trained:
            raise ValueError("Tokenizer has not been trained. Call 'fit' with training data first.")

        words = text.split()
        tokenized_text = []

        for word in words:
            word_tokens = ' '.join(list(word))
            # Naive encoding - sort the merge pairs by length in descending order
            sorted_merge_pairs = sorted(self.merge_lookup.keys(), key=len, reverse=True)

            for merge_pair in sorted_merge_pairs:
                merge_pair_str = ' '.join(merge_pair)
                if merge_pair_str in word_tokens:
                    word_tokens = word_tokens.replace(merge_pair_str, self.merge_lookup[merge_pair])

            tokenized_text.extend(word_tokens.split())

        return tokenized_text

    def encode(self, text):
        tokens = self.tokenize(text)
        encoded = [self.token_to_id.get(token, self.token_to_id['<UNK>']) for token in tokens]
        return encoded

    def decode(self, token_ids):
        tokens = [self.id_to_token.get(token_id, '<UNK>') for token_id in token_ids]
        text = ' '.join(tokens).replace('  ', ' ').strip()
        return text

tokenizer = Tokenizer(num_merges=100)
text = "".join(open("input.txt").readlines()).replace(".", "").replace(",", "").lower()
tokenizer.fit(text)
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)
print("Encoded:", encoded)
print("Decoded:", decoded)