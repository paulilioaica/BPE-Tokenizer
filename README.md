# 📝 BPE Tokenizer

A simple implementation of a Byte Pair Encoding (BPE) tokenizer in Python. This tokenizer performs tokenization based on character frequency and merges, which is useful for natural language processing tasks.

## 📚 Features

- Initialize vocabulary based on character frequency
- Perform BPE merge operations
- Encode and decode text using the trained tokenizer
- Handles unknown tokens with `<UNK>` token
- Easy to integrate and extend

## 🚀 Usage

1. **Initialize the Tokenizer**

   ```python
   tokenizer = Tokenizer(num_merges=100)
   ```

2. **Prepare the Text**

   ```python
   text = ...
   ```

3. **Train the Tokenizer**

   ```python
   tokenizer.fit(text)
   ```

4. **Encode Text**

   ```python
   encoded = tokenizer.encode(text)
   print("Encoded:", encoded)
   ```

5. **Decode Text**

   ```python
   decoded = tokenizer.decode(encoded)
   print("Decoded:", decoded)
   ```

## 🌟 How It Works

The tokenizer uses Byte Pair Encoding (BPE) to iteratively merge the most frequent pairs of characters in the text. This reduces the number of tokens needed to represent the text, which can be beneficial for machine learning models.

### Training Process

1. **Initialization**: Start with a vocabulary of all unique characters in the text.
2. **Pair Frequency Calculation**: Find the most frequent pair of symbols in the text.
3. **Merging**: Merge the most frequent pair into a single token.
4. **Vocabulary Update**: Add the new merged token to the vocabulary.
5. **Repeat**: Continue merging until the desired number of merges is reached.

## ✅ Requirements

- Python 3.x

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.
