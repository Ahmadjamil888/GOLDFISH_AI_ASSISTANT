import re

# === 1. Load vocabulary ===
def load_vocab(vocab_path="data/vocab.txt"):
    """
    Loads vocabulary from file and returns:
    - vocab: list of tokens
    - word2id: dict mapping token -> index
    - id2word: dict mapping index -> token
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]
    
    if not vocab:
        raise ValueError("❌ Vocabulary file is empty.")

    word2id = {token: idx for idx, token in enumerate(vocab)}
    id2word = {idx: token for token, idx in word2id.items()}

    # Ensure special tokens exist
    for special in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
        if special not in word2id:
            raise ValueError(f"❌ Missing required token in vocab: {special}")

    return vocab, word2id, id2word

# === 2. Normalize & tokenize text ===
def normalize(text):
    """
    Lowercase and tokenize text while preserving common punctuation as separate tokens.
    Example: "hello, world!" -> ["hello", ",", "world", "!"]
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9,.!? ]", "", text)  # keep basic punctuation
    text = re.sub(r"\s+", " ", text).strip()
    # Add space around punctuation so they're treated as separate tokens
    text = re.sub(r"([,.!?])", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()

# === 3. Tokenize text to IDs ===
def tokenize(text, word2id, max_len=50, add_bos=True, add_eos=True):
    """
    Tokenizes input text to list of token IDs.
    Adds optional <BOS> and <EOS>, and pads/truncates to max_len.
    """
    tokens = normalize(text).split()
    unk_id = word2id["<UNK>"]
    token_ids = [word2id.get(token, unk_id) for token in tokens]

    if add_bos:
        token_ids = [word2id["<BOS>"]] + token_ids
    if add_eos:
        token_ids.append(word2id["<EOS>"])

    pad_id = word2id["<PAD>"]
    if len(token_ids) < max_len:
        token_ids += [pad_id] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]

    return token_ids

# === 4. Detokenize IDs to text ===
def detokenize(token_ids, id2word):
    """
    Converts list of token IDs back to string.
    Skips <PAD>, <BOS>, and stops at <EOS>.
    """
    tokens = []
    for idx in token_ids:
        word = id2word.get(idx, "<UNK>")
        if word == "<EOS>":
            break
        if word in {"<PAD>", "<BOS>"}:
            continue
        tokens.append(word)

    # Rejoin punctuation correctly
    output = []
    for token in tokens:
        if token in {",", ".", "!", "?"}:
            if output:
                output[-1] += token
            else:
                output.append(token)
        else:
            output.append(token)
    return " ".join(output)
