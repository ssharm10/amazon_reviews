import unicodedata
import spacy

# Load SpaCy's large English model, disabling components not needed
nlp = spacy.load('en_core_web_lg', disable=["parser", "ner"])

def normalize_text(text):
    """
    Converts special Unicode characters into standard ASCII.
    """
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

def custom_tokenizer(text):
    """
    Tokenizes and lemmatizes text, normalizing Unicode characters,
    removing stopwords, and filtering out short or non-alphabetic tokens.

    Args:
        text (str): Raw text input.

    Returns:
        List[str]: List of cleaned, unique lemmatized tokens.
    """
    normalized_text = normalize_text(text)
    parsed = nlp(normalized_text)

    tokens = [
        token.lemma_.lower()
        for token in parsed
        if token.is_alpha and not token.is_stop and len(token) > 3
    ]

    # Remove duplicates while preserving order
    return list(dict.fromkeys(tokens))