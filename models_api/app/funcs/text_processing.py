import spacy


def preprocess_encode(text: str, nlp_model: spacy.language.Language) -> str:
    # use the lemmatised tokens as text
    doc = nlp_model(text)
    clean_text = " ".join([token.lemma_ for token in doc])
    return clean_text
