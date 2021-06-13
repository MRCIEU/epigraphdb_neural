import spacy


def preprocess_encode(text: str, nlp_model: spacy.language.Language) -> str:
    text = lower_capitalised(text)
    # use the lemmatised tokens as text
    doc = nlp_model(text)
    clean_text = " ".join([token.lemma_ for token in doc])
    return clean_text


def lower_capitalised(text: str) -> str:
    """
    'Body mass index' -> 'body mass index',
    'FEVER' -> 'FEVER'
    """
    def _f(text: str) -> str:
        try:
            if text[0].isupper() and text[1].islower():
                return text[0].lower() + text[1:]
            else:
                return text
        except:
            return text

    res = " ".join(
        _f(_) for _ in text.split(" ")
    )
    return res
