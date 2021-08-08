from app.es import index_name_to_meta_node, meta_node_to_index_name


def test_meta_node_to_index_name():
    meta_node = "Gwas"
    embeddings_index_name = meta_node_to_index_name(
        meta_node=meta_node, type="embeddings"
    )
    text_index_name = meta_node_to_index_name(meta_node=meta_node, type="text")
    assert embeddings_index_name == "embeddings-gwas"
    assert text_index_name == "search-global-gwas"


def test_index_name_to_meta_node():
    meta_node = "Gwas"
    embeddings_index_name = "embeddings-gwas"
    text_index_name = "search-global-gwas"
    embeddings_meta_node = index_name_to_meta_node(
        embeddings_index_name, type="embeddings"
    )
    text_meta_node = index_name_to_meta_node(text_index_name, type="text")
    assert embeddings_meta_node == meta_node
    assert text_meta_node == meta_node
