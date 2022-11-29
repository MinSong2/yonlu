from elasticsearch import Elasticsearch
from st_card_component import card_component
from sentence_transformers import SentenceTransformer
import json
import argparse
import streamlit as st


@st.experimental_singleton(show_spinner=False)
def get_embeddings_model(model_name):
    model = SentenceTransformer(model_name)
    return model


@st.experimental_singleton(show_spinner=False)
def get_es_client(host):
    client = Elasticsearch(host)
    return client


def get_sidebar_text(index, vector_dim):
    txt = f"""
    ### Elasticsearch Index Detail
    
    **Index name**: {index}

    **Vector dimension**: {vector_dim}
    ---
    ### How it Works
    The NHS search tool allows us to ask questions based on documents scraped from the NHS website.
    Ask questions like **"What are the symptms of lyme disease??"** and return relevant results!

    ---
    ### Usage

    If you'd like to restrict your search to a specific number of items (`top_k`)
    you can with the *Advanced Options* dropdown.
    See a relevant chunk of text that seems to just miss what you need? No problem, just
    click on the boxed arrow icon on the left of each result card to find the original
    source.
    """
    return txt


def query_elasticsearch(client, index, query_vector, top_k):
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
                "params": {"query_vector": query_vector},
            },
        }
    }
    response = client.search(
        index=index,
        body={
            "size": top_k,
            "query": script_query,
            "_source": {"includes": ["text", "url"]},
        },
    )
    if response:
        return response["hits"]["hits"]
    return None


def do_search(index, vector_dim, model, es_client):
    st.write("# NHS Search")

    query = st.text_input("Ask questions about health!", "")
    with st.expander("Advanced Options"):
        top_k = st.slider("top_k", min_value=1, max_value=20, value=2)
    st.sidebar.write(get_sidebar_text(index, vector_dim))
    if query != "":
        with st.spinner("Querying, please wait..."):
            # compute the query vector
            q_vec = model.encode(query)
            retrieved_docs = query_elasticsearch(es_client, index, q_vec, top_k)

        if not retrieved_docs:
            st.error("No records found ...")
        else:
            # display each context
            for i, doc in enumerate(retrieved_docs):
                context = doc["_source"]["text"]
                url = "https://www" + doc["_source"]["url"]
                score = doc["_score"]
                card_component(
                    title="",
                    context=context,
                    highlight_start=None,
                    highlight_end=None,
                    score=round(score, 2),
                    url=url,
                    key=i + 1,
                )


def main(args):
    with open(args.config) as config_file:
        config = json.load(config_file)
    index = config["es"]["index"]
    vec_dim = config["es"]["mappings"]["properties"]["text_vector"]["dims"]
    model = get_embeddings_model(config["embeddings"]["model"])
    client = get_es_client("http://localhost:9200")
    do_search(index, vec_dim, model, client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexing elasticsearch documents.")
    parser.add_argument(
        "--config",
        default="em_es_config.json",
        help="elasticsearch-dense-retrieval configurations",
    )
    args = parser.parse_args()
    main(args)
