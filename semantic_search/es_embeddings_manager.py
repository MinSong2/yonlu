
import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import os
import json
import logging
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
import re

def load_nhs_documents(docs_path):
    """Load and pre-process NHS documents.
    Parameters
    ----------
    docs_path : str
        Path of NHS documents, one text file per page.
    Returns
    -------
    list of dict
        a list of dictionaries, each element of list corresponds to a
        processed\clean paragraph (~128 tokens) and the source URL.
    """
    header = re.compile(r"(?s)Cookies.*\nHome Health A to Z\n")

    def clean(text):
        text = header.sub("", text)
        text = re.sub(r"\n+", "\n", text)
        text = text.split("Page last reviewed:")[0]
        return text

    paths = [str(x) for x in Path(docs_path).glob("*.txt")]
    nhs_text = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            nhs_text.append(text)
    nhs_text = [clean(text) for text in nhs_text]
    # The embedding model expects maximum 128 tokens of text (~400-600 chars),
    # so we split text into chunks of ~500 characters such that:
    # a. split on new line characters (\n)
    # b. include an overlap of one sentence between chunks to avoid missing a
    #  relevant sentence
    chunked_text = []
    chunk = 500  # num characters, roughly corresponds to token length limit of 128
    for i, page in enumerate(nhs_text):
        url = paths[i][8:-5].replace("_", "/")
        page = page.split("\n")
        context = ""
        for j in range(len(page)):
            if j != 0 and len(context) == 0:
                context += page[j - 1] + " "
            context += page[j] + " "
            if len(context) >= chunk:
                chunked_text.append({"text": context, "url": url})
                context = ""
    return chunked_text


def create_es_document(document, document_emb, index_name):
    """Create the elastic search document.
    Parameters
    ----------
    document : str
        a paragraph from the NHS webpage
    document_emb : numpy.ndarray
        computed embedding of the document
    index_name : str
        name of elastic search index
    Returns
    -------
    dict
        elastic search document representation
    """
    return {
        "_op_type": "index",
        "_index": index_name,
        "text": document["text"],
        "url": document["url"],
        "text_vector": document_emb.tolist(),
    }


def compute_embeddings(model, documents, batch_size=256):
    """Compute dense embeddings.
    Parameters
    ----------
    model : SentenceTransformer
        sentence transformer model instance
    documents : list of str
        list of processed/clean documents
    batch_size : int, optional
        batch size, by default 256
    Yields
    ------
    list of numpy.ndarray
        a list of document vectors
    """

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        embeddings = model.encode([doc["text"] for doc in batch_docs])
        for emb in embeddings:
            yield emb


def load_dataset(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def create_index(client, index, settings, mappings):
    try:
        client.indices.create(index=index, settings=settings, mappings=mappings,)
        logging.info(f"{index} created")
    except elasticsearch.exceptions.RequestError as ex:
        if ex.error == 'resource_already_exists_exception':
            pass  # Index already exists. Ignore.
        else:  # Other exception - raise it
            raise ex

def create_documents(data_path, processed_path, model_name, index):
    if not os.path.exists(processed_path):
        documents = load_nhs_documents(data_path)
        logging.info(f"{len(documents)} documents processed ...")
        model = SentenceTransformer(model_name)
        with open(processed_path, "w") as f:
            for doc, emb in zip(documents, compute_embeddings(model, documents)):
                d = create_es_document(doc, emb, index)
                f.write(json.dumps(d) + "\n")
        logging.info(f"{len(documents)} documents written to {processed_path} ...")


def index_documents(client, processed_path):
    doc_counts = int(
        client.count(index="nhs_health")["count"]
    )
    if doc_counts == 0:
        docs = load_dataset(processed_path)
        bulk(client, docs)
        logging.info(f"{len(docs)} documents indexed ...")


def index(args):
    with open(args.config) as config_file:
        config = json.load(config_file)
    client = Elasticsearch("http://localhost:9200")
    create_index(
        client,
        config["es"]["index"],
        config["es"]["settings"],
        config["es"]["mappings"],
    )
    create_documents(
        config["data"]["path"],
        config["data"]["processed_path"],
        config["embeddings"]["model"],
        config["es"]["index"],
    )
    index_documents(client, config["data"]["processed_path"])

