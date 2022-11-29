import logging
import argparse
from yonlu.semantic_search.es_embeddings_manager import index

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Creating elastic search index")
    parser.add_argument(
        "--config",
        default="em_es_config.json",
        help="elasticsearch-dense-retrieval configurations",
    )
    args = parser.parse_args()
    index(args)