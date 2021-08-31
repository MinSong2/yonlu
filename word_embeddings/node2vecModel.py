import gensim
import networkx as nx
from gensim.models import KeyedVectors
from node2vec import Node2Vec
import operator

# Embed edges using Hadamard method
from node2vec.edges import HadamardEmbedder
import multiprocessing

class Node2VecModel:
    def __init__(self):
        self.model = None
        self.G = nx.Graph()

    def create_random_graph(self):
        # Create a graph
        self.G = nx.fast_gnp_random_graph(n=100, p=0.5)

    def create_graph(self, co_occurrence, word_hist, threshold):
        filtered_word_list = []
        for pair in co_occurrence:
            node1 = ''
            node2 = ''
            for inner_pair in pair:
                if type(inner_pair) is tuple:
                    node1 = inner_pair[0]
                    node2 = inner_pair[1]
                elif type(inner_pair) is str:
                    inner_pair = inner_pair.split()
                    if len(inner_pair) == 2:
                        node1 = inner_pair[0]
                        node2 = inner_pair[1]
                elif type(inner_pair) is int:
                    if float(inner_pair) >= threshold:
                        # print ("X " + node1 + " == " + node2 + " == " + str(inner_pair) + " : " + str(tuple[node1]))
                        self.G.add_edge(node1, node2, weight=float(inner_pair))
                        if node1 not in filtered_word_list:
                            filtered_word_list.append(node1)
                        if node2 not in filtered_word_list:
                            filtered_word_list.append(node2)
                elif type(inner_pair) is float:
                    if float(inner_pair) >= threshold:
                        # print ("X " + node1 + " == " + node2 + " == " + str(inner_pair) + " : ")
                        self.G.add_edge(node1, node2, weight=float(inner_pair))
                        if node1 not in filtered_word_list:
                            filtered_word_list.append(node1)
                        if node2 not in filtered_word_list:
                            filtered_word_list.append(node2)

        for word in word_hist:
            if str(word) in filtered_word_list:
                self.G.add_node(word, count=word_hist[word])

        print(self.G.number_of_nodes())

    def train(self, dimensions, walk_length, num_walks, min_count=1):
        cores = multiprocessing.cpu_count()  # Count the number of cores in a computer
        # Precompute probabilities and generate walks
        node2vec = Node2Vec(self.G,
                            dimensions=dimensions,
                            walk_length=walk_length,
                            num_walks=num_walks,
                            workers=cores-1)

        ## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
        # Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
        # node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="/mnt/tmp_data")

        # Embed
        self.model = node2vec.fit(window=10, min_count=min_count,
                                  batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

    def save_graph(self, graph_file):
        nx.write_graphml_lxml(self.G, graph_file)

    def load_graph(self, graph_file):
        self.G = nx.read_graphml(graph_file)

    def compute_centrality(self, top_n=30):
        print('network statistics')
        print(nx.info(self.G))

        degree = nx.degree_centrality(self.G)
        degree_l = sorted(degree.items(), key=operator.itemgetter(1), reverse=True)
        print('degree centrality\n')
        for i, n in enumerate(degree_l,100):
            print(str(n))
        print()

        eigen = nx.eigenvector_centrality(self.G)
        print('eigenvector centrality\n')
        eigen_l = sorted(eigen.items(), key=operator.itemgetter(1), reverse=True)
        for n in eigen_l:
            print(str(n))
        print()

        betweenness = nx.betweenness_centrality(self.G)
        print('betweenness centrality\n')
        betweenness_l = sorted(betweenness.items(), key=operator.itemgetter(1), reverse=True)
        for n in betweenness_l:
            print(str(n))
        print()

    def save_model(self, embedding_filename, embedding_model_file):
        # Save embeddings for later use
        self.model.wv.save_word2vec_format(embedding_filename, binary=True)

        # Save model for later use
        self.model.save(embedding_model_file)

    def load_model(self, embedding_filename):
        self.model = KeyedVectors.load_word2vec_format(embedding_filename, binary=True, unicode_errors='ignore')

    #to do: node2vec model loading...

    def most_similars(self, word):
        # Look for most similar nodes
        return self.model.most_similar(word)  # Output node names are always strings

    def compute_similarity(self, first_node, second_node):
        edges_embs = HadamardEmbedder(keyed_vectors=self.model)

        # Look for embeddings on the fly - here we pass normal tuples
        edges_embs[(first_node, second_node)]
        ''' OUTPUT
        array([ 5.75068220e-03, -1.10937878e-02,  3.76693785e-01,  2.69105062e-02,
               ... ... ....
               ..................................................................],
              dtype=float32)
        '''

        # Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
        edges_kv = edges_embs.as_keyed_vectors()

        try:
            # Look for most similar edges - this time tuples must be sorted and as str
            results = edges_kv.most_similar(str((first_node, second_node)))

            # Save embeddings for later use
            # edges_kv.save_word2vec_format(EDGES_EMBEDDING_FILENAME)
        except:
            return []

        return results