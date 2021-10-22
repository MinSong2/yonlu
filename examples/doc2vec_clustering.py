from collections import namedtuple

from sklearn.cluster import KMeans

import numpy as np
from yonlu.word_embeddings.doc2vecModel import Doc2VecTrainer, Doc2VecSimilarity
import logging
import treform as ptm
import csv
import sys
from treform.document_clustering.documentclustering import DocumentClustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

model_file = './tmp/1630455815150_pv_dma_dim=100_window=5_epochs=20/doc2vec.model'
doc2vec = Doc2VecSimilarity()
doc2vec.load_model(model_file)
model = doc2vec.get_model()
# name either k-means, agglo, spectral_cocluster
name = 'agglo'
clustering = DocumentClustering(k=3)
# n_components means the number of words to be used as features
print(len(model.dv))
clustering.make_matrix(n_components=-1, doc2vec_matrix=model.dv.vectors)
clustering.cluster(name)

clustering.visualize()
