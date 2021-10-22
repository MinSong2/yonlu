import setuptools
import os
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

name="yonlu"

#torch for CPU only
#pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Build a list of all project modules
packages = find_packages(exclude = [])

print('packages', packages)

setuptools.setup(
    name="yonlu", # Replace with your own username
    version="1.1.12",
    author="Min Song",
    author_email="min.song@yonsei.ac.kr",
    description="A deep learning based natural language understanding module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MinSong2/yonlu",
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "cython",
        "seqeval",
        "pytorch-crf",
        "numpy",
        "sklearn-crfsuite",
        "gensim",
        "konlpy",
        "krwordrank",
        "lxml",
        "matplotlib",
        "networkx",
        "node2vec",
        "bs4",
        "pycrfsuite-spacing",
        "scikit-learn",
        "scipy",
        "seaborn",
        "soynlp",
        "torch",
        "tomotopy",
        "pyLDAvis",
        "wordcloud",
        "nltk",
        "newspaper3k",
        "selenium",
        "soylemma",
        "bokeh",
        "tensorflow-estimator>=2.4.0",
        "tensorflow>=2.4.1",
        "beautifulsoup4",
        "benepar>=0.2.0",
        "boto3",
        "kobert-transformers",
        "treform",
        "openai"
    ],
    python_requires='>=3.7',
)