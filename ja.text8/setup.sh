#!/bin/bash -eu

# installation
pip install mecab-python3
git clone https://github.com/attardi/wikiextractor.git
cd wikiextractor
mkdir -p extracted

# download, preprocess and make data
wget "https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2"
python WikiExtractor.py -o extracted "jawiki-latest-pages-articles.xml.bz2"
cd ..
python process.py
python tokenizer.py

# post process
rm wikiextractor/jawiki-latest-pages-articles.xml.bz2
rm tmp.txt
rm -rf wikiextractor/extracted
