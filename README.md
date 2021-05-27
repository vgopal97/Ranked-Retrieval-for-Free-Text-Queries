# Ranked Retrieval for Free Text Queries

## Inverted Positional Index using tf-idf scores
Built tf-idf based rank retrieval system to answer free text queries. I have built a inverted positional index based on the tf-idf scores and ranked the free text queries accordingly. 

## Champion List
Built 2 Champion List
1. Champion List local
2. Champion list golbal (global scores for the documents given)

## Cluster Pruning
Leaders are selected randomly from the given set of documents. Euclidian distance is found out of a document with all leaders and all the documents are clustered. Built a cluster pruning scheme according to Leader nodes documents provided in a pickled document.

For running the program go to the code folder and run

```sh
python code.py query.txt
```
