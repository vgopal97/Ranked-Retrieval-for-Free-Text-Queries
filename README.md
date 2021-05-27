# Ranked Retrieval for Free Text Queries

Built tf-idf based rank retrieval system to answer free text queries. I have built a inverted positional index based on the tf-idf scores and ranked the free text queries accordingly. 

Built 2 Champion List
1. Champion List local
2. Champion list golbal (global scores for the documents given)

Built a cluster pruning scheme according to Leader nodes documents provided in a pickled document.

For running the program go to the code folder and run

'''sh
python code.py query.txt
'''
