import nltk
import pickle
import numpy as np
import os
import glob
from bs4 import BeautifulSoup
from operator import itemgetter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import BracketParseCorpusReader
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import WhitespaceTokenizer
import math
import re
import sys

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
stop_words = stop_words.union(",","(",")","[","]","{","}","#","@","!",":",";",".","?" , "-" , ":" , "%")

'''
THIS CLASS IS USED FOR BUILDING THE INVERTEDINDEX USING tf-idf SCORING MECHANISM.
THIS CLASS CONSISTS OF MANY METHODS USED FOR BUILDING THE INVERTED INDEX
THE extract_text METHOD IS USED FOR EXTRACTING TEXT FROM THE HTML DOCUMENTS
THE build_semindex METHOD IS USED FOR CREATING A NORMAL INVERTD INDEX WITH POSTING OF TYPE (docID , tftd)
THE build_index METHOD TAKES THE semindex AND BUILDS THE INDEX OF TYPE (term , idf) : postings of type (docID , tftd)
'''
class Index(object):
    def __init__(self):
        self.findN()
        self.build_index()

    def findN(self):
        path = os.getcwd() + "/Dataset/Dataset"
        doclist = os.listdir(path)
        self.N = len(doclist)

    def idf(self , df):
        return math.log(self.N/df)
    
    def w_td(self , tfd):
            return math.log(1+tfd)

    def hasNumbers(self , inputString):
        return bool(re.search(r'\d', inputString))

    def extract_text(self , file):
        htmlfile = open(file)
        soup = BeautifulSoup(htmlfile, features="html.parser")
        ptags = soup.findAll("p")
        string = ""
        for tag in ptags:
            string = string + " " + tag.get_text()
        return string

    def build_semindex(self):
        path = os.getcwd() + "/Dataset/Dataset"
        os.chdir(path)
        Lemmatizer = WordNetLemmatizer()
        semindex = {}
        for file in glob.glob("*.html"):
            textdict = {}
            textfile = self.extract_text(file)
            docID = int(file[:-5])
            tokens = word_tokenize(textfile)
            terms = [Lemmatizer.lemmatize(token.lower()) for token in tokens]
            for term in terms:
                if term in textdict.keys():
                    textdict[term] += 1
                else:
                    textdict.update({term : 1})
            
            for term in textdict.keys():
                
                if term in stop_words or self.hasNumbers(term):
                    continue

                tftd = self.w_td(textdict[term])
                posting = (docID , tftd)

                if term not in semindex.keys():
                    semindex.update({term : []})
                semindex[term].append(posting)
            #print("INDEXED DOCUMENT : ",docID , " INDEX SIZE" , len(semindex))
        return semindex

    def build_index(self):
        index = {}
        semindex = self.build_semindex()

        for term in semindex.keys():
            idf_term = self.idf(len(semindex[term]))
            term_tuple = (term , idf_term)
            index.update({term_tuple : semindex[term]})
        
        self.index = index


# Function to build a championslist
def build_localchampionslist(index):
    localchampionslist = {}
    for term in index.keys():
        termlist = index[term]
        termlist.sort(key = lambda x: float(x[1]), reverse = True) 
        localchampionslist.update({term : termlist[0:50]})
    return localchampionslist

#Function to build a Global championslist
def build_globalchampionslist(index , sqscores):
    globalchampionslist = {}
    for term in index.keys():
        termlist = index[term]
        globallist = []
        for doc in termlist:
            document = list(doc)
            document[1] = document[1] + sqscores[int(document[0]) - 1]
            globallist.append(tuple(document))
        globallist.sort(key = lambda x: float(x[1]), reverse = True) 
        globalchampionslist.update({term : globallist[0:50]})

    return globalchampionslist

#creating docvectorrs for scoring
def create_docvectors(Index,N):
    t = len(Index.keys())
    docvectors = np.zeros((N,t))
    i=0
    for key in Index.keys():
        for doc in Index[key]:
            docvectors[doc[0]][i] = key[1]*doc[1]
        i=i+1
    return docvectors

#for calculating cosine between vectors
def cosine_distance(A,B):
    a = math.sqrt(np.sum(np.square(A)))
    b = math.sqrt(np.sum(np.square(B)))

    if a==0 or b==0:
        return 0

    return np.sum(A*B)/(a*b)

'''
USED FOR CREATING CLUSTERS USING CLUSTER PRUNING ALGORITHM
FROM HE GIVEN LEADERS.
'''

def create_clusters(docvectors , Leaders, N):

    clusters = [[] for x in range(len(Leaders))]
    for i in range(N):

        distance = np.zeros(len(Leaders))

        for j in range(len(Leaders)):
            distance[j] = cosine_distance(docvectors[i] , docvectors[Leaders[j]])
            #print(distance[j])
        
        leader = np.argmax(distance)
        clusters[leader].append(i)
    return clusters


''' CLASS USED FOR SCORING, USES VARIETY OF METHODS TO 
CALCULATE THE SCORE AND PRINT THE SEARCH RESULTS.
'''  
class Scoring(object):

    def __init__(self , index , docvectors , clusters , Leaders ,localchampionslist , globalchampionslist , filename):
        self.index = index
        self.docvectors = docvectors
        self.clusters = clusters
        self.filename = filename
        self.Leaders = Leaders
        self.localchampionslist = localchampionslist
        self.globalchampionslist = globalchampionslist
        self.get_queries()

    def get_queries(self):
        Queries = open(self.filename).read().splitlines()
        self.queries = Queries
    
    def cosine_distance(self,A,B):
        a = math.sqrt(np.sum(np.square(A)))
        b = math.sqrt(np.sum(np.square(B)))

        if a==0 or b==0:
            return 0
        return np.sum(A*B)/(a*b)

    def hasNumbers(self , inputString):
        return bool(re.search(r'\d', inputString))

    def Union(self , lst1, lst2): 
        final_list = list(set(lst1) | set(lst2)) 
        return final_list 
    
    def create_strings(self , lst):
         string=""
         for doc in lst:
             string += "< Doc"+str(doc[0])+" , " + str(doc[1]) + " >  " 
         return string

    

    def tf_idf_score(self,Q):
        
        N = len((self.docvectors))
        scores = []

        for docID in range(N):
            tup = (docID ,self.cosine_distance(Q , self.docvectors[docID]))
            scores.append(tup)
        
        scores.sort(key = lambda x: float(x[1]), reverse = True) 

        return scores[0:10]

    def local_championlist_score(self,Q,Qterms):

        N = 50
        scores = []
        docs=[]

        for term in self.localchampionslist.keys():
            if term[0] in Qterms:
                qdocs =[]
                for docID in self.localchampionslist[term]:
                    qdocs.append(docID[0])
                
                docs = self.Union(docs , qdocs)

        for docID in docs:
            tup = (docID ,self.cosine_distance(Q , self.docvectors[docID]))
            scores.append(tup)
        
        scores.sort(key = lambda x: float(x[1]), reverse = True) 

        return scores[0:10]


        

    def global_championlist_score(self,Q,Qterms):

        N = 50
        scores = []
        docs=[]

        for term in self.globalchampionslist.keys():
            if term[0] in Qterms:
                qdocs =[]
                for docID in self.globalchampionslist[term]:
                    qdocs.append(docID[0])
                
                docs = self.Union(docs , qdocs)

        for docID in docs:
            tup = (docID ,self.cosine_distance(Q , self.docvectors[docID]))
            scores.append(tup)
        
        scores.sort(key = lambda x: float(x[1]), reverse = True) 

        return scores[0:10]

    
    def cluster_pruning_score(self,Q):

        N = len(self.docvectors)
        scores = np.zeros(N)
        cluster_scores = []

        for doc in self.Leaders:
            scores[doc] = self.cosine_distance(Q , self.docvectors[doc])
        
        leader = np.argmax(scores)

        try:
            l = self.Leaders.index(leader)
        except:
            return []

        for docID in self.clusters[l]:
            tup = (docID , self.cosine_distance(Q , self.docvectors[docID]))
            cluster_scores.append(tup)

        cluster_scores.append((leader , scores[leader]))
        cluster_scores.sort(key = lambda x: float(x[1]), reverse = True) 

        return cluster_scores[0:10]
    
    def get_scores(self):
        N = len(self.docvectors)
        Lemmatizer = WordNetLemmatizer()
        docvectors= [0]*N
        f = open("RESULTS2_20CS60R45.txt", "w")
        for q in self.queries:
             qvector = np.zeros(len(self.index.keys()))
             tokens = word_tokenize(q)
             terms = [Lemmatizer.lemmatize(token.lower()) for token in tokens]

             qterms =[]

             for t in terms:
                 if t not in stop_words and not self.hasNumbers(t):
                     qterms.append(t)

     
             i=0

             for term in self.index.keys():
                 if term[0] in qterms:
                     qvector[i] = term[1]
                 i = i + 1

             tf_idf_docs = self.tf_idf_score(qvector)
             lcs_docs = self.local_championlist_score(qvector , qterms)
             gcs_docs = self.global_championlist_score(qvector , qterms)
             cps_docs = self.cluster_pruning_score(qvector)

             f.write(q)
             f.write("\n")
             f.write(self.create_strings(tf_idf_docs))
             f.write("\n")
             f.write(self.create_strings(lcs_docs))
             f.write("\n")
             f.write(self.create_strings(gcs_docs))
             f.write("\n")
             f.write(self.create_strings(cps_docs))
             f.write("\n")
        
        f.close()

        #print("SUCCESFUULY EVALUATED QUERIES")



os.chdir("..")
index = Index()
InvertedIndex = index.index
os.chdir("..")
#print(os.getcwd())
with open('StaticQualityScore.pkl', 'rb') as f:
    sqscores = pickle.load(f)

with open('Leaders.pkl', 'rb') as f:
    Leaders = pickle.load(f)
print(Leaders)

os.chdir("..")
path = os.getcwd()

os.chdir(path + "/code")
localchampionslist = build_localchampionslist(InvertedIndex)
globalchampionslist = build_globalchampionslist(InvertedIndex , sqscores)
docvectors = create_docvectors(InvertedIndex , index.N)
clusters = create_clusters(docvectors , Leaders , index.N)
score = Scoring(InvertedIndex , docvectors , clusters , Leaders , localchampionslist , globalchampionslist , sys.argv[1])
score.get_scores()