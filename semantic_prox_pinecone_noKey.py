'''
This script uses a roberta model to encode the text (both the input and the queries).
It uses a spacy tokenizer to tokenize the text.
The embeddings are loaded in a pinecone database that is then used for the queries.
It calculates the semantic distance between given sentences (the queries) and the input text

The source text is from a txt file (extracted from wikipedia).
'''

# importing the required libraries
import pinecone as pc
from sentence_transformers import SentenceTransformer
import torch
import spacy
from datetime import datetime
from time import time

# === loading the input text and the queries
# reading a text file and loading it into a list
txt_file = open("Barack_Obama.txt","r", encoding='utf8')
txt_content = []
for t_line in txt_file:
    if t_line != '\n':
        txt_content.append(t_line)

# loading the spacy tokenizer
spacy_tokenizer = spacy.load('en_core_web_sm')

# tokenizing by sentence the input text
corpus_sentences = []
for line in txt_content:
    doc = spacy_tokenizer(line)
    for sent in doc.sents:
        corpus_sentences.append(sent.text)

#print ('len of sentences in input text: ', len(corpus_sentences))

# defining the query sentences
query_sentences = ['obama was born in honolulu, hawaii', 'the president graduated from columbia university',
                   'obama is an american politician', 'was obama a president of the USA']
#print ('len of sentences in query text: ', len(query_sentences))

# === preparing pinecone and loading the model to create the embeddings
# initializing pinecone
pc.init(api_key="------------------------------",
              environment="us-west1-gcp-free")

active_indexes = pc.list_indexes()
index_description = pc.describe_index(active_indexes[0])
index_name = '------------------------'

# loading the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'sentence-transformers/all-roberta-large-v1'
model = SentenceTransformer(model_name)

# creating the Pinecone index
index = pc.Index(index_name)

# === embedding the text
# printing starting time
start_time = datetime.now()
print ('\n-Starting the embedding process for the input text at {}'.format(start_time), '\n')

t = time()

# creating the embeddings for the input text
corpus_embedding = model.encode(corpus_sentences, show_progress_bar=True).tolist()
print('Time to embed the input text: {} mins'.format(round((time() - t) / 60, 4)),'\n')
#print ('len of the corpus embeddings: ', len(corpus_embedding))

# creating the embeddings for the query sentences
query_embeddings = model.encode(query_sentences, show_progress_bar=True).tolist()


# === adding the input text embeddings to the pinecone index
#   IDs for the vectors would be added. They are 1 to n

# creating a list of (id, vector) tuples
#    generating unique IDs for each vector
ids = [str(i) for i in range(len(corpus_embedding))]
#metadata_label = ['sentence:']
data = list(zip(ids, corpus_embedding))
#index_description = pc.describe_index("myindex")
#print (index_description)
# upserting the corpus embeddings to the index
index.upsert(data)

# printing index information
#print("\nIndex info:", pc.describe_index(index_name))
#print("\nIndex stats:", index.describe_index_stats())

t = time()

# === querying the database/index with the query sentences

# looping over the query embeddings and sentences
for query_embedding, query_sentence in zip(query_embeddings, query_sentences):
    print("\nQuery:", query_sentence)
    res = index.query(vector= query_embedding, top_k=3, include_values=True)
    # printing the matched sentences and their scores
    for res in res.matches:
        print ("Matched sentence:", corpus_sentences[int(res.id)])
        print("   --- score:", res.score)

print('\nTime to evaluate the matching: {} mins'.format(round((time() - t) / 60, 4)))

