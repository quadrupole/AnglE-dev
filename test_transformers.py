#from transformers import AutoTokenizer, AutoModel
#import torch
## Sentences we want sentence embeddings for
#sentences = ["This is a test, and its a really long sentence!"]
#
## Load model from HuggingFace Hub
#tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
#model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
#model.eval()
#
## Tokenize sentences
#encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
## for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
## encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')
#
## Compute token embeddings
#with torch.no_grad():
#    model_output = model(**encoded_input)
#    # Perform pooling. In this case, cls pooling.
#    sentence_embeddings = model_output[0][:, 0]
## normalize embeddings
#sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
#print("Sentence embeddings:", sentence_embeddings)
#



#from transformers import pipeline
#
##pipe = pipeline(model="facebook/bart-large-mnli")
#pipe = pipeline(model="WhereIsAI/UAE-Large-V1")
#x = pipe("Tesc Stores 2342 JUN18",
#    candidate_labels=["shopping", "loans", "transfers", "services", "travel"],
#)
#
#print(x)


import numpy as np

from angle_emb import AnglE
import time
angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
#vec = angle.encode('hello world', to_numpy=True)
#print(vec)


# Check that the embeddings can handle a negation nicely
#query = "someone who is not interested in sports"
#query = "someone who is interested in sports"
#query = "people who care about their appearance the most"
#query = "people who enjoy going to cafes"
query = "people with video gaming interests, nerds that dont like physical contact"


sport_sentence = "I really like basketball its my favourite thing to do in the world!"
reading_sentence = "I really enjoy being at home rather than outside. I like to read and challenge my mind more than anything."
machine_sentence = "this machine can process thousands of words per minute."
physical_sentence = "central gym with scenic views of london with nice cafe"
dental_sentence = "cheapest dental plan available, get your perfect smile"

sentences = [query, sport_sentence, reading_sentence,machine_sentence, physical_sentence, dental_sentence]

t0 = time.time()
embeddings = angle.encode(sentences, to_numpy=True)
print(time.time()-t0)
print(embeddings.shape)

query_vector = embeddings[0,:]
embeddings = embeddings[1::,:]
sentences = sentences[1::]


embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
query_vector_normalized = query_vector / np.linalg.norm(query_vector)

# Calculate cosine similarity
similarity_scores = np.dot(embeddings_normalized, query_vector_normalized)

# Rank the embeddings based on similarity scores
ranking_indices = np.argsort(similarity_scores)[::-1]
ranked_embeddings = embeddings[ranking_indices]

print("Ranked embeddings:", ranked_embeddings)
print(np.asarray(sentences)[ranking_indices])
print(similarity_scores[ranking_indices])




