from novelty import decider
from word2vec_final_2 import classify
from ner import call_model




def complete(user_query):

    #from word_vectors import TfIdfEmbedding

    #print(dir())

    nd = decider([user_query])

    if nd == 1:
        classification = classify([user_query])
        print(classification)
        call_model(user_query,classification[0])

    else:
        print("OUTLIER")

complete("Book a meeting with Piyush for 2pm")