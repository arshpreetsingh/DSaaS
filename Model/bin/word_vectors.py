import pandas as pd
import gensim as gs
import numpy as np

def convert_to_txt(csv_path):

    df = pd.read_csv(csv_path,sep=",",header=0,names=["id","text","agent"])
    df.dropna(inplace=True)

    with open("../data/word2vec_final/training.txt","wt") as fp:
        for text in df['text'].values:
            fp.write(text)
            fp.write("\n")

    fp.close()
    return df


def list_of_lists(csv_path):

    from nltk import word_tokenize

    list = []

    train_data = convert_to_txt(csv_path)

    with open("../data/word2vec_final/training.txt","rt") as fp :

        for line in fp:
        #text = fp.readline()
            if line is not None:
                #print(line)
                token_list = word_tokenize(line)
                list.append(token_list)



    return list,train_data



######## RESTRICTIONS - The size has to be in the same size as vocab size for multiplication
        # 200,5 seems to be the sweet spot

def create_gensim_word2vec(csv_path):

    X,train_data = list_of_lists(csv_path)

    print("CREATING WORD VECTORS !!!!!!")

    model = gs.models.Word2Vec(X,size = 200,min_count=5)
    w2v = dict(zip(model.wv.index2word,model.wv.syn0))

    model.save("../data/word2vec_final/trained_word_vectors.wv")
    print("WORD VECTORS CREATED ")

    return w2v,train_data


class TfIdfEmbedding(object):

    def __init__(self,word2vec):

        self.word2vec = word2vec
        #self.dimsize = len(word2vec.values()[1].shape[1])
        self.dimsize = 200
        self.word2weight = None

    def fit(self,X,y):

        # from sklearn.feature_extraction.text import TfidfVectorizer
        #
        # tfidf = TfidfVectorizer(analyzer=lambda x : x)
        # tfidf.fit(X)
        # max_idf = max(tfidf.idf_)
        #
        # self.word2weight = defaultdict(lambda : max_idf,[(w , tfidf.idf_[i]) for w,i in tfidf.vocabulary_.items()])

        return self


    def transform(self,X):

        from nltk import word_tokenize
        sentence_vector = np.empty([1, self.dimsize])


        i = 0

        for sentences in X:


            if i == 0:

                sentence_vector = np.array([np.mean(
                    [self.word2vec[w] for w in
                     word_tokenize(sentences) if w in self.word2vec ] or [np.zeros(self.dimsize)],axis=0
                )
                ])

                i += 1
            else:
                try :

                    sentence_vector = np.append(sentence_vector, np.array([np.mean(
                        [self.word2vec[w]  for w in word_tokenize(sentences) if w in self.word2vec ] or [np.zeros(self.dimsize)
                         ], axis=0)
                    ]), axis=0)
                    i+=1

                except :
                    print("Sentence vector is giving the error : ",str(i+1))
                    i+=1

        return sentence_vector