from time import time
import sys,os
from nltk import word_tokenize
import pandas as pd
import sys

parent = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent+"/../mitielib")

from mitie import *


def feature_extractor():

    #start_feature = time()
    trainer = ner_trainer("/home/piyushkat/PycharmProjects/Classical_ML/Newsgroup20/MITIE-models/english/total_word_feature_extractor.dat")
    #print("FEATURE IMPORT TIME : ",time()-start_feature)

    return trainer


def ner_trainer_function(class_sheet_path,entity_sheet_path):

    class_df = pd.read_csv(class_sheet_path,names = ["id","text","agent"],sep=",",header=0)
    entity_df = pd.read_csv(entity_sheet_path, names=["id", "entity", "start", "end"], sep=",", header=0)

    return class_df,entity_df


def create_model(class_path,entity_path):

    #print("Started the model training for the " + service.upper() + " service.")

    c_df, e_df = ner_trainer_function(class_path,entity_path)
    new_trainer = feature_extractor()

    information = ["d","t","l"]

    for info in information :

        print("Executing for " + info)

        new_trainer = feature_extractor()

        for ids in c_df['id']:

            #print("IDS : " + str(ids))
            sub_df = e_df[e_df['id'] == ids]
            #print(sub_df)

            if len(sub_df) != 0:


                sample = ner_training_instance(word_tokenize(c_df[c_df['id'] == ids]['text'].values[0] +"."))

                for i,row in sub_df.iterrows():
                    #print(i)
                    try :
                        #print(row['entity'][-1])
                        if row['entity'][0] == info:
                            #print(row['entity'])
                            sample.add_entity(range(int(row['start']),int(row['end'])) ,row['entity'])

                            new_trainer.add(sample)
                    except:
                        print("Problem in ID : ",str(ids))

        new_trainer.num_threads = 16
        ner_climate = new_trainer.train()
        ner_climate.save_to_disk(info + "_ner_model.dat")



def call_model(user_prompt):

    ner = named_entity_extractor("t_ner_model.dat")

    #user_dict = dictionary_creator(classification,ner)

    tokens = word_tokenize(user_prompt)
    entities1 = ner.extract_entities(tokens)
    print(entities1)

    #user_dict = dictionary_creator(classification,ner)

    for e in entities1:
        a = e[0]
        b = e[1]
        entity = " ".join(tokens[i] for i in a)

        print("ENTITY : " + str(b) + " VALUE : " + entity)



#create_model("dtltext.csv","dtlentities.csv")


#call_model("i can't meet you at expensive places like Taj Hotels i am having some monetry issues please cooperate and meet me at Cafe coffee day")

#x,y,flag = ner_trainer_function("pnertext.csv","pnerentitiesold.csv")

#print(flag)

# def try1():
#     import pandas as pd
#
#     x = pd.read_csv("pnertext.csv", names=["id", "text", "agent"], sep=",", header=0)
#     y = pd.read_csv("pnerentities.csv", names=["id", "entity", "start", "end"], sep=",", header=0)
#     print(x[:32])
#     print(y[:35])
#
# try1()

#call_model("No I don't think that would work. Can we do it on Thursday","meeting")