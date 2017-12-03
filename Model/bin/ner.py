from time import time
import sys,os
from nltk import word_tokenize
import pandas as pd
import sys


parent = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent+"/../lib/ext_lib/mitielib")

from mitie import *


def feature_extractor():

    trainer = ner_trainer("../data/ext_lib/total_word_feature_extractor.dat")

    return trainer


def ner_trainer_function(class_sheet_path,entity_sheet_path):

    class_sheet_path = "../data/ner/" + class_sheet_path
    entity_sheet_path = "../data/ner/" + entity_sheet_path

    class_df = pd.read_csv(class_sheet_path,names = ["id","text",'agent'],sep=",",header=0)
    entity_df = pd.read_csv(entity_sheet_path, names=["id", "entity", "start", "end"], sep=",", header=0)

    ###### Training for all possible models
    for classes in class_df.agent.unique():
        create_model(class_df[class_df["agent"]==classes],entity_df,classes)

    ###### Training the scheduling ner model
    #create_model(class_df[class_df["agent"]=="meeting"],entity_df,"meeting")
    ###### Training the sms ner model
    #create_model(class_df[class_df["agent"]=="sms"],entity_df,"sms")
    ###### Training the email ner model
    #create_model(class_df[class_df["agent"]=="email"],entity_df,"email")


def create_model(c_df,e_df,service):

    print("Started the model training for the " + service.upper() + " service.")

    if service =="flight":
        trainer_flight = feature_extractor()

    elif service == "meeting":
        trainer_meeting = feature_extractor()

    elif service == "connect":
        trainer_connect = feature_extractor()

    elif service == "iot":
        trainer_iot = feature_extractor()

    elif service == "dictionary":
        trainer_dictionary = feature_extractor()

    elif service == "joke":
        trainer_joke = feature_extractor()

    elif service == "climate":
        trainer_climate = feature_extractor()


    for ids in c_df['id']:

        print("IDS : " + str(ids))
        sub_df = e_df[e_df['id'] == ids]
        print(sub_df)

        if len(sub_df) != 0:


            sample = ner_training_instance(word_tokenize(c_df[c_df['id'] == ids]['text'].values[0] +"."))

            for i,row in sub_df.iterrows():
                #print(i)
                sample.add_entity(range(int(row['start']),int(row['end'])) ,row['entity'])

            if service == "flight":
                #trainer_flight = feature_extractor()
                trainer_flight.add(sample)

            elif service == "meeting":
                #trainer_meeting = feature_extractor()
                trainer_meeting.add(sample)

            elif service == "connect":
                trainer_connect.add(sample)

            elif service == "iot":
                trainer_iot.add(sample)

            elif service == "dictionary":
                trainer_dictionary.add(sample)

            elif service == "joke":
                trainer_joke.add(sample)

            elif service == "climate":
                trainer_climate.add(sample)


    if service == "flight":
        print("Reached the flight one !")
        trainer_flight.num_threads = 16
        ner_flight = trainer_flight.train()
        ner_flight.save_to_disk("../data/ner/flight_ner_model.dat")
    elif service == "meeting":
        print("Reached the meeting one !")
        trainer_meeting.num_threads=16
        ner_meeting = trainer_meeting.train()
        ner_meeting.save_to_disk("../data/ner/meeting_ner_model.dat")
    elif service == "connect":
        print("Reached the CONNECT one ! ")
        trainer_connect.num_threads = 16
        ner_connect = trainer_connect.train()
        ner_connect.save_to_disk("../data/ner/connect_ner_model.dat")
    elif service == "iot":
        print("Reached the IOT one !")
        trainer_iot.num_threads = 16
        ner_iot = trainer_iot.train()
        ner_iot.save_to_disk("../data/ner/iot_ner_model.dat")
    elif service == "dictionary":
        print("Reached the DICTIONARY one !")
        trainer_dictionary.num_threads = 16
        ner_dictionary = trainer_dictionary.train()
        ner_dictionary.save_to_disk("../data/ner/dictionary_ner_model.dat")
    elif service == "joke":
        print("Reached the JOKE one !")
        trainer_joke.num_threads = 16
        ner_joke = trainer_joke.train()
        ner_joke.save_to_disk("../data/ner/joke_ner_model.dat")
    else:
        print("Reached the CLIMATE one !")
        trainer_climate.num_threads = 16
        ner_climate = trainer_climate.train()
        ner_climate.save_to_disk("../data/ner/climate_ner_model.dat")


def dictionary_creator(classify,ner_model):

    labels = ner_model.get_possible_ner_tags()

    none_list = [None]*len(labels)

    tuples_list = list(zip(labels,none_list))

    create_dict = {"service" : classify,"values": dict(tuples_list) }

    return create_dict



def call_model(user_prompt,classification):

    ner = named_entity_extractor("../data/ner/" + classification + "_ner_model.dat")

    tokens = word_tokenize(user_prompt)
    entities1 = ner.extract_entities(tokens)
    print(entities1)

    user_dict = dictionary_creator(classification,ner)

    counter = 0

    for e in entities1:
        a = e[0]
        b = e[1]
        entity = " ".join(tokens[i] for i in a)

        entity_list = []
        entity_list.append(entity)

        for previous in entities1[:entities1.index(e)]:
            ran = previous[0]
            entity_prev = previous[1]

            if b == entity_prev:
                counter += 1
                entity_list.append(" ".join(tokens[i] for i in ran))
                #print(entity_list)

        if counter >= 1:
            user_dict["values"][b] = entity_list

        else:
            user_dict["values"][b] = entity


    print(user_dict)




#ner_trainer_function("commands.csv","entities.csv")

#from word2vec_final import classify

# def ner_try(user_array):
#
#     master_query = []
#
#     for user_queries in user_array:
#
#         classification = classify(user_queries)
#
#         user_dict = dictionary_creator(classification, ner)
#
#
#
#
#

    # if classification == "meeting":
    #
    #     user_dict = {"service":classification,
    #                  "values":{"date" : None,
    #                            "time" : None,
    #                            "place" : None,
    #                            "person" : None}
    #                  }
    #     counter = 0
    #
    #     for e in entities1:
    #         a = e[0]
    #         b = e[1]
    #         entity = " ".join(tokens[i] for i in a)
    #
    #         entity_list = []
    #         entity_list.append(entity)
    #
    #         for previous in entities1[:entities1.index(e)]:
    #             ran = previous[0]
    #             entity_prev = previous[1]
    #
    #             if b == entity_prev:
    #                 counter += 1
    #                 entity_list.append(" ".join(tokens[i] for i in ran))
    #                 print(entity_list)
    #
    #         if counter >= 1:
    #             user_dict["values"][b] = entity_list
    #
    #         else:
    #             user_dict["values"][b] = entity



        # for e in entities1:
        #     a = e[0]
        #     b = e[1]
        #     entity = " ".join(tokens[i] for i in a)
        #
        #     for previous in entities1[:entities1.index(e)]:
        #         ran = previous[0]
        #         entity_prev = previous[1]
        #         if b == entity_prev:
        #             counter+=1
        #             entity_list = []
        #             entity_list.append(entity)
        #             entity_list.append(" ".join(tokens[i] for i in ran))
        #             user_dict["values"][b] = entity_list
        #
        #
        #     if counter == 0:
        #
        #         user_dict["values"][b] = entity


    # elif classification == "flight":
    #
    #     user_dict = {"service" : classification,
    #                  "values":{"dest":None,
    #                            "time":None,
    #                            "source":None,
    #                            "class":None,
    #                            "date":None,
    #                            "passengers":None}
    #                  }
    #
    #     counter = 0
    #
    #     for e in entities1:
    #         a = e[0]
    #         b = e[1]
    #         entity = " ".join(tokens[i] for i in a)
    #
    #         entity_list = []
    #         entity_list.append(entity)
    #
    #         for previous in entities1[:entities1.index(e)]:
    #             ran = previous[0]
    #             entity_prev = previous[1]
    #
    #             if b == entity_prev:
    #                 counter += 1
    #                 entity_list.append(" ".join(tokens[i] for i in ran))
    #                 print(entity_list)
    #
    #         if counter >= 1:
    #             user_dict["values"][b] = entity_list
    #
    #         else:
    #             user_dict["values"][b] = entity
    # #     for e in entities1:
    #         a = e[0]
    #         b = e[1]
    #         entity = " ".join(tokens[i] for i in a )
    #         user_dict["values"][b] = entity
    #
    # else:
    #     print("Service currently not available")