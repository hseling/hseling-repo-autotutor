from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, RadiusNeighborsRegressor
import json
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from ..database import redis_db
from ..testclass import get_redis_list, get_set_elements


import math
import operator
from tqdm.auto import tqdm
import json

RECOMMENDATION_TEXTS_COUNT = 1
"""
import argparse
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('answers_json', help='path to the file with raw text')
args = parser.parse_args()

with open(args.answers_json, encoding = "utf-8") as f:
    ans_dict = json.load(f)
#print(ans_dict)
"""

def get_passed_test_count(username):
    user_id_handled_texts_storage = "username_" + username + "_handled_texts"
    handled_texts = redis_db.get(user_id_handled_texts_storage)
    if handled_texts is None:
        print("no data for get_passed_test_count")
        return None
    else:
        user_handled_texts_indexes = handled_texts.decode("utf-8")
        user_handled_texts_indexes_list = user_handled_texts_indexes.split("&")
        if '' in user_handled_texts_indexes_list:
            user_handled_texts_indexes_list.remove('')
        user_handled_texts_indexes_set = set(user_handled_texts_indexes_list)
        return len(user_handled_texts_indexes_set)


def extract_text_map_features(text_map_path):

    rec_text_map = redis_db.get(text_map_path).decode("utf-8")
    rec_text_map = json.loads(rec_text_map)
    # load text features
    rec_text_features_vector = []
    rec_text_features_vector.append(rec_text_map['lix'])
    rec_text_features_vector.append(rec_text_map['ttr'])
    rec_text_features_vector.extend(rec_text_map['average_sent_properties'])

    #load sentence features
    sentence_map = rec_text_map['sentences_map']
    recommended_sentences = []
    for sentence_ind in range(len(sentence_map)):
        rec_sent_feat = []
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['negation'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['coreference'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['vozvr_verb'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['prich'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['deepr'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['case_complexity'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['mean_depend_length'])
        rec_sent_vec = np.array(rec_sent_feat).reshape(1, -1)
        recommended_sentences.append(rec_sent_vec)

    #load words features
    recommended_words = []
    for sentence in sentence_map:
        for word_vector in sentence['words_vectors_list']:
            word_vector = np.array(word_vector).reshape(1, -1)
            recommended_words.append(word_vector)
    return recommended_words, recommended_sentences, rec_text_features_vector, len(rec_text_map['sentences_map'])

def predict_text_understanding(rec_text_words_vectors,rec_text_sentences_vectors, rec_text_txt_vector, word_model, sent_model, text_model):
    word_predictions = []
    for word_arr in rec_text_words_vectors:
        #word_arr = np.array(word).reshape(1, -1)
        prediction = word_model.predict(word_arr)
        word_predictions.append(prediction)
    positive_understanding = 0 
    negative_understanding = 0 
    for w_pred in word_predictions:
        if w_pred > 0:
            positive_understanding += 1
        elif w_pred < 0:
            negative_understanding += 1    
    words_understanding_probability = positive_understanding / (positive_understanding + negative_understanding)
    #print(words_understanding_probability)
    
    sent_predictions = []
    for sent in rec_text_sentences_vectors:
        sent_arr = np.array(sent).reshape(1, -1)
        prediction = sent_model.predict(sent_arr)
        sent_predictions.append(prediction)
    positive_understanding = 0 
    negative_understanding = 0
    # print("sent_predictions", sent_predictions)
    for snt_pred in sent_predictions:
        if snt_pred[0] >= 0.5:
            positive_understanding += 1
        elif snt_pred[0] < 0.5:
            negative_understanding += 1    
    sent_understanding_probability = positive_understanding / (positive_understanding + negative_understanding)
    #print(sent_understanding_probability)
    
    text_arr = np.array(rec_text_txt_vector).reshape(1, -1)
    text_understanding_pobability = text_model.predict(text_arr)[0]
    #print(text_understanding_pobability)
    
    return words_understanding_probability, sent_understanding_probability, text_understanding_pobability

def calc_dev_from_eighty_percent(understanding_vector):
    diff_squared = 0
    for value in understanding_vector:
        diff_squared += (value - 0.8) ** 2
        #print(value - 0.8, (value - 0.8) ** 2,diff_squared )
    diff_squared /= 3
    #print(diff_squared)
    st_dev = math.sqrt(diff_squared)
    return st_dev


def get_entity_model(user_id, entity_type):
    from collections import Counter
    entity_feature_objects_number_id = user_id + "_" + entity_type + "_features_size"
    print("entity_feature_objects_number_id", entity_feature_objects_number_id)
    entity_features_number = redis_db.get(entity_feature_objects_number_id)
    if entity_features_number is not None:
        entity_features_number = int(entity_features_number)
    else:
        return "user_not_exist"
    print("total",entity_features_number,entity_type)
    if entity_type == "text" and entity_features_number <=3:
        return "not_enough_features"
    entity_db_arr = []
    enitites_sizes_list = []
    for entity_feat_ind in range(entity_features_number):
        entity_feature_vector_id = user_id + "_" + entity_type + "_feature_" + str(entity_feat_ind)
        entity_feature_vector = get_redis_list(entity_feature_vector_id)
        enitites_sizes_list.append(len(entity_feature_vector))
        # print("get_recommended_text_json ", entity_type, len(entity_feature_vector))
        if len(entity_feature_vector) > 0:
            entity_db_arr.append(entity_feature_vector)
    print("entity_arr_sizes", entity_type, Counter(enitites_sizes_list))
    # word_db_arr = np.array([np.array(list(word_db.iloc[ind])) for ind in range(len(word_db))])
    entity_db_arr = np.array(entity_db_arr).astype(np.float64)
    # print("entity_db_arr.shape",entity_db_arr.shape )
    entity_db_arr_X = entity_db_arr[:, :-1]
    entity_db_arr_y = entity_db_arr[:, -1]
    # X_train, X_test, y_train, y_test = train_test_split(word_db_arr_X, word_db_arr_y, test_size=0.3)
    neigh_entity = KNeighborsRegressor(n_neighbors=3)
    neigh_entity.fit(entity_db_arr_X, entity_db_arr_y)
    return neigh_entity


def get_recommended_text_json(user_id, raw_json = False, save_json_to_directory = False):
    #print("GET RECOMMENDED STARTED")

    # user_id = ans_dict['user_id']
    #подгружаем базу текстов
    # db_3000 = pd.read_csv("cinema.csv")

    #обучаем ближайших соседей
    # word_features_number = int(redis_db.get(user_id + "_word_features_size"))
    # word_db_arr = []
    # for vocab_feat_ind in range(word_features_number):
    #     vocab_feature_vector_id = user_id + "_word_feature_" + str(vocab_feat_ind)
    #     vocab_feature_vector = get_redis_list(vocab_feature_vector_id)
    #     print("get_recommended_text_json len(vocab_feature_vector)", len(vocab_feature_vector))
    #     word_db_arr.append(vocab_feature_vector)
    # # word_db_arr = np.array([np.array(list(word_db.iloc[ind])) for ind in range(len(word_db))])
    #
    # word_db_arr = np.array(word_db_arr)
    #
    # print("word_db_arr.shape", word_db_arr.shape)
    # word_db_arr_X = word_db_arr[:,:-1]
    # word_db_arr_y = word_db_arr[:,-1]
    # #X_train, X_test, y_train, y_test = train_test_split(word_db_arr_X, word_db_arr_y, test_size=0.3)
    # neigh_words = KNeighborsRegressor(n_neighbors=6)
    # neigh_words.fit(word_db_arr_X, word_db_arr_y)

    neigh_words = get_entity_model(user_id, "word")


    # sent_features_number = int(redis_db.get(user_id + "_sent_features_size"))
    # sent_db_arr = []
    # for sent_feat_ind in range(sent_features_number):
    #     sent_feature_vector_id = user_id + "_sent_feature_" + str(sent_feat_ind)
    #     sent_feature_vector = get_redis_list(sent_feature_vector_id)
    #     sent_db_arr.append(sent_feature_vector)
    # sentence_db_arr = np.array(sent_db_arr)
    # sentence_db_arr_X = sentence_db_arr[:,:-1]
    # sentence_db_arr_y = sentence_db_arr[:,-1]
    # #X_train, X_test, y_train, y_test = train_test_split(sentence_db_arr_X, sentence_db_arr_y, test_size=0.15)
    # neigh_sent = KNeighborsClassifier(n_neighbors=2)
    # neigh_sent.fit(sentence_db_arr_X, sentence_db_arr_y)

    neigh_sent = get_entity_model(user_id, "sent")

    # text_features_number = int(redis_db.get(user_id + "_text_features_size"))
    # text_db_arr = []
    # for text_feat_ind in range(text_features_number):
    #     text_feature_vector_id = user_id + "_text_feature_" + str(text_feat_ind)
    #     text_feature_vector = get_redis_list(text_feature_vector_id)
    #     text_db_arr.append(text_feature_vector)
    # text_db_arr = np.array(text_db_arr)
    # text_db_arr_X = text_db_arr[:,:-1]
    # text_db_arr_y = text_db_arr[:,-1]
    # #X_train, X_test, y_train, y_test = train_test_split(text_db_arr_X, text_db_arr_y, test_size=0.15)
    # neigh_text = KNeighborsRegressor(n_neighbors=5)
    # neigh_text.fit(text_db_arr_X, text_db_arr_y)

    neigh_text = get_entity_model(user_id, "text")
    if neigh_text == "user_not_exist":
        return {"error": "User features not found in database"}
    elif neigh_text == "not_enough_features":
        return {"error": "Not enough data for recommendation algorithm. Please pass more tests"}
    text_recommendation_vector_dict = {}
    text_recommendation_dict = {}

    user_id_handled_texts_storage = "username_" + user_id + "_handled_texts"
    user_handled_texts_indexes = redis_db.get(user_id_handled_texts_storage).decode("utf-8")
    user_handled_texts_indexes_list = user_handled_texts_indexes.split("&")
    print("user_handled_texts_indexes", user_handled_texts_indexes)

    available_text_types = get_set_elements("text_map_types")
    # print("available_text_types", available_text_types)
    for text_type in available_text_types:
        type_storage_reference = "text_maps_" + text_type + "_storage_size"
        # print("type_storage_reference",type_storage_reference)
        text_maps_type_size = redis_db.get(type_storage_reference).decode("utf-8")
        for text_type_ind in range(int(text_maps_type_size)):
            text_map_path = "text_maps_" + text_type + "_" + str(text_type_ind)
            if text_map_path not in user_handled_texts_indexes_list:
                wrd_feat, snts_feat, text_feat, sentences_count  = extract_text_map_features (text_map_path)
                #predict_text_understanding(wrd_feat, snts_feat, text_feat, neigh_words, neigh_sent, neigh_text)
                trigr_recommendation, sentence_recommendation, text_recommendation = predict_text_understanding(wrd_feat,
                                                                                                                snts_feat, text_feat,
                                                                                                                neigh_words, neigh_sent, neigh_text)
                if sentences_count > 10 and sentences_count < 30:
                    text_standard_deviation = calc_dev_from_eighty_percent([trigr_recommendation, sentence_recommendation, text_recommendation])
                    text_recommendation_vector_dict[text_map_path] = [trigr_recommendation, sentence_recommendation, text_recommendation]
                    text_recommendation_dict[text_map_path] = text_standard_deviation
    sorted_text_feat_dict = sorted(text_recommendation_dict.items(), key=operator.itemgetter(1), reverse = False)

    output_texts = []
    rec_texts_list = []
    # for rec_text_el in sorted_text_feat_dict[:RECOMMENDATION_TEXTS_COUNT]:
    rec_text_el = sorted_text_feat_dict[0]
    text_map_path = rec_text_el[0]
    user_handled_texts_indexes += text_map_path + "&"
    # text_map_path = "text_maps_music_" + str(text_ind)
    text_map = redis_db.get(text_map_path).decode("utf-8")
    text_map = json.loads(text_map)
    # if len (raw_text) > 50:
    # text_name = "recommended_text_" + str(rex_text_index)

    rec_texts_list.append({"text_ind":text_map_path,"raw_text": text_map['raw_text']})
    
    # negative_sorted_text_feat_dict = sorted(text_recommendation_dict.items(), key=operator.itemgetter(1), reverse = True)
    # non_rec_texts_list = []
    # for non_rec_text_el in negative_sorted_text_feat_dict[:RECOMMENDATION_TEXTS_COUNT]:
    #     text_map_path = non_rec_text_el[0]
    #     user_handled_texts_indexes += text_map_path + "&"
    #     # text_map_path = "text_maps_music_" + str(text_ind)
    #     text_map = redis_db.get(text_map_path).decode("utf-8")
    #     text_map = json.loads(text_map)
    #     # if len (raw_text) > 50:
    #     # text_name = "recommended_text_" + str(rex_text_index)
    #     non_rec_texts_list.append({"text_ind":text_map_path,"raw_text": text_map['raw_text']})

    # for rec_text_json, non_rec_text_json in zip(rec_texts_list, non_rec_texts_list):
    #     output_texts.append({"recommended":rec_text_json, "non_recommended":non_rec_text_json})

    redis_db[user_id_handled_texts_storage] = user_handled_texts_indexes

    # if save_json_to_directory:
    #     recommendation_json_path = user_id + "_text_recommendation.json"
    #     print("save to", recommendation_json_path)
    #     with open(recommendation_json_path, "w", encoding = "utf-8") as f:
    #         json.dump(output_texts, f, ensure_ascii=False, indent = 4)
    # else:
    if len(rec_texts_list) == 0:
        return {"error":"no more texts in database"}
    return rec_texts_list