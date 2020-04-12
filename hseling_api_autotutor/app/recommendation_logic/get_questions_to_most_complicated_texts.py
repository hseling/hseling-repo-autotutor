
import json
from tqdm.auto import tqdm
import operator
import os
from ..database import redis_db
from ..testclass import get_set_elements

MAX_TEXTS_TO_TEST_NUMBER = 1

def extract_text_features(rec_text_map):
    # with open(text_map_path, "r", encoding = "utf-8") as f:
    #     rec_text_map = json.load(f)
        #load text features
    #rec_text_map = json.loads(text_map)
    rec_text_features_vector = []
    rec_text_features_vector.append(rec_text_map['lix'])
    rec_text_features_vector.append(rec_text_map['ttr'])
    rec_text_features_vector.extend(rec_text_map['average_sent_properties'])
    return rec_text_features_vector, len(rec_text_map['sentences_map'])

def extract_sent_features(text_map):
    # with open(text_map_path, "r", encoding = "utf-8") as f:
    #     rec_text_map = json.load(f)
        #load text features
    #rec_text_map = json.loads(text_map)
    recommended_sentences = []
    sentence_map = text_map['sentences_map']
    for sentence_ind in range(len(sentence_map)):
        rec_sent_feat = []
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['negation'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['coreference'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['vozvr_verb'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['prich'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['deepr'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['case_complexity'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['mean_depend_length'])
        recommended_sentences.append(rec_sent_feat)
    return recommended_sentences#, rec_text_map

def get_sentence_text(text_map, sentence_ind):
    text = text_map['raw_sentences_list'][sentence_ind]
    words_count = len(text_map['sentences_map'][sentence_ind]['sentence_words'])
    # sent_map = text_map['sentences_map'][sentence_ind]
    # text = ''
    # words_count = 0
    # for word_el in sent_map['sentence_words']:
    #     text += word_el['word'] + ' '
    #     words_count += 1
    # text = text.strip()
    return text, words_count

def get_complicated_texts(user_id):
    get_test_from_non_answered_stack = False
    user_tests_stack = redis_db.get("username_" + user_id + "_last_test")
    text_sentence_complexity_dict = {}
    if user_tests_stack is None:
        already_handled_texts_ids = redis_db.get("username_" + user_id + "_handled_texts")
        if already_handled_texts_ids is None:
            return {"error":"user does not exist"}
        elif len (already_handled_texts_ids) > 0:
            already_handled_texts_ids = already_handled_texts_ids.decode("utf-8").split("&")
        else:
            already_handled_texts_ids = []
        available_text_types = get_set_elements("text_map_types")
        # print("available_text_types", available_text_types)
        for text_type in available_text_types:
            type_storage_reference = "text_maps_" + text_type + "_storage_size"
            # print("type_storage_reference",type_storage_reference)
            text_maps_type_size = redis_db.get(type_storage_reference).decode("utf-8")
            for text_type_ind in range(int(text_maps_type_size)):
                if text_type_ind not in already_handled_texts_ids:
                    text_map_full_ind = "text_maps_" + text_type + "_" + str(text_type_ind)
                    text_map = redis_db.get(text_map_full_ind).decode("utf-8")
                    text_map = json.loads(text_map)
                    text_feat, sentences_count  = extract_text_features (text_map)
                    if sentences_count > 10 and sentences_count <= 25:
                        sent_complexity = sum(text_feat)/len(text_feat)
                        #print(text_feat, sum(text_feat)/len(text_feat))
                        text_sentence_complexity_dict[text_map_full_ind] = sent_complexity
    else:
        print(user_tests_stack)
        get_test_from_non_answered_stack = True
        user_tests_stack = user_tests_stack.decode("utf-8")
        user_tests_stack_list = user_tests_stack.split("&")
        for text_map_full_ind in user_tests_stack_list:
            if text_map_full_ind:
                print("text_map_full_ind", text_map_full_ind)
                # text_map_full_ind = "text_maps_" + text_type + "_" + str(text_type_ind)
                text_map = redis_db.get(text_map_full_ind).decode("utf-8")
                text_map = json.loads(text_map)
                text_feat, sentences_count = extract_text_features(text_map)
                # if sentences_count > 10 and sentences_count <= 25:
                sent_complexity = sum(text_feat) / len(text_feat)
                # print(text_feat, sum(text_feat)/len(text_feat))
                text_sentence_complexity_dict[text_map_full_ind] = sent_complexity
    sorted_text_feat_dict = sorted(text_sentence_complexity_dict.items(), key=operator.itemgetter(1), reverse = False)
    complicated_texts_ids = []
    recommended_texts_count = 0
    handled_texts = redis_db.get("username_" + user_id + "_handled_texts").decode("utf-8")
    for rec_text_el in sorted_text_feat_dict:
        # print("rec_text_el", rec_text_el)
        text_ind = rec_text_el[0]
        # print("rec", text_ind )
        if text_ind in handled_texts and get_test_from_non_answered_stack == False:
            continue#if text_ind already handled and we are not getting test from stack go on
        complicated_texts_ids.append(text_ind)
        recommended_texts_count += 1
        if recommended_texts_count >= MAX_TEXTS_TO_TEST_NUMBER:
            break
    complicated_texts_ids_list = redis_db.get("username_" + user_id + "_handled_texts")
    if complicated_texts_ids_list is None:
        complicated_texts_ids_list = ''
    else:
        complicated_texts_ids_list = complicated_texts_ids_list.decode("utf-8")
    last_recommended_texts_indexes = ''
    for com_txt_id in complicated_texts_ids:
        last_recommended_texts_indexes += com_txt_id + "&"
        if com_txt_id not in complicated_texts_ids_list:
            complicated_texts_ids_list += com_txt_id + "&"
    if get_test_from_non_answered_stack == False:#write new instance of last_test in case we are note giving last one right now
        redis_db["username_" + user_id + "_last_test"] = last_recommended_texts_indexes
    redis_db["username_" + user_id + "_handled_texts"] = complicated_texts_ids_list
    questions_json = {}
    for compl_text_id in complicated_texts_ids:
        questions_json[compl_text_id] = {"whole_text": '', "questions": []}
        sent_complicated_dicts = {}
        # print("compl_text_id", compl_text_id)
        text_map = redis_db.get(compl_text_id).decode("utf-8")
        text_map = json.loads(text_map)
        # print("text_map['raw_text']", text_map['raw_text'])
        questions_json[compl_text_id]['whole_text'] = text_map['raw_text']
        # print("questions_json[compl_text_id]['whole_text']", questions_json[compl_text_id]['whole_text'])
        sent_features_list = extract_sent_features (text_map)
        sent_index = 0
        for sent_feat in sent_features_list:
            av_sent_feat = sum(list(sent_feat))/len(sent_feat)
            sent_complicated_dicts[sent_index] = av_sent_feat
            sent_index += 1
        sorted_sent_feat_dict = sorted(sent_complicated_dicts.items(), key=operator.itemgetter(1), reverse = False)
        # print(sorted_sent_feat_dict)
        collected_questions_sentences_count = 0
        for rec_sent_el in sorted_sent_feat_dict:
            sent_ind = int(rec_sent_el[0])
            #print(sent_ind)
            sentence_text, words_count  = get_sentence_text(text_map,sent_ind)
            # print(type(sentence_text),"sentence_text")
            if words_count > 5:
                questions_json[compl_text_id]['questions'].append({"sent_ind":sent_ind,"sent_text":sentence_text})
                collected_questions_sentences_count += 1
            if collected_questions_sentences_count == 5:
                break
    # print("questions_json", questions_json)
    return questions_json

# already_handled_texts_ids = [1,2]
# get_complicated_texts(already_handled_texts_ids)

