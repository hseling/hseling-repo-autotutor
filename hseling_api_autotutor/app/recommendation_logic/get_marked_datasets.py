from .test_and_recommendation_w2v import user_vector
from tqdm.auto import tqdm
from ..database import redis_db

from collections import OrderedDict
#from text_processing_udpipe_w2v import get_text_map
import os
from random import randint
import random
import numpy as np
import pandas as pd
import json

"""
import argparse
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('answers_json', help='path to the file with raw text')
args = parser.parse_args()
"""
#with open(args.answers_json, encoding = "utf-8") as f:
 #   answer_dict = json.load(f)

"""
json_text_map = get_text_map("./text_8.txt")

#print(json_text_map['sentences_map'][0])

user.update_vector_with_answer_sentence(json_text_map['sentences_map'][0],1)
user.update_vector_with_answer_sentence(json_text_map['sentences_map'][4],1)
user.update_vector_with_answer_sentence(json_text_map['sentences_map'][6],0)

user.end_text(json_text_map)
"""

def generate_user_knowledge_database(answers_dict):

    # texts = pd.read_csv("3000.csv")
    user_id = answers_dict['username']
    user_tests_stack = redis_db.get("username_" + user_id + "_last_test")
    print("generate_user_knowledge_database >> user_tests_stack", user_tests_stack)
    if user_tests_stack is not None:
        # user_tests_stack["username_" + user_id + "_last_test"] = None
        redis_db.delete("username_" + user_id + "_last_test")
    user_id_handled_texts_storage = redis_db.get("username_" + user_id + "_handled_texts")
    if user_id_handled_texts_storage is None:
        user_id_handled_texts_storage = ''
    else:
        user_id_handled_texts_storage = user_id_handled_texts_storage.decode("utf-8")
    user = user_vector(user_id, debug = False)
    for text_path in tqdm(answers_dict['answers'].keys()):
        # if text_path not in user_id_handled_texts_storage:
        text_map = redis_db.get(text_path).decode("utf-8")
        text_map = json.loads(text_map)
        if text_path not in user_id_handled_texts_storage:
            user_id_handled_texts_storage += text_path + '&'
        user.start_new_text()
        answers_to_current_text_questions = answers_dict['answers'][text_path]
        for sentence_index in answers_to_current_text_questions:
            correctness = answers_to_current_text_questions[sentence_index]
            user.update_vector_with_answer_sentence(text_map['sentences_map'][int(sentence_index)],correctness)
        user.end_text(text_map)
    user.export_user_db()
    redis_db["username_" + user_id + "_handled_texts"] = user_id_handled_texts_storage

    # user.start_new_text()
    # for answer_ind, correctness in zip(answer_indexes,correct_ans):
    #     user.update_vector_with_answer_sentence(text_map['sentences_map'][answer_ind],correctness)
    # user.end_text(text_map)
    # user.export_user_vector()

# generate_user_knowledge_database(answer_dict, raw_json = False)

"""
answer_dict = {"222" :['0-' ,'1-' ,'2+' ,'4+'],
"862":['1+','2-', '4+', '6-', '12-'],
"321":['5+' ,'0-' ,'4-' ,'1+'],
"364":['2+', '6-', '3-', '7-' ],
"502":['13-', '3+', '0-', '4+'],
"878":['3-', '0+', '4-', '1-'],
"666":['3-', '1-', '2-' ],
"92":['1+', '2-'],
"615":['1+', '2-'],
"450":['6-', '7-', '11+', '9+' ,'10+' ],
"722":['1+', '6-', '2+', '3+', '6-', '8-'],
"732":['8-', '0-', '3-', '4-', '2+', '1-'],
"611":['1+'],
"251":['5-', '4+', '2+', '1+', '3+']}
"""