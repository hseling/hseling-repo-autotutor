#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ..database import redis_db
from ..testclass import get_redis_list

from string import punctuation
full_punctuation = punctuation + "–" + "," + "»" + "«" + "…" +'’'

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from gensim.models.keyedvectors import FastTextKeyedVectors
# from collections import OrderedDict
import copy

import os
from statistics import mean 
import numpy as np

import pymorphy2

from .ud_class import Model

import json


BASE = os.path.dirname(os.path.abspath(__file__))

# try:
#     fasttext = FastTextKeyedVectors.load("/Users/lilyakhoang/input/araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model")
# except:
#     fasttext = FastTextKeyedVectors.load(os.path.join(BASE,"./vectors/araneum_none_fasttextcbow_300_5_2018.model"))


from ..word2vec import word2vec #!!!!!боевой вариант


# try:
#     with open("/Users/nbabakov/input/word2vec.json") as f:
#         word2vec = json.load(f)
# except:
#     with open(os.path.join(BASE, "processing_data", 'word2vec.json')) as f:
#         word2vec = json.load(f)

#"smart_colloc_freq.json"

with open (os.path.join(BASE,"lyashevskaya_freq_dict_norm.json") , "r", encoding="utf-8") as f:
    lyashevskaya_freq_dict = json.load(f)

def read_text(path):
    raw_text = ''
    with open (path, 'r', encoding = "utf-8") as f:
        for line in f.readlines():
            raw_text += line + ' ' 
    return raw_text

def get_conllu_from_unite_line_text(text, model):
    sentences = model.tokenize(text)
    for s in sentences:
        model.tag(s)
        model.parse(s)
    conllu = model.write(sentences, "conllu")
    return conllu
    
def get_conllu_text_map(conllu_parsed_object):
    conllu_text_map = []
    conllu_sentence_map = []
    for line in conllu_parsed_object.split('\n'):
        if line:
            if line[0].isdigit():
                #print(line.split('\t'))
                conllu_sentence_map.append(line.split('\t'))
            else:
                if(len(conllu_sentence_map) > 0):
                    conllu_text_map.append(conllu_sentence_map)
                    conllu_sentence_map = []   
                    #print("appended")
    if(len(conllu_sentence_map) > 0):
        conllu_text_map.append(conllu_sentence_map)
    return conllu_text_map
    
def get_lemm_and_orig_text_from_udmap(conllu_map):
    lemm_sentences_list = []
    sentences_list = []
    for sentence in conllu_map:
        lemm_line = ''
        raw_text_line = ''
        for word in sentence: 
            if (word[3] != 'PUNCT'):
                #print(word[2])
                clean_lemma = ''
                for char in word[2]:
                    if char not in full_punctuation:
                        clean_lemma += char.lower()
                lemm_line += clean_lemma + ' '
                raw_text_line += ' ' + word[1]
            else:
                raw_text_line += word[1]

        lemm_sentences_list.append(lemm_line.strip())
        sentences_list.append(raw_text_line.strip())
        #print()
    return lemm_sentences_list, sentences_list
    
def get_tf_idf_dict(lemm_text_list):
    vect = TfidfVectorizer()#stop_words = russian_stopwords
    tfidf_matrix = vect.fit_transform(lemm_text_list)
    df = pd.DataFrame(tfidf_matrix.toarray(), columns = vect.get_feature_names())
    #print(df.head())
    # if (save_to_csv): df.to_csv("./text_0_tfidf.xlsx", sep = '\t')
    tf_idf_dict = df.to_dict()
    return tf_idf_dict
    
def get_verb_prop(word, morph):
    analysis = morph.parse(word)[0]
    try: 
        if (analysis.tag.POS == 'GRND' ):
            return 'GRND'
        elif("PRT" in analysis.tag.POS):
            return 'PRT'
        else:
            return None
    except:
        return None
        
def create_map(conllu_map, tf_idf_dict):
    morph = pymorphy2.MorphAnalyzer()
    text_map = []
    sentence_ind = 0
    #bar = progressbar.ProgressBar(maxval=len(conllu_map),widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    #bar.start()
    
    for sentence in conllu_map:
        sentence_map = []
        for word in sentence: 
            if (word[3] != 'PUNCT'):
                clean_lemma = ''
                for char in word[2]:
                    if char not in full_punctuation:
                        clean_lemma += char.lower()
                # weight = OrderedDict([("word", word[1]),("lemma",clean_lemma), ("vocabulary_prop",(OrderedDict([("tf_idf", 0),("nominal_index",word[0])]))),
                #                      ("grammar_prop", OrderedDict([('pos',word[3] )]))])#,("lex_vector",None)
                weight = {"word":word[1], "lemma":clean_lemma, "vocabulary_prop":{"tf_idf":0, "nominal_index": word[0]},
                          "grammar_prop": {"pos": word[3]}}

                if(word[3] == "VERB"):
                    verb_prop = get_verb_prop (word[1], morph)
                    if verb_prop:
                        weight['grammar_prop']['verb_prop'] = verb_prop
                if (word[3] == "NOUN"):
                    if("Case" in word[5]):
                        grammar = word[5].split("|")
                        for gr in grammar:
                            if("Case" in gr):
                                case = gr.split("=")[1]
                                weight['grammar_prop']['case'] = case
                if (clean_lemma in tf_idf_dict):
                    weight["vocabulary_prop"]["tf_idf"] = tf_idf_dict[clean_lemma][sentence_ind]
                sentence_map.append(weight)
        text_map.append(sentence_map)
        sentence_ind += 1
    return text_map
    
def get_dependencies (conllu_map, text_map_input):
    sentence_map = []
    assert len(conllu_map) == len(text_map_input) #sentences count is equal
    text_map = copy.deepcopy(text_map_input)
    for sentence, text_map_sentence in zip(conllu_map,text_map):
        # one_sentence_map = OrderedDict([("spec_sentence_features",(OrderedDict([("negation", 0),('coreference',0),("vozvr_verb",0),("total_vozvr",0),
        #                                                                         ("prich",0),("total_prich",0),("deepr",0),("total_deepr",0),
        #                                                                ("case_complexity",0),("total_case",0)]))), ("syntax_prop",OrderedDict()),
        #                                  ("sentence_words", [])])#("average_vocabulary", []),
        nominal2real_index_dict = {}

        one_sentence_map = {"spec_sentence_features":{"negation":0,"coreference":0,"vozvr_verb":0,"total_vozvr":0,"prich":0,"total_prich":0,
                                                             "deepr":0,"total_deepr":0,"case_complexity":0,"total_case":0},"syntax_prop":{},
                                   "sentence_words":[]}
        real_index = 1

        for word in sentence:
            if (word[3] != 'PUNCT'):
                nominal2real_index_dict[int(word[0])] = int(real_index)
                real_index += 1
                #print(word[1], "head_word_nominal_index =", word[6])

        #print(nominal2real_index_dict)
        distances_list= []
        for word in sentence:
            if (word[3] != 'PUNCT'):
                #print("DIST CALC PROCESS")
                head_nominal_index = word[6]
                if (int(head_nominal_index) != 0 and int(head_nominal_index) in nominal2real_index_dict):
                    current_element_real_index = nominal2real_index_dict[int(word[0])]
                    head_element_real_index = nominal2real_index_dict[int(head_nominal_index)]
                    distance = abs(current_element_real_index - head_element_real_index)
                    distances_list.append(distance)
                    #print(word[1],current_element_real_index,  head_element_real_index)
                else:
                    #print(head_nominal_index, "NOT IN DIST LIST")
                    pass
        #print(distances_list)
        one_sentence_map["syntax_prop"]["distances_list"] = distances_list
        sentence_vocab_importance = 0
        for map_word in text_map_sentence:
            sentence_vocab_importance += map_word["vocabulary_prop"]["tf_idf"]
            one_sentence_map["sentence_words"].append(map_word)
        one_sentence_map["syntax_prop"]["sent_vocab_imp"] = sentence_vocab_importance
        sentence_map.append(one_sentence_map)
    return sentence_map
    
#vector function here
def update_with_vectors(text_map_input):
    text_map = copy.deepcopy(text_map_input)
    for sentence in text_map:
        sentence['words_vectors_list'] = []
        for ind in range (len(sentence['sentence_words'])):
            sentence['collocation_vectors_list'] = []
            lemma = sentence['sentence_words'][ind]['lemma']

            if lemma in word2vec:
                sentence['words_vectors_list'].append(word2vec[lemma])
            else:
                sentence['words_vectors_list'].append(np.zeros(300).tolist())

            #print("lemmas", lemma)
            # w2v = fasttext[lemma]
            # if lemma in lyashevskaya_freq_dict:
            #     w2v = np.hstack([w2v, lyashevskaya_freq_dict[lemma]]).tolist()
            # else:
            #     w2v = np.hstack([w2v, 0]).tolist()
            # #print(w2v)
            # sentence['words_vectors_list'].append(w2v)\

            # sentence['words_vectors_list'].append(np.zeros(300).tolist())
            # word_key = "w2v_" + lemma
            # w2v = redis_db.lrange(word_key, 0, 1)
            # if len(w2v) > 0 and w2v is not None:
            #     #print(type(w2v))
            #     # if lemma in lyashevskaya_freq_dict:
            #     #     w2v = np.hstack([w2v, lyashevskaya_freq_dict[lemma]]).tolist()
            #     # else:
            #     #     w2v = np.hstack([w2v, 0]).tolist()
            #     #print(w2v)
            #     w2v = get_redis_list(word_key)
            #     sentence['words_vectors_list'].append(w2v)
            #     #raise "w2v"
            # else:
            #     sentence['words_vectors_list'].append(np.zeros(300).tolist())

    return text_map
           
def increment_dict(dict_name, property_name, value):
    if property_name in dict_name:
        dict_name[property_name] += value
    else:
        dict_name[property_name] = value
   
def features_extraction(sentence_map_input):
    sentence_map = copy.deepcopy(sentence_map_input) 
    for sentence in sentence_map:
        previous_word_is_noun = False
        previous_noun_case = None
        previous_noun_vocab_importance = 0 
        
        current_sentence_vocab_vectors = []
        
        if len(sentence['syntax_prop']['distances_list']) > 0:
            sentence['spec_sentence_features']['mean_depend_length'] = mean(sentence['syntax_prop']['distances_list']) * 0.1
        else:
            sentence['spec_sentence_features']['mean_depend_length'] = 0
        
        for word in sentence['sentence_words'] :
            #current_lex_vector = word['lex_vector']
            #current_sentence_vocab_vectors.append(current_lex_vector)
            
            if word['lemma'] == 'который' or word['lemma'] == 'это' or word['lemma'] == 'этот':
                #spec_word_partial_importance = word['vocabulary_prop']['tf_idf']/sentence['syntax_prop']['sent_vocab_imp']
                increment_dict(sentence['spec_sentence_features'],'coreference', 1/len(sentence['sentence_words']))
            elif word['lemma'] == 'бы' or word['lemma'] == 'не' or word['lemma'] == 'ни':
                increment_dict(sentence['spec_sentence_features'],'negation', 1/len(sentence['sentence_words']))
            elif word['grammar_prop']['pos'] == 'VERB':
                if word['word'].endswith('ся') or word['word'].endswith('ся'):
                    spec_word_partial_importance = word['vocabulary_prop']['tf_idf']/sentence['syntax_prop']['sent_vocab_imp']
                    increment_dict(sentence['spec_sentence_features'], 'total_vozvr', word['vocabulary_prop']['tf_idf'])
                    increment_dict(sentence['spec_sentence_features'], 'vozvr_verb', spec_word_partial_importance)
                    
            if 'verb_prop' in word['grammar_prop']:
                if word['grammar_prop']['verb_prop'] == 'PRT':
                    spec_word_partial_importance = word['vocabulary_prop']['tf_idf']/sentence['syntax_prop']['sent_vocab_imp']
                    increment_dict(sentence['spec_sentence_features'], 'total_prich', word['vocabulary_prop']['tf_idf'])
                    increment_dict(sentence['spec_sentence_features'], 'prich', spec_word_partial_importance)
                elif word['grammar_prop']['verb_prop'] == 'GRND':
                    spec_word_partial_importance = word['vocabulary_prop']['tf_idf']/sentence['syntax_prop']['sent_vocab_imp']
                    increment_dict(sentence['spec_sentence_features'], 'total_deepr', word['vocabulary_prop']['tf_idf'])
                    increment_dict(sentence['spec_sentence_features'], 'deepr', spec_word_partial_importance)
                    
            if (word['grammar_prop']['pos'] == 'NOUN'):       
                if previous_word_is_noun == True and 'case' in word['grammar_prop']:
                    if previous_noun_case != word['grammar_prop']['case']:
                        total_importance = previous_noun_vocab_importance + word['vocabulary_prop']['tf_idf']
                        #print(sentence)
                        if (sentence['syntax_prop']['sent_vocab_imp'] > 0):
                            spec_word_partial_importance = total_importance/sentence['syntax_prop']['sent_vocab_imp'] 
                        increment_dict(sentence['spec_sentence_features'], 'total_case', total_importance)
                        increment_dict(sentence['spec_sentence_features'], 'case_complexity', spec_word_partial_importance)
                elif('case' in word['grammar_prop']):
                    #передаем инфу для следующего потенциального существительного
                    #print(word)
                    previous_word_is_noun = True
                    previous_noun_case = word['grammar_prop']['case']
                    previous_noun_vocab_importance = word['vocabulary_prop']['tf_idf']
            else:
                previous_word_is_noun = False
        
        current_sentence_vocab_vectors = np.matrix(current_sentence_vocab_vectors)
        """
        mean_sentence_vocab_vector = current_sentence_vocab_vectors.mean(0)
        mean_sentence_vocab_vector = mean_sentence_vocab_vector.tolist()
        sentence['average_vocabulary'] = mean_sentence_vocab_vector[0]
        """
    return sentence_map
    
def calculate_lix_from_list_of_sentences(processed_text_sentences):
    sentences_count = len(processed_text_sentences)
    words_count = sum([len(line.split(' ')) for line in processed_text_sentences])
    long_words_count = 0 #more than 6
    for line in processed_text_sentences:
        for word in line.split():
            if len(word) > 6:
                long_words_count += 1
    lix = words_count/ sentences_count + (long_words_count * 100) / words_count

    return round(lix,2)
        
def calculate_type_token_ratio(lemm_text_sentences):
      all_words = []
      for sentence in lemm_text_sentences:
          words = sentence.split()
          for word in words:
              all_words.append(word)

      unqie_words = set(all_words)
      types = len(unqie_words)
      tokens = len (all_words)

      return round(types/tokens,2)
      
def text_features_cal(sentence_map, orig_sentences_list, lemm_sentences_list, raw_text):
    # print("sentence_map", sentence_map)
    # print("orig_sentences_list", orig_sentences_list)
    text_map = {"lix":0, "ttr":0, "raw_text":[],"average_sent_properties":[],"sentences_map":sentence_map,
                "raw_sentences_list":orig_sentences_list}
    lix = calculate_lix_from_list_of_sentences(orig_sentences_list)
    ttr = calculate_type_token_ratio(lemm_sentences_list)
    text_map['lix'] = lix *0.01
    text_map['ttr'] = ttr
    text_map['sentences_count'] = 0
    text_map['average_sentence_length'] = 0
    text_map['raw_text'] = raw_text
    sentence_ind = 0
    words_count = 0
    for sentence in sentence_map:
        sentencce_json = {}
        sentencce_json[sentence_ind] = []
        words_count += len(sentence['sentence_words'])
        # for word_element in sentence['collocation_index_list']:
        #     sentencce_json[sentence_ind].append((word_element[0],word_element[1][0]))
        
        sentence_ind += 1
    #
    sentences_count = 0
    negation_count = 0
    coreference_count = 0
    #
    overall_vocab_importance = 0
    vozvr_verb_importance = 0
    prich_verb_importance = 0 
    deepr_verb_importance = 0
    case_complexity_importance = 0 
    #
    synt_distance = 0
    
    vocabulary_vectors = []
    for sentence in sentence_map:
        sentences_count += 1
        
        synt_distance += sentence['spec_sentence_features']['mean_depend_length']
        
        if(sentence['spec_sentence_features']['negation'] > 0  ):
            negation_count += 1
        
        if(sentence['spec_sentence_features']['coreference'] > 0  ):
            coreference_count += 1
        
        overall_vocab_importance += sentence['syntax_prop']['sent_vocab_imp']
        
        vozvr_verb_importance += sentence['spec_sentence_features']['total_vozvr']
        prich_verb_importance += sentence['spec_sentence_features']['total_prich']
        deepr_verb_importance += sentence['spec_sentence_features']['total_deepr']
        case_complexity_importance += sentence['spec_sentence_features']['total_case']

    text_map['sentences_count'] = sentences_count * 0.01
    text_map['average_sentence_length'] = words_count/sentences_count * 0.01

    text_map['average_sent_properties'].append(negation_count/sentences_count)#negation_count
    text_map['average_sent_properties'].append(coreference_count/sentences_count)#coreference_count
    text_map['average_sent_properties'].append(vozvr_verb_importance/overall_vocab_importance)
    text_map['average_sent_properties'].append(prich_verb_importance/overall_vocab_importance)
    text_map['average_sent_properties'].append(deepr_verb_importance/overall_vocab_importance)
    text_map['average_sent_properties'].append(case_complexity_importance/overall_vocab_importance)
    text_map['average_sent_properties'].append(synt_distance/sentences_count)
    
    
    return text_map
    
    #"D:\input\music_smart_colloc_freq.json"
    #"C:\Autotutor\improved_approach\colloc\music_unigr_freq.json"
def get_text_map(text, raw_text_input = False):
    model = Model(os.path.join(BASE,'processing_data','russian-syntagrus-ud-2.0-170801.udpipe'))
    if raw_text_input:
        raw_text = text 
    else:
        raw_text = read_text(text)
    conllu = get_conllu_from_unite_line_text(raw_text, model)
    conllu_text_map = get_conllu_text_map(conllu)
    lemm_sentences,sentences_list = get_lemm_and_orig_text_from_udmap(conllu_text_map)
    tf_idf_dict = get_tf_idf_dict (lemm_sentences)
    text_map = create_map(conllu_text_map, tf_idf_dict)
    sentence_map_dep =  get_dependencies(conllu_text_map, text_map)
    #print(sentence_map_dep)
    #sentence_map_colloc = update_with_vectors (sentence_map_dep,colloc_db,unigramm_db)
    sentence_map_colloc = update_with_vectors (sentence_map_dep)
    sentence_map_feat = features_extraction(sentence_map_colloc) 
    json_text_map = text_features_cal(sentence_map_feat, sentences_list, lemm_sentences, raw_text)
    #print("json_text_map",json_text_map['raw_text'])
    return json.dumps(json_text_map).encode("utf-8")
    
   
# text = """Указом президента России Бориса Ельцина внесены изменения в  существующую структуру Федеральной службы безопасности РФ, утвержденную в июле прошлого года. Как говорится в поступившем сегодня сообщении Центра общественных связей ФСБ, в соответствии с Основами (Концепцией) государственной политики Российской Федерации по военномустроительству на период до 2005 года, на базе Департамента по борьбе с терроризмом и Управления конституционной безопасности ФСБ создан Департамент по защите конституционного строя и борьбе с терроризмом. В составе департамента организуются три управления с четко определенной компетенцией. В ФСБ отмечают, что "в современных условиях для российскойгосударственности имеют приоритетное значение вопросы защитыконституционного строя, сохранения целостности страны, борьбыс терроризмом и всеми формами экстремизма, а также разведывательно-подрывной деятельностью спецслужб и организаций иностранных государств". Как подчеркивается в сообщении, "органам безопасности в решении данных проблем отведена особая роль"""
#
# text_short = """Однажды в поликлинику пришел больной.
# – Что у вас болит? – спросил врач.
# – У меня болит живот, – ответил молодой человек.
# – Что вы ели вчера?
# – Зеленые яблоки.
# – Хорошо. Я дам вам лекарство для глаз, – сказал врач больному.
# – Почему для глаз? Ведь у меня болит живот? – удивился молодой человек.
# – Я дам вам лекарство для глаз, чтобы вы лучше видели, что вы едите, – сказал врач.
# """
#json_text_map = get_text_map(text, raw_text_input = True)


# json_text_map = get_text_map(text,raw_text_input = True)

# with open("text_map_improved_example.json", "w") as f:
#     json.dump(json_text_map,f, indent = 4, ensure_ascii = False)

# print( json_text_map['average_sent_properties'])

# for sent in json_text_map['sentences_map']:
#     #print(sent["collocation_index_list"],'\n')
#     print(sent["spec_sentence_features"],'\n')
#     print("~~~~")
#     for word in sent["sentence_words"]:
#         print(word['word'],word['vocabulary_prop'], word['grammar_prop'])
#         print("\n")
#     print ("====================")
