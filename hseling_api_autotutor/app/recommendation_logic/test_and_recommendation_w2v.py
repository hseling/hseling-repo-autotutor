#from text_processing_udpipe_w2v import get_text_map
#from ud_class import Model
import random
import numpy as np
from collections import OrderedDict
from ..database import redis_db

"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Perceptron, SGDRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
"""

class user_vector:
    def __init__(self,user_id, debug = False):
        self.debug = debug
        self.user_id = user_id
        self.vocab_features =[]
        self.sentence_features =[]
        #coreference_items, negation_items, sent_special_pos, dependencies_length, Y (answer)                   
        self.text_features = [] #OrderedDict([("lix",[]),("ttr",[])])
        self.answers_count = OrderedDict([("correct_answers",0),("incorrect_answers",0)])
        # self.trigramms_list = []
        
    def start_new_text(self):
        self.answers_count['correct_answers'] = 0
        self.answers_count['incorrect_answers'] = 0
        if self.debug:print("answers count has been reset", self.answers_count['correct_answers'], self.answers_count['incorrect_answers'])
        
    def end_text(self, text_map):
        if self.debug:
            print("\n========")
            print("SUM UP TEXT VALUES")
            print("========\n")
        correct_answers_rate = round(self.answers_count['correct_answers'] / (self.answers_count['correct_answers'] 
                                                                              + self.answers_count['incorrect_answers']),2)
        current_text_features = []
        if self.debug: 
            print("answers_count", self.answers_count)
        current_text_features.append(text_map['lix'])
        current_text_features.append(text_map['ttr'])
        #current_text_features.extend(text_map['vocab_properties'])
        current_text_features.extend(text_map['average_sent_properties'])
        current_text_features.append(correct_answers_rate)
        self.text_features.append(current_text_features)
        if self.debug: print("TEXT FEATURES",self.text_features)
        
    def update_vector_with_answer_sentence(self, sentence_map, correct_answer):
        #update setnence and text features
        if self.debug:
            print("\n===NEW REPLY CALCULATION====")
            print("\n========")
            print("ADDING SENTENCE RESULTS")
            print("========\n")
            
        #update setnence and text features
        if correct_answer == True:
            answer_value = 1
            self.answers_count['correct_answers'] += 1
            if self.debug: print("Answer for this question is correct")
        else:
            answer_value = 0
            self.answers_count['incorrect_answers'] += 1
            if self.debug: print("Answer for this question is incorrect")
        if self.debug:print("check answers count", self.answers_count['correct_answers'], self.answers_count['incorrect_answers'])
        current_sentence_features = []
        current_sentence_features.append(sentence_map['spec_sentence_features']['negation'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['coreference'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['vozvr_verb'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['prich'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['deepr'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['case_complexity'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['mean_depend_length'])
        #current_sentence_features.extend(sentence_map['average_vocabulary'])
        current_sentence_features.append(answer_value)#target variable
        self.sentence_features.append(current_sentence_features)
        
        if self.debug: print("SENTENCE FEATURES", current_sentence_features)
        
        if self.debug:
            print("\n========")
            print("ADDING VOCABULARY RESULTS")
            print("========\n")
            
        understanding_importance_list = []
        for word_w in sentence_map['sentence_words']:
            understanding_importance = word_w['vocabulary_prop']['tf_idf']
            understanding_importance_list.append(understanding_importance)
        understanding_importance_list_normalized = [el/sum(understanding_importance_list) for el in understanding_importance_list]
        #print(understanding_importance_list_normalized)

        #print(len(sentence_map['sentence_words']), len(sentence_map['words_vectors_list'])) 
        assert len(sentence_map['sentence_words']) == len(sentence_map['words_vectors_list']) == len(understanding_importance_list_normalized)
        for word_index in range(len(sentence_map['sentence_words'])):
            current_lex_vector = []
            #print("current_element", current_element)
            current_lex_vector.extend(sentence_map['words_vectors_list'][word_index])
            if (correct_answer == True): 
                current_lex_vector.append(understanding_importance_list_normalized[word_index])
            elif (correct_answer == False):
                current_lex_vector.append(-1 * understanding_importance_list_normalized[word_index])
            self.vocab_features.append(current_lex_vector)
        # for word_w in sentence_map['sentence_words']:
        #     understanding_importance = word_w['vocabulary_prop']['tf_idf']
        #     understanding_importance_sum += understanding_importance
        #     understanding_importance_list.append([word_w['lemma'], understanding_importance,word_w['lex_vector'],word_w['lex_trigram']])
            
        # for un_unit in understanding_importance_list:
        #     if(understanding_importance_sum > 0):
        #         un_unit[1] /= understanding_importance_sum
        # #if self.debug:print("understanding_importance_list", understanding_importance_list)
                       
        
        # for unit_index in range(len(understanding_importance_list) ):
        #     current_element = understanding_importance_list[unit_index][2]
        #     """
        #     left_unit_index = unit_index - 1
        #     if left_unit_index <0:
        #         left_element = 300 * [0] 
        #     else:
        #         left_element = understanding_importance_list[left_unit_index][2]
        #     right_unit_index = unit_index + 1
        #     if right_unit_index >=  len(understanding_importance_list):
        #         right_element = 300 * [1]
        #     else:
        #         right_element = understanding_importance_list[right_unit_index][2]
        #      """   
        #     current_lex_vector = []
        #     #print("current_element", current_element)
        #     current_lex_vector.extend(current_element)
            
        #     if (correct_answer): 
        #         current_lex_vector.append(understanding_importance_list[unit_index][1])
        #         self.trigramms_list.append(understanding_importance_list[unit_index][3])
        #         #current_lex_vector.append(understanding_importance_list[unit_index][3])
        #         self.vocab_features.append(current_lex_vector)
        #         #print("current_lex_vector", current_lex_vector)
        #     else:
        #         current_lex_vector.append(-1 * understanding_importance_list[unit_index][1])
        #         self.trigramms_list.append(understanding_importance_list[unit_index][3])
        #         #current_lex_vector.append(understanding_importance_list[unit_index][3])
                # self.vocab_features.append(current_lex_vector)

    def export_user_db(self):
        # words_db = np.array([np.array(word) for word in self.vocab_features])
        def save_feature(entity_name, self_features, user_id):
            entity_feature_size_id = user_id + "_" + entity_name + "_features_size"
            # print("process ", entity_name)
            entity_name_features_number = redis_db.get(user_id + "_" + entity_name + "_features_size")
            print("current entity_name_features_number", entity_name_features_number)
            if entity_name_features_number is None:
                entity_name_features_number = 0
                redis_db[entity_feature_size_id] = 0
            else:
                entity_name_features_number = int(entity_name_features_number)
            # for vocab_feat_ind in range(entity_name_features_number, entity_name_features_number + len(self.vocab_features)): !!!!!!ЗАМЕНИТЬ НА ЭТО НА ПРОДЕ!!!!!! И ВЕЗДЕ В ПРЕДЖДЛОЖЕНИЯХ И ТЕКСТАХ
            # print("will iterate over ", len(self_features))
            for entity_feat_ind_in_current_test_results in range(len(self_features)):
                global_fature_index = str(entity_name_features_number + entity_feat_ind_in_current_test_results - 1)
                feature_index = user_id + "_" + entity_name + "_feature_" + global_fature_index
                check_existese = redis_db.lrange(feature_index, 0, 1)
                # print("existence",feature_index,  check_existese)
                if len(check_existese) == 0 or check_existese is None:
                    for entity_feat in self_features[entity_feat_ind_in_current_test_results]:
                        redis_db.rpush(feature_index, entity_feat)
                    entity_name_features_number += 1
            print("entity_feature_size_id", entity_feature_size_id, entity_name_features_number)
            redis_db[entity_feature_size_id] = entity_name_features_number
            # print("current total", entity_name, redis_db[entity_feature_size_id])


        # word_features_number = redis_db.get(self.user_id + "_word_features_size")
        # if word_features_number is None:
        #     word_features_number = 0
        #     redis_db[self.user_id + "_word_features_size"] = 0
        # else:
        #     word_features_number = int(word_features_number)
        # # for vocab_feat_ind in range(word_features_number, word_features_number + len(self.vocab_features)): !!!!!!ЗАМЕНИТЬ НА ЭТО НА ПРОДЕ!!!!!! И ВЕЗДЕ В ПРЕДЖДЛОЖЕНИЯХ И ТЕКСТАХ
        #
        # for vocab_feat_ind in range(len(self.vocab_features)):
        #     feature_index = self.user_id + "_word_feature_" + str(vocab_feat_ind)
        #     check_existese = redis_db.lrange(feature_index, 0, 1)
        #     if check_existese is not None:
        #         redis_db.delete(feature_index)
        #         rewrite_word_feat = True
        #     else:
        #         rewrite_word_feat = False
        #         pass
        #     for vocab_feat in self.vocab_features[vocab_feat_ind]:
        #         redis_db.rpush(self.user_id + "_word_feature_"+str(vocab_feat_ind), vocab_feat)
        #     if rewrite_word_feat == False:
        #         redis_db[self.user_id + "_word_features_size"] = int(redis_db[self.user_id + "_word_features_size"]) + 1
        save_feature("word", self.vocab_features, self.user_id)

        # ============
        # sent_features_number = redis_db.get(self.user_id + "_sent_features_size")
        # if sent_features_number is None:
        #     sent_features_number = 0
        #     redis_db[self.user_id + "_sent_features_size"] = 0
        # else:
        #     sent_features_number = int(sent_features_number)
        # # for sent_feat_ind in range(sent_features_number, sent_features_number + len(self.sentence_features)):
        # for sent_feat_ind in range(len(self.sentence_features)):
        #     try:
        #         redis_db.delete(self.user_id + "_sent_feature_" + str(sent_feat_ind))
        #         rewrite_sent_feat = True
        #     except:
        #         rewrite_sent_feat = False
        #         pass
        #     for sent_feat in self.sentence_features[sent_feat_ind]:
        #         redis_db.rpush(self.user_id + "_sent_feature_" + str(sent_feat_ind), sent_feat)
        #     if rewrite_sent_feat == False:
        #         redis_db[self.user_id + "_sent_features_size"] = int(redis_db[self.user_id + "_sent_features_size"]) + 1

        # sentence_db = [sent for sent in self.sentence_features]
        # sentence_db_path = self.user_id + '_sentence_db'
        # redis_db[sentence_db_path] = sentence_db
        save_feature("sent", self.sentence_features, self.user_id)
        # ============
        text_features_number = redis_db.get(self.user_id + "_text_features_size")
        # if text_features_number is None:
        #     text_features_number = 0
        #     redis_db[self.user_id + "_text_features_size"] = 0
        # else:
        #     text_features_number = int(text_features_number)
        # # for text_feat_ind in range(text_features_number, text_features_number + len(self.text_features)):
        # for text_feat_ind in range(len(self.text_features)):
        #     try:
        #         redis_db.delete(self.user_id + "_text_feature_" + str(text_feat_ind))
        #         rewite_text_feat = True
        #     except:
        #         rewite_text_feat = False
        #         pass
        #     for text_feat in self.text_features[text_feat_ind]:
        #         redis_db.rpush(self.user_id + "_text_feature_" + str(text_feat_ind), text_feat)
        #     if rewite_text_feat == False:
        #         redis_db[self.user_id + "_text_features_size"] = int(redis_db[self.user_id + "_text_features_size"]) + 1
        save_feature("text", self.text_features, self.user_id)
        # text_db = [text for text in self.text_fearues]
        # text_db_path = self.user_id + '_text_db'
        # redis_db[text_db_path] = text_db

        # words_db = np.array([np.array(word) for word in self.vocab_features])
        # word_db_path = self.user_id + '_word_db.csv'
        # np.savetxt(word_db_path, words_db, delimiter=',')
        #
        # sentence_db = np.array([np.array(sent) for sent in self.sentence_features])
        # sentence_db_path = self.user_id + '_sentence_db.csv'
        # np.savetxt(sentence_db_path, sentence_db, delimiter=',')
        #
        # text_db = np.array([np.array(text) for text in self.text_fearues])
        # text_db_path = self.user_id + '_text_db.csv'
        # np.savetxt(text_db_path, text_db, delimiter=',')


    """
    def export_user_vector(self):
        #vocabulary vector
        
        #y_words = self.vocab_features[:,-1]
        words_db = np.array([np.array(word) for word in self.vocab_features])
        #words_db = np.matrix(words_db)
        X_words = words_db[:,:-1]
        y_words = words_db[:,-1]
        
        if self.debug:
            print("\n========")
            print("VOCAB VECTOR CALCULCATION")
            print("========\n")
            print("word db example\n",words_db[1,:])
        X_train, X_test, y_train, y_test = train_test_split(X_words, y_words, test_size=0.15)
        words_feat_reg = SGDRegressor(max_iter=100, tol=1e-3)
        words_feat_reg.fit(X_train, y_train)
        accuracy = words_feat_reg.score(X_test, y_test)
        
        rf_vocab = RandomForestRegressor(max_depth=2, random_state=0,
                              n_estimators=100)
        rf_vocab.fit(X_train, y_train)
        rf_accuracy = rf_vocab.score(X_test, y_test)
        
        vocab_lin_reg = LinearRegression()
        vocab_lin_reg.fit(X_train, y_train)
        linreg_accuracy = rf_vocab.score(X_test, y_test)
        
        vocab_model = rf_vocab
        
        model = Sequential()
        model.add(Dense(12, input_dim=300, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=150, batch_size=10)
        y_test = model.predict(X_test)
        #keras.metrics.categorical_accuracy(y_true, y_pred)
        
        if self.debug:
            print("SGDRegressor model trained with accuracy = ", accuracy)
            print("RandomForestRegressor model trained with accuracy = ", rf_accuracy)
            print("LinearRegression model trained with accuracy = ", linreg_accuracy)
        #sentence vector
        sentence_db = np.array([np.array(sent) for sent in self.sentence_features])
        #sentence_db = np.matrix(self.sentence_features)
        X_sent = sentence_db[:,:-1]
        y_sent = sentence_db[:,-1]
        sent_feat_reg = LinearRegression().fit(X_sent, y_sent)
        if self.debug:
            print("\n========")
            print("SENTENCE VECTOR CALCULCATION")
            print("========\n")
            print("sentence db example\n",sentence_db[1,:])
        sentence_model = sent_feat_reg
        
        #text vector
        if self.debug:
            print("\n========")
            print("TEXT VECTOR CALCULCATION")
            print("========\n")
        
        print()
        text_db = np.array([np.array(text) for text in self.text_fearues])
        #text_db = np.matrix(self.text_fearues)
        X_text = text_db[:,:-1]
        y_text = text_db[:,-1]
        text_feat_reg = LinearRegression().fit(X_text, y_text)
        if self.debug: 
            print("text db example\n", text_db[1,:])
        text_model = text_feat_reg
            
        return vocab_model, sentence_model, text_model
        """
        
        