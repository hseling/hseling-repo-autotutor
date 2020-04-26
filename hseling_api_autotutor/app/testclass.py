import json
from .database import redis_db


def get_redis_list(redis_list_key):
    python_standard_list = []
    for i in range(0, redis_db.llen(redis_list_key)):
        # print(redis_db.lindex(redis_list_key, i))
        python_standard_list.append(float(redis_db.lindex(redis_list_key, i)))
    return python_standard_list

def add_to_dictionary(key, value):
    key_type_storage_size = key + "_storage_size"
    key_storage_index = int(redis_db.get(key_type_storage_size) or 0)
    redis_db[key_type_storage_size] = key_storage_index + 1
    type_storage_index = key + "_" + str(key_storage_index)
    redis_db[type_storage_index] = value
    return type_storage_index

def add_to_set(set_reference_name, value):
    present_in_db = redis_db.get(set_reference_name)
    if present_in_db is None:
        redis_db[set_reference_name] = value
    else:
        # print(redis_db[set_reference_name], type(redis_db[set_reference_name]))
        handled_text_types = present_in_db.decode("utf-8")
        available_set_element = set(handled_text_types.split("&"))
        # print(available_set_element)
        if value not in available_set_element:
            extended_handled_types_list = handled_text_types + "&" + value
            redis_db[set_reference_name] = extended_handled_types_list

    # reference_plus_name = set_reference_name + "_" + value
    # present_in_db = redis_db.get(reference_plus_name)
    # if present_in_db is None:
    #     redis_db[reference_plus_name] = 1

def get_set_elements(set_reference_name):
    present_in_db = redis_db.get(set_reference_name)
    if present_in_db is None:
        print("no such set")
    else:
        handled_text_types = redis_db[set_reference_name].decode("utf-8")
        available_set_element = set(handled_text_types.split("&"))
        return available_set_element

def save_to_db(text_map, map_type="user_generated", require_status=True):
    # id = int(redis_db.get('size') or 0)
    # print("will_be_saved_to_id", id)
    # redis_db['size'] = id + 1
    # redis_db[id] = self.data
    # self.id = id
    # print("current_state_is", redis_db[id])
    add_to_set("text_map_types", map_type)
    text_map_id = add_to_dictionary(key="text_maps_" + map_type, value=text_map)
    # text_map_type = "text_maps_" + map_type
    # text_map_type_storage_size = text_map_type +"_size"
    #
    # text_map_id = int(redis_db.get(text_map_type_storage_size) or 0)
    # redis_db['text_maps_storage_size'] = text_map_id + 1
    # text_map_id = "text_map_" + str(text_map_id)
    # redis_db[text_map_id] = text_map
    if require_status == True:
        return {"status": "saved at" + text_map_id}


def get_text_by_id(id, text_type="user_generated"):
    text_map_id = "text_maps_" + text_type + "_" + str(id)
    text = redis_db.get(text_map_id)
    if text is None:
        return {'error': 'not exist'}
    text = json.loads(text)
    return text['raw_text']


# def get_test_by_id(id):
#     text_map_id = "text_map_" + str(id)
#     text_map = redis_db.get(text_map_id)
#     if text_map is None:
#         return {'error': 'not exist'}
#     test = json.loads(text_map)
#     return test['raw_text']


class TestClass():
    """
    curl --header "Content-Type: application/json; charset=UTF-8"
    --request POST
    --data '{"type":"with choice","question":"Вопросик 1", "answers": {"a":"1", "b":"2"}, "true answer":"a"}'
    http://localhost:5000/addtest

    """

    @classmethod
    def check_test(self, data):
        print(data)
        data = json.loads(data)
        if 'student answer' not in data:
            return {'error': u'нет ответа'}

        if 'student_id' not in data:
            return {'error': u'нет id студента'}

        test = redis_db.get(id)
        if test is None:
            return {'error': u'тест не существует'}

        if data.get('type') == 'with choice':
            if data['student answer'] == test['true answer']:
                return {'status': u'OK', }

        test.pop('true answer')
        return test

    def save_to_db(self):
        # id = int(redis_db.get('size') or 0)
        # print("will_be_saved_to_id", id)
        # redis_db['size'] = id + 1
        # redis_db[id] = self.data
        # self.id = id
        # print("current_state_is", redis_db[id])
        text_map_id = int(redis_db.get('text_maps_storage_size') or 0)
        redis_db['text_maps_storage_size'] = text_map_id + 1
        text_map_id = "text_map_" + str(text_map_id)
        redis_db[text_map_id] = self.data
        return text_map_id

    def parse_from_str(self, jsonstr):
        # import pdb; pdb.set_trace()
        data = json.loads(jsonstr)
        self.error = ''
        # введем типы вопросов.
        if data.get('type') is None:
            self.error = 'не указан тип'
            return
        elif data.get('type') == 'with choice':
            # проверка, что есть вопрос
            if data.get('question') is None or not isinstance(data.get('question'), str):
                self.error = 'вопрос не распознан'
                return
            # проверка вариантов ответов
            if data.get('answers') is None or not isinstance(data.get('answers'), dict):
                self.error = 'ответы не обнаружены не распознаны'
                return
            for key, val in data.get('answers').items():
                if not isinstance(key, str) or not isinstance(val, str):
                    self.error = 'ответы не обнаружены не распознаны'
                    return
            # проверка истинного ответа
            if data.get('true answer') is None or not isinstance(data.get('true answer'), str) or data.get(
                    'true answer') not in data.get('answers'):
                self.error = 'правильный ответ не распознан'
                return


        else:
            self.error = 'Тип ' + data.get('type') + ' неизвестен'
            return

        self.data = jsonstr

    @classmethod
    def get_test_by_id(self, id):
        text_map_id = "text_map_" + str(id)
        test = redis_db.get(text_map_id)
        if test is None:
            return {'error': 'not exist'}

        test = json.loads(test)

        test.pop('true answer')
        return test
