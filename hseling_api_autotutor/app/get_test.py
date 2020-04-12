from flask import Blueprint, request
import json
from .recommendation_logic.get_questions_to_most_complicated_texts import get_complicated_texts

get_test = Blueprint('get_test', __name__)


@get_test.route('/gettest', methods=['GET'])
def gettest():
    data = request.data.decode()
    data = json.loads(data)
    test_sentences_json = get_complicated_texts(data['username'])
    # print("test_sentences_json", test_sentences_json, type(test_sentences_json))
    return test_sentences_json
