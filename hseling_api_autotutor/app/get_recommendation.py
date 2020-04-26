from flask import Blueprint, request
import json
from .recommendation_logic.get_answers_return_recommendation import send_test_results,get_reading_recommendation,is_enough_tests_for_recommendation
import json

send_results = Blueprint('send_results', __name__)
get_recommendation = Blueprint('get_recommendation', __name__)

@send_results.route('/send_test_results', methods=['POST'])
def send_res():
    data = request.data.decode()
    data = json.loads(data)
    test_sentences_json = send_test_results(data)#data['answers']
    return json.dumps(test_sentences_json, ensure_ascii=False)

@get_recommendation.route('/get_recommendation', methods=['GET'])
def get_rec():
    data = request.data.decode()
    data = json.loads(data)
    test_sentences_json = get_reading_recommendation(data['username'])
    return json.dumps(test_sentences_json, ensure_ascii=False)

@get_recommendation.route('/check_recommend_availability', methods=['GET'])
def is_rec_availabe():
    data = request.data.decode()
    data = json.loads(data)
    availability_response = is_enough_tests_for_recommendation(data['username'])
    return json.dumps(availability_response)

