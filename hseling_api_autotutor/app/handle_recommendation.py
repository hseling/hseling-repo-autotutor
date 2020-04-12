from flask import Blueprint, request
import json
from .recommendation_logic.evaluate_algo_results import record_recommendation_accuracy, get_accuracy_rec
import json


evaluate_recommended_texts = Blueprint('evaluate_recommended_texts', __name__)
get_user_accuracy_record = Blueprint('get_user_accuracy_record', __name__)


@evaluate_recommended_texts.route('/evaluate_recommended_texts', methods=['POST'])
def eval():
    data = request.data.decode()
    data = json.loads(data)
    test_sentences_json = record_recommendation_accuracy(data)#data['answers']
    return json.dumps(test_sentences_json, ensure_ascii=False)


@get_user_accuracy_record.route('/get_user_accuracy_record', methods=['GET'])
def get_acc_rec():
    data = request.data.decode()
    data = json.loads(data)
    accuracy_record = get_accuracy_rec(data)#data['user_id']

    return json.dumps(accuracy_record)