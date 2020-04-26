import json

from flask import Blueprint, request

from .testclass import get_text_by_id

get_text = Blueprint('get_text', __name__)


@get_text.route('/gettext', methods=['GET'])
def gettest():
    data = request.data.decode()

    data = json.loads(data)
    return get_text_by_id(id=data['id'])


"""curl --header "Content-Type: application/json; charset=UTF-8" --request GET --data '{ "id":"1"}' http://localhost:5000/gettest"""

"""curl --header "Content-Type: application/json; charset=UTF-8" --request GET --data '{ "id":"1111111"}' http://localhost:5000/gettest"""
