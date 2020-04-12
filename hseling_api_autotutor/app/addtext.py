from flask import Blueprint, request

from .testclass import save_to_db
from .recommendation_logic.text_processing import get_text_map

import json

addtext_page = Blueprint('addtext_page', __name__)


@addtext_page.route('/addtext', methods=['POST'])
def addtext():
    data = request.data.decode()
    data = json.loads(data)
    # print(data, type(data))
    # user_name = data['user_name']
    # el = TestClass()
    # el = TestClass.check_test(data)
    # import pdb; pdb.set_trace()
    # el.parse_from_str(data)
    # if el.error:
    #     return {'error': el.error}
    textmap = get_text_map(data['text'], raw_text_input=True)
    print("textmap", textmap)
    added_id = save_to_db(textmap)
    return added_id
