from flask import Blueprint, request

from .testclass import TestClass
import json

addtest_page = Blueprint('checktest_page', __name__)


@addtest_page.route('/checktest', methods=['GET'])
def checktest():
    data = request.data.decode()
    res = TestClass.check_test(data)

    el.parse_from_str(data)
    if el.error:
        return {'error': el.error}
    el.save_to_db()
    return str(el.id)
