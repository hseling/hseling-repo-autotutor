from flask import Blueprint, request
import json
from .database import redis_db

add_user_info = Blueprint('add_user_info', __name__)


@add_user_info.route('/adduser', methods=['POST'])
def adduser():
    data = request.data.decode()
    data = json.loads(data)
    username_in_db = "username_" + data['username']
    is_in_db = redis_db.get(username_in_db)
    if is_in_db is None:
        redis_db[username_in_db] = 1
        redis_db[username_in_db + "_handled_texts"] = ''
        return {"status": "username created"}
    else:
        return {"error": "username already taken"}
