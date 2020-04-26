import os
from flask import Flask
from app.database import db


from tqdm.auto import tqdm
import os
from .recommendation_logic.text_processing import get_text_map
import pandas as pd
from .testclass import save_to_db
from .database import redis_db


def create_app():

    BASE = os.path.dirname(os.path.abspath(__file__))
    texts_db = pd.read_csv(os.path.join(BASE, "./recommendation_logic/articles/cherdak_texts.csv"))
    uploaded_map_type = "music"
    TEXTS_TO_BE_UPLOADED = 100
    print("checking texts availability in database")
    for i, data in tqdm(texts_db.iterrows(), total=min(TEXTS_TO_BE_UPLOADED, len(texts_db))):
        potential_index_in_text_types_list = "text_maps_"+ uploaded_map_type + "_" + str(i)
        check_availability = redis_db.get(potential_index_in_text_types_list)
        if check_availability is None:
            text_map = get_text_map(data['text'], raw_text_input=True)
            save_to_db(text_map, map_type = uploaded_map_type, require_status=False)
        else:
            #print(potential_index_in_text_types_list , "has already been uploaded. SKIP")
            pass
        if i == TEXTS_TO_BE_UPLOADED: break

    app = Flask(__name__)
    # app.config.from_object(os.environ['APP_SETTINGS'])
    # db.init_app(app)
    # with app.test_request_context():
    #   db.create_all()

    # from app.addtext import addtext_page
    # from app.get_text_map import get_text
    from app.add_user_info import add_user_info
    from app.get_test import get_test
    from app.get_recommendation import send_results
    from app.get_recommendation import get_recommendation
    from app.handle_recommendation import evaluate_recommended_texts
    from app.handle_recommendation import get_user_accuracy_record

    # app.register_blueprint(addtext_page)
    # app.register_blueprint(get_text)
    app.register_blueprint(add_user_info)
    app.register_blueprint(get_test)
    app.register_blueprint(send_results)
    app.register_blueprint(get_recommendation)
    app.register_blueprint(evaluate_recommended_texts)
    app.register_blueprint(get_user_accuracy_record)
    app.run(debug=True)

    return app


"""
curl --header "Content-Type: application/json; charset=UTF-8" --request POST --data '{"type":"with choice","question":"Вопросик 1", "answers": {"a":"1", "b":"2"}, "true answer":"a"}' http://localhost:5000/addtest

curl --header "Content-Type: application/json; charset=UTF-8" --request GET --data '{ "id":"1"}' http://localhost:5000/gettest

curl --header "Content-Type: application/json; charset=UTF-8" --request POST --data '{"text":"hello ueba"}' http://localhost:5000/addtest

"""
