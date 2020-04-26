from flask_sqlalchemy import SQLAlchemy

import redis

db = SQLAlchemy()

redis_db = redis.Redis(host='redis', port=6379)#на облаке
# redis_db = redis.Redis()#локально

# sudo lsof -i:5000
# kill PID