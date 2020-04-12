FROM hseling/hseling-api-base:python3.6-alpine3.7 as build

LABEL maintainer="Nikolay Babakov"

RUN mkdir /dependencies
COPY ./hseling_api_autotutor/requirements.txt /dependencies/requirements.txt

RUN pip install -r /dependencies/requirements.txt

FROM hseling/hseling-api-base:python3.6-alpine3.7 as production

COPY --from=build /usr/local/lib/python3.6/site-packages /usr/local/lib/python3.6/site-packages
COPY --from=build /usr/lib/python3.6/site-packages /usr/lib/python3.6/site-packages

COPY --from=build /dependencies /dependencies

RUN mkdir /hseling_api_autotutor

WORKDIR /hseling_api_autotutor

COPY . /hseling_api_autotutor/

EXPOSE 5000

RUN pip3 install -r requirements.txt

RUN wget https://github.com/jwijffels/udpipe.models.ud.2.0/raw/master/inst/udpipe-ud-2.0-170801/russian-syntagrus-ud-2.0-170801.udpipe -O /hseling_api_autotutor/app/recommendation_logic/processing_data/russian-syntagrus-ud-2.0-170801.udpipe

RUN wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz -O /hseling_api_autotutor/app/recommendation_logic/processing_data/cc.ru.300.vec.gz
RUN gzip -d  /hseling_api_autotutor/app/recommendation_logic/processing_data/cc.ru.300.vec.gz

CMD flask run --host='0.0.0.0'
