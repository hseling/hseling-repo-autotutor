FROM python:3.7.5

RUN mkdir /hseling_api_autotutor

WORKDIR /hseling_api_autotutor

COPY . /hseling_api_autotutor/

EXPOSE 5000

RUN pip3 install -r hseling_api_autotutor/requirements.txt

#RUN wget https://github.com/jwijffels/udpipe.models.ud.2.0/raw/master/inst/udpipe-ud-2.0-170801/russian-syntagrus-ud-2.0-170801.udpipe -O hseling_api_autotutor/app/recommendation_logic/processing_data/russian-syntagrus-ud-2.0-170801.udpipe

#RUN wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz -O hseling_api_autotutor/app/recommendation_logic/processing_data/cc.ru.300.vec.gz
#RUN gzip -d  hseling_api_autotutor/app/recommendation_logic/processing_data/cc.ru.300.vec.gz

CMD flask run --host='0.0.0.0'