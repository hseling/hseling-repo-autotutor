FROM hseling/hseling-api-base:python3.6-alpine3.7 as build

LABEL maintainer="Natalia Isupova <info@linghub.ru>"

RUN mkdir /dependencies
COPY ./hseling_api_autotutor/requirements.txt /dependencies/requirements.txt
COPY ./hseling_api_autotutor/setup.py /dependencies/setup.py

RUN pip install -r /dependencies/requirements.txt

FROM hseling/hseling-api-base:python3.6-alpine3.7 as production

COPY --from=build /usr/local/lib/python3.6/site-packages /usr/local/lib/python3.6/site-packages
COPY --from=build /usr/lib/python3.6/site-packages /usr/lib/python3.6/site-packages

COPY --from=build /dependencies /dependencies

COPY ./hseling_lib_autotutor /dependencies/hseling_lib_autotutor
RUN pip install /dependencies/hseling_lib_autotutor

COPY ./hseling_api_autotutor/hseling_api_autotutor /app/hseling_api_autotutor
COPY ./hseling_api_autotutor/main.py /app/main.py

ENV HSELING_API_AUTOTUTOR_SETTINGS="/app/hseling_api_autotutor/default_settings.py"

HEALTHCHECK CMD curl --fail http://localhost/healthz || exit 1
