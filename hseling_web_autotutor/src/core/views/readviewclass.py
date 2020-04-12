from django.shortcuts import render, redirect, HttpResponse

import requests
import json

from .baseviewclass import BaseViewClass
from baseproject.views import error_500


class ReadViewClass(BaseViewClass):

    def get(self, request):
        if request.user.profile.unsv_counter < 3:
            return redirect('/tutor/tasks/')
        data = {"username": request.user.username}
        try:
            response = requests.get('http://' + self.api_url + '/get_recommendation',
                                    headers=self.headers, data=json.dumps(data))
        except:
            return error_500(request)

        status_code = response.status_code
        if status_code != 200:
            return error_500(request)
        #
        sometext = json.loads(response.text)
        if 'error' in sometext or 'erorr' in sometext:
            return HttpResponse('Необходимо пройти ещё несколько тестов')

        context = []
        for i, j in enumerate(sometext):
            context.append((i, j['raw_text']))

        return render(request, template_name='text.html', context={'data': context})

    def post(self, request):

        unsv = dict(request.POST)
        result = []
        for i in range(10):
            a = {"recommended": {
                "marked_easier": False,
            },
                "non_recommended": {
                    "marked_easier": False,
                }
            }
            if str(i) in unsv:
                if 'Да' in unsv[str(i)]:
                    a['recommended']['marked_easier'] = True
                else:
                    a['non_recommended']['marked_easier'] = True
                result.append(a)
            else:
                break
        result = {'username': request.user.username,
                  'student_evaluation_json':result}
        try:
            response = requests.post('http://' + self.api_url + '/evaluate_recommended_texts', headers=self.headers,
                                     data=json.dumps(result))
        except:
            return HttpResponse('Коля, поднимай свой сервер')
        return redirect("/tutor/tasks/")
