import requests
import json

from django.shortcuts import render, redirect, HttpResponse

from .baseviewclass import BaseViewClass

from baseproject.views import error_500


class TaskViewClass(BaseViewClass):

    def get(self, request):
        data = {"username": request.user.username}
        try:
            response = requests.get('http://' + self.api_url + '/gettest', headers=self.headers, data=json.dumps(data))

            status_code = response.status_code
        except:
            return error_500(request)

        if status_code != 200:
            return error_500(request)

        sometext = json.loads(response.text)

        if 'error' in sometext:
            return error_500(request)

        if len(sometext) == 0:
            return HttpResponse('Тесты закончились')

        try:
            response = requests.get('http://' + self.api_url + '/check_recommend_availability',
                                    headers=self.headers, data=json.dumps(data))
            go_to_text = response.text
        except:
            pass

        name = [*sometext][0]  # возможна ошибка
        context = sometext[name]
        context['name'] = name
        context['go_to_text'] = go_to_text == '"can_provide_recommendations"'

        return render(request, template_name='tasks.html', context=context)

    def post(self, request):
        unsv = dict(request.POST)
        unsv = {i: j[0] for i, j in unsv.items()}

        data = {'username': request.user.username,
                'answers': {unsv['qname']: {int(i): 'Да' == j for i, j in unsv.items() if
                                            i not in ('qname', 'csrfmiddlewaretoken', '_answer')}}}

        try:
            response = requests.post('http://' + self.api_url + '/send_test_results', headers=self.headers,
                                     data=json.dumps(data))
        except:
            return error_500(request)

        if response.status_code == 200:
            user = request.user
            user.profile.unsv_counter += 1
            user.save()
            return redirect("/tutor/tasks/")
        return HttpResponse('Коля, поднимай свой сервер')
