from django.shortcuts import render, redirect, HttpResponse

import requests
import json
from collections import OrderedDict

# не знаю почему, но следующую строчку удалять нельзя. Перестает работать
from ..models.profile import Profile

from .baseviewclass import BaseViewClass


class ProfileViewClass(BaseViewClass):
    def get(self, request):

        user = request.user
        if not user.profile.is_in_api:
            data = {"username": user.username}
            try:
                response = requests.post('http://' + self.api_url + '/adduser', headers=self.headers,
                                         data=json.dumps(data))

                assert (response.status_code, 200)

                user.profile.is_in_api = True
                user.save()
            except:
                pass
                # return HttpResponse('Коля, поднимай свой сервер')
        data = {"username": user.username}
        try:
            response = requests.get('http://' + self.api_url + '/get_user_accuracy_record', headers=self.headers,
                                     data=json.dumps(data))

            assert (response.status_code, 200)

        except:
            return HttpResponse('Коля, поднимай свой сервер')

        try:
            dataSource = OrderedDict()
            dataSource["data"] = []
            accuracy = response.text
            if accuracy is not None and len(accuracy) > 0:
                accuracy = json.loads(accuracy)
            accuracy = [ round(float(i), 2) for i in accuracy]
            for i in range(len(accuracy)):
                dataSource["data"].append({"label": '{}'.format(i+1), "value": '{}'.format(accuracy[i]),
                                           "tooltext": "Рекомендация {}, Точность: {}".format(i+1, accuracy[i])})

            chartConfig = OrderedDict()
            chartConfig["caption"] = "Как хорошо наш алгоритм рекомендует тексты"
            chartConfig["subCaption"] = "Чем ближе к 1, тем тексты больше подходят"
            chartConfig["xAxisName"] = "Номер рекомендации"
            chartConfig["yAxisName"] = "Качество рекомендации"
            #chartConfig["numberSuffix"] = "K"
            chartConfig["theme"] = "gammel"

            dataSource["chart"] = chartConfig
            column2D = FusionCharts("column2d", "myFirstChart", "600", "400", "myFirstchart-container", "json",
                                    dataSource)

        except:
            accuracy = []
            column2D = None

        if column2D is not None:
            context = {'username': user.username,
                       'last_name': user.last_name,
                       'first_name': user.first_name,
                       'email': user.email,
                       'is_in_api': user.profile.is_in_api,
                       'accuracy': accuracy,
                       'output': column2D.render()}
        else:
            context = {'username': user.username,
                       'last_name': user.last_name,
                       'first_name': user.first_name,
                       'email': user.email,
                       'is_in_api': user.profile.is_in_api}

        return render(request, template_name='profile.html', context=context)

    def post(self, request):
        return redirect('/tutor/profile_edit/')
