from django.shortcuts import render, redirect, HttpResponse

import requests
import json
from .baseviewclass import BaseViewClass


class ProfileEditViewClass(BaseViewClass):
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
                return HttpResponse('Коля, поднимай свой сервер')

        context = {'username': user.username, 'last_name': user.last_name, 'first_name': user.first_name,
                   'email': user.email}
        return render(request, template_name='profile_edit.html', context=context)

    def post(self, request):
        user = request.user
        u_details = dict(request.POST)
        user.first_name = u_details['first_name'][0]
        user.last_name = u_details['last_name'][0]
        user.email = u_details['email'][0]
        # import pdb; pdb.set_trace()
        # return HttpResponse('post       ')
        user.save()
        return redirect('/tutor/profile/')
