from django.views.generic.base import View

from django.contrib.auth.mixins import LoginRequiredMixin

from baseproject.settings import API_URL


class BaseViewClass(LoginRequiredMixin, View):
    login_url = '/account/login/'
    logout_url = '/'
    api_url = API_URL
    headers = {'Content-Type': 'application/json; charset=UTF-8'}
