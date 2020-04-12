from django.views.generic.base import View
from django.shortcuts import render

from django.contrib.auth.mixins import LoginRequiredMixin


class SuccessRegistrationClass(View):
    def get(self, request):

        return render(request, template_name='registration/success_registration.html')
