from django.shortcuts import render, redirect, HttpResponse

import requests
import json
from .baseviewclass import BaseViewClass


class AboutViewClass(BaseViewClass):

    def get(self, request):

        # name = [*sometext][0]  # возможна ошибка
        # context = sometext[name]
        # context['name'] = name
        return render(request, template_name='about.html')



