from django.shortcuts import render
from django.http import HttpResponse
from django.template import Context, loader
from django.views.defaults import page_not_found, permission_denied, bad_request, server_error


def error_404(request, exception, template_name="errors/404.html"):
    return page_not_found(request, exception, template_name)


def error_500(request, template_name="errors/500.html"):
    return server_error(request, template_name)


def error_400(request, exception, template_name="errors/400.html"):
    return bad_request(request, exception, template_name)


def error_403(request, exception, template_name="errors/403.html"):
    return permission_denied(request, exception, template_name)
