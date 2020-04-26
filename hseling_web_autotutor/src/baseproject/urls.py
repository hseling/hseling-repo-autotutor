from django.contrib import admin
from django.urls import path, include
from django.shortcuts import render, redirect
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

handler404 = 'baseproject.views.error_404'
handler500 = 'baseproject.views.error_500'
handler403 = 'baseproject.views.error_403'
handler400 = 'baseproject.views.error_400'


def base(r):
    return render(r, 'base.html')


def landing(request):
    if request.user.is_authenticated:
        return redirect('/tutor/profile/')
    return render(request,
                  'landing.html')


urlpatterns = [
    path('admin/', admin.site.urls),
    path('account/', include('account.urls')),
    path('tutor/', include('core.urls')),
    path('', landing),
]

urlpatterns += staticfiles_urlpatterns()
