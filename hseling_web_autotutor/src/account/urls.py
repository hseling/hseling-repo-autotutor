"""deduplication URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from django.contrib.auth import views
from .views.registration import RegisterFormView
from .views.success_registration import SuccessRegistrationClass
from .views.about import AboutRegistrationClass


urlpatterns = (
    path(r'login/', views.LoginView.as_view(redirect_authenticated_user=True), name='login'),
    path(r'logout/', views.LogoutView.as_view(), name='logout'),
    path(r'register/', RegisterFormView.as_view(), name='register'),
    path(r'success/', SuccessRegistrationClass.as_view(), name='success_registration'),
    path(r'about/', AboutRegistrationClass.as_view(), name='about')

)
