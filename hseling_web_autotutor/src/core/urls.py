from django.urls import path

from core.views.tasksviewclass import TaskViewClass
from core.views.readviewclass import ReadViewClass
from core.views.profileviewclass import ProfileViewClass
from core.views.profileeditviewclass import ProfileEditViewClass
from core.views.aboutviewclass import AboutViewClass

urlpatterns = [
    # path('test/', BaseViewClass.as_view()),
    path('tasks/', TaskViewClass.as_view()),
    path('read/', ReadViewClass.as_view()),
    path('profile/', ProfileViewClass.as_view()),
    #path('profile_edit/', ProfileEditViewClass.as_view()),
    path('about/', AboutViewClass.as_view())
]
