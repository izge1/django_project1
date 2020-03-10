from django.conf.urls import url
from chatbot import views
 
urlpatterns = [
    url(r'^$', views.index, name='chatbot_index'),
    url(r'^message/$', views.message, name='message'),
    url(r'^message_list/$', views.message_list, name='message_list'),
    #url(r'^chatbot/$', views.test_redirect, name='test_redirect'),
]