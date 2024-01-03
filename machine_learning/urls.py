from django.contrib import admin
from django.urls import path,include
from . import views
from .views import *

urlpatterns = [
    path('',views.scan_face,name = "scan_face"),
    path('video/analyse/',views.analyse_video,name = "analyse_video"),
    path('analyzed/video/view/<int:video_id>/',views.analized_video_detail,name = "analized_video_detail"),
    path('analized/video/list/',views.analized_video_list,name = "analized_video_list"),

    path('video_detail/<int:video_id>/',views.video_detail,name = "video_detail"),
    path('delete/data/',views.delete_data,name = "analyzed_pdf_view"),
    path('frame/data/',views.frame,name = "frame"),
    path('get/data/<int:id>/',views.get_data,name = "get_data"),
]
