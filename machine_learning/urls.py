from django.contrib import admin
from django.urls import path,include
from . import views
from .views import *

urlpatterns = [
    path('',views.video_analysis,name = "video_analysis"),
    path('scan/live/face/',views.scan_live_face,name = "scan_live_face"),
    path('analyzed/video/view/<int:video_id>/',views.analized_video_detail,name = "analized_video_detail"),
    path('analized/video/list/',views.analized_video_list,name = "analized_video_list"),
    
    path('live/analysis/', LiveAnalysisView.as_view()),
    path('analysed/video/list/', AnalysedVideoListView.as_view()),
]