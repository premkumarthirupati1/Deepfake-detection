"""
deepfake_detection URL Configuration
Modernized for Django 5.x & Real Deep Learning Computer Vision Forensics
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from Remote_User import views as remoteuser
from Service_Provider import views as serviceprovider

urlpatterns = [
    path('admin/', admin.site.urls),

    # Remote User routes
    path('', remoteuser.login, name="login"),
    path('login/', remoteuser.login, name="user_login"),
    path('logout/', remoteuser.logout_view, name="logout"),
    path('Register1/', remoteuser.Register1, name="Register1"),
    path('video_scanner/', remoteuser.video_scanner, name="video_scanner"),
    path('Predict_deepfake_detection/', remoteuser.Predict_deepfake_detection, name="Predict_deepfake_detection"),
    path('ViewYourProfile/', remoteuser.ViewYourProfile, name="ViewYourProfile"),

    # Service Provider routes
    path('serviceproviderlogin/', serviceprovider.serviceproviderlogin, name="serviceproviderlogin"),
    path('serviceproviderlogout/', serviceprovider.serviceproviderlogout, name="serviceproviderlogout"),
    path('View_Remote_Users/', serviceprovider.View_Remote_Users, name="View_Remote_Users"),
    path('charts/<str:chart_type>/', serviceprovider.charts, name="charts"),
    path('charts1/<str:chart_type>/', serviceprovider.charts1, name="charts1"),
    path('likeschart/<str:like_chart>/', serviceprovider.likeschart, name="likeschart"),
    path('View_Deepfake_Prediction_Ratio/', serviceprovider.View_Deepfake_Prediction_Ratio, name="View_Deepfake_Prediction_Ratio"),
    path('train_model/', serviceprovider.train_model, name="train_model"),
    path('View_deepfake_prediction_Type/', serviceprovider.View_deepfake_prediction_Type, name="View_deepfake_prediction_Type"),
    path('Download_Trained_DataSets/', serviceprovider.Download_Trained_DataSets, name="Download_Trained_DataSets"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
