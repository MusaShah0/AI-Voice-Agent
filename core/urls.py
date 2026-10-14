from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.index_view, name='index'),
    path('api/doc/dashboard/', views.doctor_dashboard_data, name='doctor-dashboard-data'),
]