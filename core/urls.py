from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.index_view, name='index'),
    path('api/doc/dashboard/', views.doctor_dashboard_data, name='doctor-dashboard-data'),
    path('api/doc/reminders/active/', views.active_reminders, name='active-reminders'),
    path(
        'api/doc/reminders/<int:reminder_id>/call-token/',
        views.reminder_call_token,
        name='reminder-call-token',
    ),
]