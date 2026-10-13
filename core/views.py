from django.http import HttpResponse

def index_view(request):
    return HttpResponse("Voice Scheduling Agent Backend is running successfully.")