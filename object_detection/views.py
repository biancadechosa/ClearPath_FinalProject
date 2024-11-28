from django.shortcuts import render

def index(request):
    return render(request, 'object_detection/index.html')
