from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('summarizer/', include('summarizer.urls')),
    path('admin/', admin.site.urls),
]
