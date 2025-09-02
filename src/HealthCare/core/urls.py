"""
URL configuration for the core app.

This file maps the app-specific URLs to their corresponding views.
"""
from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static


# Define the URL patterns for the core app
urlpatterns = [
    # Home page URL
    path('', views.home, name='home'),
    # Login page URL
    path('login/', views.login_view, name='login'),
    # Logout functionality URL
    path('logout/', views.logout_view, name='logout'),
    # User profile page URL
    path('profile/', views.profile, name='profile'),
    # Signup page URL
    path('signup/', views.signup_view, name='signup'),
    #path for the results page
    path('results/', views.results_view, name='results'),
    
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
