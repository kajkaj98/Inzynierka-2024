from django.contrib import admin
from django.urls import path
from style_classification import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='homepage'),
    path('predictImage', views.predictImage, name='predictImage'),
    path('methods', views.methods, name='methods'),
    path('about', views.about, name='about'),
    path('styles', views.styles, name='styles'),
    path('byzantin_iconography', views.style_various, {'style_type': 'Ikona_bizantyjska'}, name='byzantin_iconography'),
    path('early_renaissance', views.style_various, {'style_type': 'Renesans'}, name='early_renaissance'),
    path('northern_renaissance', views.style_various, {'style_type': 'Neorenesans'}, name='northern_renaissance'),
    path('high_renaissance', views.style_various, {'style_type': 'Wysoki_renesans'}, name='high_renaissance'),
    path('baroque', views.style_various, {'style_type': 'Barok'}, name='baroque'),
    path('rococo', views.style_various, {'style_type': 'Rokoko'}, name='rococo'),
    path('romantism', views.style_various, {'style_type': 'Romantyzm'}, name='romantism'),
    path('realism', views.style_various, {'style_type': 'Realizm'}, name='realism'),
    path('impressionism', views.style_various, {'style_type': 'Impresionizm'}, name='impressionism'),
    path('post_impressionism', views.style_various, {'style_type': 'Postimpresionizm'}, name='post_impressionism'),
    path('expressionism', views.style_various, {'style_type': 'Ekspresionizm'}, name='expressionism'),
    path('symbolism', views.style_various, {'style_type': 'Symbolizm'}, name='symbolism'),
    path('fauvism', views.style_various, {'style_type': 'Fowizm'}, name='fauvism'),
    path('cubism', views.style_various, {'style_type': 'Kubizm'}, name='cubism'),
    path('surrealism', views.style_various, {'style_type': 'Surrealizm'}, name='surrealism'),
    path('abstract_art', views.style_various, {'style_type': 'Abstrakcja'}, name='abstract_art'),
    path('naive_art', views.style_various, {'style_type': 'Prymitywizm'}, name='naive_art'),
    path('pop_art', views.style_various, {'style_type': 'Pop_Art'}, name='pop_art'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
