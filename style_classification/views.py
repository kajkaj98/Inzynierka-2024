from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from style_classification.pre_processing import execute_model_p1, execute_model_p2, execute_model_p3
import os, re
from PIL import Image
from io import BytesIO
import random


def index(request):
    return render(request, 'index.html', context={'a': 1})

def methods(request):
    return render(request, 'methods.html', context={'a': 1})

def styles(request):
    return render(request, 'styles.html', context={'a': 1})

def about(request):
    return render(request, 'about.html', context={'a': 1})

labels = {0: 'Ikona_bizantyjska',
          1: 'Renesans',
          2: 'Neorenesans',
          3: 'Wysoki_renesans',
          4: 'Barok',
          5: 'Rokoko',
          6: 'Romantyzm',
          7: 'Realizm',
          8: 'Impresjonizm',
          9: 'Postimpresjonizm',
          10: 'Ekspresjonizm',
          11: 'Symbolizm',
          12: 'Fowizm',
          13: 'Kubizm',
          14: 'Surrealizm',
          15: 'Abstrakcja',
          16: 'Prymitywizm',
          17: 'Pop_Art'}

map = {'Ikona_bizantyjska': 'byzantin_iconography',
        'Renesans': 'early_renaissance',
        'Neorenesans': 'northern_renaissance',
        'Wysoki_renesans': 'high_renaissance',
        'Barok': 'baroque',
        'Rokoko': 'rococo',
        'Romantyzm': 'romantism',
        'Realizm': 'realism',
        'Impresjonizm': 'impressionism',
        'Postimpresjonizm': 'post_impressionism',
        'Ekspresjonizm': 'expressionism',
        'Symbolizm': 'symbolism',
        'Fowizm': 'fauvism',
        'Kubizm': 'cubism',
        'Surrealizm': 'surrealism',
        'Abstrakcja': 'abstract_art',
        'Prymitywizm': 'naive_art',
        'Pop_Art': 'pop_art'}

def predictImage(request):

    fileObj = request.FILES['filePath']
    with Image.open(BytesIO(fileObj.read())) as img:
        image_size = img.size
    print(image_size)
    selectedModel = request.POST.get('models', '')
    if request.POST.get('fileName', ''):
        filename = request.POST.get('fileName', '')
    else:
        filename_tmp = fileObj.name
        filename, extension = os.path.splitext(filename_tmp)
    filename = filename.replace(' ','_').replace('/','_').replace('\"','_')
    fs = FileSystemStorage()
    filePathName = fs.save(filename, fileObj)
    filePathName = fs.url(filePathName)
    testimage = '.' + filePathName

    if selectedModel == 'model1':
        modelPath = './models/pt_p1.pth'
        predict, avg_pred, max_prob = execute_model_p1(modelPath, testimage)
        prob = f'z prawdopodobieństwem: {round(max_prob*100, 2)}%'
    elif selectedModel == 'model2':
        modelPath = './models/pt_p2.pth'
        predict, avg_pred, max_prob = execute_model_p2(modelPath, testimage)
        prob = f'z prawdopodobieństwem: {round(max_prob*100, 2)}%'
    elif selectedModel == 'model3':
        modelPath = './models/p3_final.pth'
        modelPath2 = './models/random_forest_final.joblib'
        predict = execute_model_p3(modelPath, modelPath2, testimage)
        prob = ''

    predictedLabel = labels[predict]

    new_file_path = './media/new/' + filename + '_' + selectedModel + '_' + predictedLabel
    while os.path.exists(new_file_path):
        base_name, extension = os.path.splitext(new_file_path)
        new_file_path = f"{base_name}_{random.randint(1000, 9999)}{extension}"
    os.rename(testimage, new_file_path)

    predictedLabel1 = predictedLabel.replace('_', ' ')

    context = {'filePathName': new_file_path, 'filename': filename, 'predictedLabel': predictedLabel1, 'prob': prob, 'link': map[predictedLabel]}
    return render(request, 'index.html', context=context)


def style_various(request, style_type):
    files = [f for f in os.listdir("./media/new/") if f.endswith(style_type)]
    urls = [os.path.join("./media/new/", f) for f in files]
    titles = [os.path.basename(f) for f in files]
    titles_new = [title[:re.search('model', title).start() - 1] for title in titles]
    files_data = sorted(zip(urls, titles_new), key=lambda x: x[1])
    return render(request, f'styles/{map[style_type]}.html', context={'files': files_data})