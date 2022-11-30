from io import BytesIO
from django.shortcuts import render
from rest_framework import status,viewsets
from rest_framework.generics import (ListCreateAPIView,DestroyAPIView,UpdateAPIView)
from django.http import JsonResponse

#import models
from .models import laThuoc
from .serializers import *

#thư viện sử lý predict
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import requests

# Create your views here.

class GetPredictedResult(ListCreateAPIView):
    resnet_model=load_model('/Users/haidang/Downloads/resnet50.model')
    class_names=['An Xoa', 'Cà Gai Leo', 'Mã Đề', 'Sam Biển', 'Dây Thìa Canh', 'Đu Đủ', 'Lá Dâu Tầm', 'Lá Ô Liu', 'Lá Sen', 'Ngải Tía', 'Nghệ Xanh', 'Ngô', 'Trái Mấm', 'Xạ Đen']
    def get(self,request):
        url = self.request.query_params.get('url')
        #img = Image.open(BytesIO(requests.get('/Users/haidang/Downloads/dudu.jpg').content)) 
        img=cv2.imread(url)                                
        image=cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        image_resized = cv2.resize(image,(224,224))
        image=np.expand_dims(image_resized,axis=0)
        pred = self.resnet_model.predict(image)
        accuracy = float("{:.2f}".format(max(pred[0]))) * 100
        accuracy = int(accuracy)
        return JsonResponse({
                'Loai': ''+self.class_names[np.argmax(pred)],
                'DoChinhXac': accuracy,
            }, status=status.HTTP_201_CREATED,safe=False, json_dumps_params={'ensure_ascii': False})

class LaCayViewSet(viewsets.ModelViewSet,DestroyAPIView,UpdateAPIView):
    queryset=laThuoc.objects.all()
    serializer_class =LaCaySerializer

    def destroy(self, request, *args, **kwargs):
        return super().destroy(request, *args, **kwargs)

class BenhGanViewSet(viewsets.ModelViewSet):
    queryset=benhGan.objects.all()
    serializer_class =BenhGanSerializer