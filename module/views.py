from django.shortcuts import render
from rest_framework import status
from django.http import JsonResponse
from rest_framework.generics import ListCreateAPIView, RetrieveUpdateDestroyAPIView

#thư viện sử lý predict
from keras.models import load_model
import numpy as np
import cv2

# Create your views here.

class GetPredictedResult(ListCreateAPIView):
    resnet_model=load_model('/Users/haidang/Downloads/resnet50.model')
    class_names=['AnXoa', 'CaGaiLeo', 'CayMaDe', 'CaySamBien', 'DayThiaCanh', 'DuDu', 'LaDauTam', 'LaOLiu', 'LaSen', 'NgayTia', 'NgheXanh', 'Ngo', 'TraiMam', 'XaDen']
    def get(self,request):
        url = 'https://duockienminh.vn/sites/default/files/anh_bai_viet/1-la-du-du-la-gi-tai-sao-nhieu-ngu.jpg'
        # req = urllib.request.urlopen(url)
        # arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        # image = cv2.imdecode(arr,1)
        image=cv2.imread('/Users/haidang/Documents/Zalo Received Files/DuLieu/phanloaila/3_XaDen/1-2874.jpeg')
        image_resized = cv2.resize(image,(224,224))
        image=np.expand_dims(image_resized,axis=0)
        pred = self.resnet_model.predict(image)
        accuracy = float("{:.2f}".format(max(pred[0]))) * 100
        accuracy = int(accuracy)
        return JsonResponse({
                'Loai': ''+self.class_names[np.argmax(pred)],
            }, status=status.HTTP_201_CREATED)