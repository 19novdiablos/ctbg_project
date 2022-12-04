from rest_framework import serializers
from .models import *

class LaCaySerializer(serializers.ModelSerializer):
    class Meta:
        model=laThuoc
        fields=('maLa','tenLa','tenKhac','tenKhoaHoc','moTa')

class BenhGanSerializer(serializers.ModelSerializer):
    class Meta:
        model=benhGan
        fields=('maBenh','timHieuChung','nguyenNhan','nguyCo','dieuTri','cheDoSinhHoat')

class ClipboardSerializer(serializers.ModelSerializer):
    class Meta:
        model=clipboard
        fields=('file',)