from django.db import models

# Create your models here.

class dangNhap(models.Model):
    maDangNhap=models.CharField(null=False,max_length=255)
    matKhau=models.CharField(null=False,max_length=255)