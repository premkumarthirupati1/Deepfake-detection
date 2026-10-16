from django.db import models

class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=50, unique=True)
    email = models.EmailField(max_length=100)
    password = models.CharField(max_length=128)
    phoneno = models.CharField(max_length=25)
    country = models.CharField(max_length=60)
    state = models.CharField(max_length=60)
    city = models.CharField(max_length=60)
    address = models.CharField(max_length=1000, blank=True, default='')
    gender = models.CharField(max_length=30, blank=True, default='')

    def __str__(self):
        return self.username

    class Meta:
        verbose_name = "Client Registration"
        verbose_name_plural = "Client Registrations"


class deepfake_detection_type(models.Model):
    url = models.CharField(max_length=3000)
    ipaddress = models.CharField(max_length=100)
    videoname = models.CharField(max_length=255)
    original_width = models.CharField(max_length=50)
    original_height = models.CharField(max_length=50)
    country = models.CharField(max_length=100)
    locale = models.CharField(max_length=100)
    latitude = models.CharField(max_length=50)
    longitude = models.CharField(max_length=50)
    Prediction = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.videoname} -> {self.Prediction}"

    class Meta:
        verbose_name = "Deepfake Detection Record"
        verbose_name_plural = "Deepfake Detection Records"


class detection_accuracy(models.Model):
    names = models.CharField(max_length=100)
    ratio = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.names}: {self.ratio}%"

    class Meta:
        verbose_name = "Detection Accuracy"
        verbose_name_plural = "Detection Accuracies"


class detection_ratio(models.Model):
    names = models.CharField(max_length=100)
    ratio = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.names}: {self.ratio}%"

    class Meta:
        verbose_name = "Detection Ratio"
        verbose_name_plural = "Detection Ratios"
