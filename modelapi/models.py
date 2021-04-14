from django.db import models

# Create your models here.
class Question(models.Model):
    query = models.TextField()
    date =  models.DateTimeField(auto_now_add = True)
    count = models.PositiveIntegerField(default = 1)
    answered =models.BooleanField(default = False)
    answer = models.TextField(default = '')
    def __str__(self):
        return self.query
    def counter(self):
        return self.count
