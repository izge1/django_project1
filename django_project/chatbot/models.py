from django.db import models
#from django.template.defaultfilters import slugify

# Create your models here.
class Message(models.Model):
    messageContent = models.TextField()
    messageDate = models.DateTimeField(auto_now_add=True)
 
    class Meta:
        verbose_name_plural = "Message"
 
    def __str__(self):
        return self.name + "-" +  self.email
