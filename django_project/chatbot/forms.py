from django import forms
#from django.core.exceptions import ValidationError
from .models import Message
from django.template.defaultfilters import slugify
 
class MessageForm(forms.ModelForm):
 
    class Meta:
        model = Message
        fields = '__all__'