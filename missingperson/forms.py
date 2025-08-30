from django import forms
from .models import CriminalRecord
from .models import MissingPerson

class MissingPersonForm(forms.ModelForm):
    class Meta:
        model = MissingPerson
        fields = '__all__'

class CriminalRecordForm(forms.ModelForm):
    class Meta:
        model = CriminalRecord
        fields = '__all__'
