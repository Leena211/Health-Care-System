"""
Forms for the core application.

This file defines the forms used for user input, including symptom selection,
user signup, and user login. Django's forms provide a structured way to
handle and validate user data.
"""
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


# ... (SymptomForm remains the same) ...
class SymptomForm(forms.Form):
    """
    A form for users to input their symptoms.
    It dynamically creates choice fields for up to 4 symptoms.
    """
    # New field for text input
    symptom_text = forms.CharField(
        label='Enter a symptom',
        max_length=100,
        required=False,  # Make this optional
        widget=forms.TextInput(attrs={'placeholder': 'e.g., runny nose, headache'})
    )
    # Choice fields will be added dynamically in __init__
    def __init__(self, *args, **kwargs):
        symptoms = kwargs.pop('symptoms', [])
        super(SymptomForm, self).__init__(*args, **kwargs)
        symptom_choices = [('', 'Choose a symptom')] + [(symptom, symptom.replace('_', ' ').title()) for symptom in symptoms]

        self.fields['symptom_1'] = forms.ChoiceField(
            choices=symptom_choices,
            required=True,
            widget=forms.Select(attrs={'class': 'form-control'})
        )
        self.fields['symptom_2'] = forms.ChoiceField(
            choices=symptom_choices,
            required=True,
            widget=forms.Select(attrs={'class': 'form-control'})
        )
        self.fields['symptom_3'] = forms.ChoiceField(
            choices=symptom_choices,
            required=False,
            widget=forms.Select(attrs={'class': 'form-control'})
        )
        self.fields['symptom_4'] = forms.ChoiceField(
            choices=symptom_choices,
            required=False,
            widget=forms.Select(attrs={'class': 'form-control'})
        )

    def clean(self):
        cleaned_data = super().clean()
        symptoms = [
            cleaned_data.get('symptom_1'),
            cleaned_data.get('symptom_2'),
            cleaned_data.get('symptom_3'),
            cleaned_data.get('symptom_4'),
        ]
        # Ensure at least two symptoms are selected
        if len([s for s in symptoms if s]) < 2:
            raise forms.ValidationError("Please select at least two symptoms.")
        return cleaned_data


class SignUpForm(UserCreationForm):
    """
    A form for new user registration, inheriting from Django's UserCreationForm.
    We customize the widgets to apply CSS classes for styling.
    """
    class Meta(UserCreationForm.Meta):
        model = User
        fields = ('first_name', 'last_name', 'username', 'email', 'password1', 'password2')
        widgets = {
            'first_name': forms.TextInput(attrs={'class': 'form-input', 'placeholder': 'Enter your first name'}),
            'last_name': forms.TextInput(attrs={'class': 'form-input', 'placeholder': 'Enter your last name'}),
            'username': forms.TextInput(attrs={'class': 'form-input', 'placeholder': 'Choose a username (for login)'}),
            'email': forms.EmailInput(attrs={'class': 'form-input', 'placeholder': 'Enter your email'}),
            'password1': forms.PasswordInput(attrs={'class': 'form-input', 'placeholder': 'Enter a password'}),
            'password2': forms.PasswordInput(attrs={'class': 'form-input', 'placeholder': 'Confirm your password'}),
        }


class LoginForm(forms.Form):
    """
    A simple login form to authenticate users.
    """
    username = forms.CharField(max_length=150, widget=forms.TextInput(attrs={'class': 'form-input', 'placeholder': 'Enter your username'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-input', 'placeholder': 'Enter your password'}))

