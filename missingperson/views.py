from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponse
from datetime import datetime
import cv2
import numpy as np
from keras.models import load_model
from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.template import loader
from django.contrib.auth.decorators import login_required

import cv2
import face_recognition

from .models import *
from .forms import CriminalRecordForm
# core/views.py

from django.core.mail import EmailMessage
from django.core.files.base import ContentFile
import cv2

def send_email_with_image(frame):
    subject = '🚨 Violence Detected - Snapshot Attached'
    body = 'Alert! Violence has been detected. Snapshot attached.'
    recipient_list = ['karthickmg2022@gmail.com']  # Replace with actual email

    # Encode frame to JPEG
    ret, buffer = cv2.imencode('.jpg', frame)
    image_data = ContentFile(buffer.tobytes(), 'snapshot.jpg')

    email = EmailMessage(subject, body, to=recipient_list)
    email.attach('snapshot.jpg', image_data.read(), 'image/jpeg')
    email.send()

violence_model = load_model('core/surveillance_model/violence_detection_model.h5')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D


# Existing feature extractor (MobileNetV2 without top layers)
feature_extractor = MobileNetV2(include_top=False, input_shape=(224,224,3), weights='imagenet')

# Reduce channels from 1280 to 512 using 1x1 conv
input_layer = Input(shape=(224, 224, 3))
x = feature_extractor(input_layer)
x = Conv2D(512, (1, 1), activation='relu')(x)  # Now shape is (7, 7, 512)
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))
frame_sequence = []  # Global list to store last 20 frames

# Create new model for feature extraction
reduced_feature_extractor = Model(inputs=input_layer, outputs=x)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

violence_model = Sequential([
    LSTM(256, input_shape=(20, 1280)),
    Dense(1, activation='sigmoid')
])

from collections import deque
import numpy as np

# A queue to store last 20 frames' features
frame_sequence = deque(maxlen=20)
def detect_violence(frame):
    global frame_sequence
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    features = feature_extractor.predict(np.expand_dims(normalized_frame, axis=0))  # Shape (1,1280)
    
    frame_sequence.append(features[0])  # Shape (1280,)
    
    if len(frame_sequence) < 20:
        return False  # Not enough frames yet
    elif len(frame_sequence) > 20:
        frame_sequence = frame_sequence[-20:]  # Keep last 20 frames
    
    input_batch = np.expand_dims(frame_sequence, axis=0)  # Shape (1, 20, 1280)
    
    prediction = violence_model.predict(input_batch)  # LSTM expects (None, 20, 1280)
    return prediction[0][0] > 0.6


import face_recognition
import cv2

def load_and_resize_image(image_path, scale=0.5):
    image = face_recognition.load_image_file(image_path)
    small_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    return small_image



def gen(camera):
    face_detected_ids = set()
    face_label_memory = {}

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Simulate detect() core logic manually for each frame (without modifying detect())
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            face_id = tuple(encoding[:5].round(2))
            label = "Unknown"
            matched = False

            # Check Missing Persons
            for person in MissingPerson.objects.all():
                try:
                    small_image = load_and_resize_image(person.image.path, scale=0.5)  # Resize to 50%
                    encodings = face_recognition.face_encodings(small_image)
                    if len(encodings) > 0:
                        stored_encoding = encodings[0]
                    else:
                         stored_encoding = None  # Handle case where no face is detected

                except IndexError:
                    continue
                if face_recognition.compare_faces([stored_encoding], encoding)[0]:
                    label = f"{person.first_name} {person.last_name} (Missing)"
                    matched = True
                    if person.id not in face_detected_ids:
                        face_detected_ids.add(person.id)
                        # Email alert
                        context = {
                            "first_name": person.first_name,
                            "last_name": person.last_name,
                            'fathers_name': person.father_name,
                            "aadhar_number": person.aadhar_number,
                            "missing_from": person.missing_from,
                            "date_time": datetime.now().strftime('%D-%M-%Y %H:%M'),
                            "location": "India"
                        }
                        html_message = render_to_string('findemail.html', context)
                        send_mail('Missing Person Found', '', 'nkaliraja14@gmail.com', [person.email], html_message=html_message)
                    break

            # Check Criminals
            if not matched:
                for criminal in CriminalRecord.objects.all():
                    try:
                        stored_encoding = face_recognition.face_encodings(face_recognition.load_image_file(criminal.image.path))[0]
                    except IndexError:
                        continue
                    if face_recognition.compare_faces([stored_encoding], encoding)[0]:
                        label = f"{criminal.alias_name} (Criminal)"
                        if criminal.id not in face_detected_ids:
                            face_detected_ids.add(criminal.id)
                            # Email alert
                            context = {
                                "alias_name": criminal.alias_name,
                                "criminal_id": criminal.criminal_id,
                                "last_known_location": criminal.last_known_location,
                                "crime_details": criminal.description,
                                "date_time": datetime.now().strftime('%D-%M-%Y %H:%M'),
                                "location": "India"
                            }
                            html_message = render_to_string('criminal_alert.html', context)
                            send_mail('Criminal Detected', '', 'nkaliraja14@gmail.com', ['karthickmg2022@gmail.com'], html_message=html_message)
                        break

            # Draw rectangles and labels
            face_label_memory[face_id] = label
            color = (0, 0, 255) if "Missing" in label else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Violence Detection per frame (optional)
        is_violence = detect_violence(frame)
        if is_violence:
            cv2.putText(frame, 'VIOLENCE DETECTED!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            send_email_with_image(frame)

        # Encode and yield frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    camera.release()
    cv2.destroyAllWindows()
def video_feed(request):
    return StreamingHttpResponse(gen(cv2.VideoCapture(0)),
                                 content_type='multipart/x-mixed-replace; boundary=frame')



# Helper for police-only views
def is_police(user):
    return user.is_staff

# ======================== Home & Static Views ========================

def home(request):
    return render(request, "index.html")

def surveillance(request):
    return render(request, "surveillance.html")

def location(request):
    return render(request, "location.html")

# ======================== Authentication ========================

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            if user.groups.filter(name='Police').exists():
                return redirect('police_dashboard')  # 👈 must match your `urls.py` name
            else:
                return redirect('home')  # Or any other default view
        else:
            return render(request, 'login.html', {'error': 'Invalid credentials'})
    return render(request, 'login.html')
# ======================== Police Views ========================

@login_required
@user_passes_test(is_police)
def delete_criminal(request, pk):
    criminal = get_object_or_404(CriminalRecord, pk=pk)
    criminal.delete()
    return redirect('police_dashboard')
@login_required
@user_passes_test(is_police)
@login_required(login_url='/login/')
@user_passes_test(is_police, login_url='/login/')
def police_dashboard(request):
    criminals = CriminalRecord.objects.all()
    return render(request, 'police_dashboard.html', {'criminals': criminals})

@login_required
@user_passes_test(is_police)
def register_criminal(request):
    if request.method == 'POST':
        form = CriminalRecordForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('police')
    else:
        form = CriminalRecordForm()
    return render(request, 'criminalsregister.html', {'form': form})

@login_required
@user_passes_test(is_police)
def update_criminal(request, pk):
    criminal = get_object_or_404(CriminalRecord, pk=pk)
    if request.method == 'POST':
        form = CriminalRecordForm(request.POST, request.FILES, instance=criminal)
        if form.is_valid():
            form.save()
            return redirect('police_dashboard')
    else:
        form = CriminalRecordForm(instance=criminal)
    return render(request, 'update_criminal.html', {'form': form})

# ======================== Criminal Registration (manual fallback) ========================

def criminalsregister(request):
    if request.method == "POST":
        data = {
            key: request.POST.get(key) for key in [
                "criminal_id", "alias_name", "crime_type", "dob", "last_known_location",
                "aadhar_number", "gender", "description"]
        }
        image = request.FILES.get("image")
        CriminalRecord.objects.create(image=image, **data)
        return redirect("home")
    return render(request, "criminalsregister.html")

# ======================== Missing Person Management ========================

def register(request):
    if request.method == 'POST':
        data = {
            'first_name': request.POST.get('first_name'),
            'last_name': request.POST.get('last_name'),
            'father_name': request.POST.get('fathers_name'),
            'date_of_birth': request.POST.get('dob'),
            'address': request.POST.get('address'),
            'phone_number': request.POST.get('phonenum'),
            'aadhar_number': request.POST.get('aadhar_number'),
            'missing_from': request.POST.get('missing_date'),
            'email': request.POST.get('email'),
            'gender': request.POST.get('gender'),
            'image': request.FILES.get('image'),
        }

        if MissingPerson.objects.filter(aadhar_number=data['aadhar_number']).exists():
            messages.info(request, 'Aadhar Number already exists')
            return redirect('/register')

        person = MissingPerson.objects.create(**data)
        person.save()

        messages.success(request, 'Case Registered Successfully')
        current_time = datetime.now().strftime('%d-%m-%Y %H:%M')
        context = data.copy()
        context["date_time"] = current_time
        subject = 'Case Registered Successfully'
        html_message = render_to_string('regmail.html', context)
        send_mail(subject, 'nkaliraja14@gmail.com', 'karthickmg2022@gmail.com', [data['email']], html_message=html_message)

    return render(request, "register.html")

def missing(request):
    query = request.GET.get('search', '')
    people = MissingPerson.objects.filter(aadhar_number__icontains=query) if query else MissingPerson.objects.all()
    return render(request, "missing.html", {'missingperson': people})

def delete_person(request, person_id):
    get_object_or_404(MissingPerson, id=person_id).delete()
    return redirect('missing')

def update_person(request, person_id):
    person = get_object_or_404(MissingPerson, id=person_id)

    if request.method == 'POST':
        fields = ['first_name', 'last_name', 'fathers_name', 'dob', 'address', 'email', 'phonenum', 'aadhar_number', 'missing_date', 'gender']
        for field in fields:
            setattr(person, field, request.POST.get(field, getattr(person, field)))

        if new_image := request.FILES.get('image'):
            person.image = new_image

        person.save()
        return redirect('missing')

    return render(request, 'edit.html', {'person': person})

# ======================== Face Recognition Surveillance ========================

def detect(request):
    video_capture = cv2.VideoCapture(0)
    face_detected_ids = set()
    face_label_memory = {}

    while True:
        ret, frame = video_capture.read()
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            face_id = tuple(encoding[:5].round(2))
            label = "Unknown"
            matched = False

            # Check Missing Persons
            for person in MissingPerson.objects.all():
                try:
                    stored_encoding = face_recognition.face_encodings(face_recognition.load_image_file(person.image.path))[0]
                except IndexError:
                    continue
                if face_recognition.compare_faces([stored_encoding], encoding)[0]:
                    label = f"{person.first_name} {person.last_name} (Missing)"
                    matched = True
                    if person.id not in face_detected_ids:
                        face_detected_ids.add(person.id)
                        context = {
                            "first_name": person.first_name,
                            "last_name": person.last_name,
                            'fathers_name': person.father_name,
                            "aadhar_number": person.aadhar_number,
                            "missing_from": person.missing_from,
                            "date_time": datetime.now().strftime('%d-%m-%Y %H:%M'),
                            "location": "India"
                        }
                        html_message = render_to_string('findemail.html', context)
                        send_mail('Missing Person Found', '', 'nkaliraja14@gmail.com', [person.email], html_message=html_message)
                    break

            # Check Criminals
            if not matched:
                for criminal in CriminalRecord.objects.all():
                    try:
                        stored_encoding = face_recognition.face_encodings(face_recognition.load_image_file(criminal.image.path))[0]
                    except IndexError:
                        continue
                    if face_recognition.compare_faces([stored_encoding], encoding)[0]:
                        label = f"{criminal.alias_name} (Criminal)"
                        if criminal.id not in face_detected_ids:
                            face_detected_ids.add(criminal.id)
                            context = {
                                "alias_name": criminal.alias_name,
                                "criminal_id": criminal.criminal_id,
                                "last_known_location": criminal.last_known_location,
                                "crime_details": criminal.description,
                                "date_time": datetime.now().strftime('%d-%m-%Y %H:%M'),
                                "location": "India"
                            }
                            html_message = render_to_string('criminal_alert.html', context)
                            send_mail('Criminal Detected', '', 'nkaliraja14@gmail.com', ['karthickmg2022@gmail.com'], html_message=html_message)
                        break

            face_label_memory[face_id] = label
            color = (0, 0, 255) if "Missing" in label else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return render(request, "surveillance.html")
