from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.core.mail import send_mail, EmailMessage
from django.template.loader import render_to_string
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import StreamingHttpResponse
from datetime import datetime
import cv2
import numpy as np
import face_recognition
from django.core.files.base import ContentFile

from .models import *
from .forms import CriminalRecordForm

# ======================== YOLOv8 Violence Detection ========================
from ultralytics import YOLO
yolo_model = YOLO("../core/surveillance_model/yolo_small_weights.pt")  # YOLOv8-small weights

def send_email_with_image(frame):
    subject = '🚨 Violence Detected - Snapshot Attached'
    body = 'Alert! Violence has been detected. Snapshot attached.'
    recipient_list = ['karthickmg2022@gmail.com']

    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return
    image_data = ContentFile(buffer.tobytes(), 'snapshot.jpg')

    email = EmailMessage(subject, body, to=recipient_list)
    email.attach('snapshot.jpg', image_data.read(), 'image/jpeg')
    email.send()

# ======================== Streaming with Face & Violence Detection ========================
def gen(camera):
    face_detected_ids = set()
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # ----------- YOLOv8 Violence Detection -----------
        results = yolo_model.predict(frame, imgsz=640, conf=0.5, device='cpu')
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy()
            for (x1, y1, x2, y2), cls in zip(boxes, class_ids):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if int(cls) == 1:  # fight class
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "VIOLENCE DETECTED!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    send_email_with_image(frame)

        # ----------- Face Recognition -----------
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            face_id = tuple(encoding[:5].round(2))
            label = "Unknown"
            matched = False

            # Missing Persons
            for person in MissingPerson.objects.all():
                try:
                    stored_encoding = face_recognition.face_encodings(
                        face_recognition.load_image_file(person.image.path)
                    )[0]
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
                            "fathers_name": person.father_name,
                            "aadhar_number": person.aadhar_number,
                            "missing_from": person.missing_from,
                            "date_time": datetime.now().strftime('%d-%m-%Y %H:%M'),
                            "location": "India"
                        }
                        html_message = render_to_string('findemail.html', context)
                        send_mail('Missing Person Found', '', 'nkaliraja14@gmail.com', [person.email], html_message=html_message)
                    break

            # Criminals
            if not matched:
                for criminal in CriminalRecord.objects.all():
                    try:
                        stored_encoding = face_recognition.face_encodings(
                            face_recognition.load_image_file(criminal.image.path)
                        )[0]
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

            color = (0, 255, 0) if "Missing" in label else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # ----------- Encode Frame for Streaming -----------
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen(cv2.VideoCapture(0)),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


# ======================== Helpers & Authentication ========================
def is_police(user):
    return user.is_staff

def home(request):
    return render(request, "index.html")

def surveillance(request):
    return render(request, "surveillance.html")

def location(request):
    return render(request, "location.html")

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            if user.groups.filter(name='Police').exists():
                return redirect('police_dashboard')
            else:
                return redirect('home')
        else:
            return render(request, 'login.html', {'error': 'Invalid credentials'})
    return render(request, 'login.html')


# ======================== Police Views ========================
@login_required
@user_passes_test(is_police)
def police_dashboard(request):
    criminals = CriminalRecord.objects.all()
    return render(request, 'police_dashboard.html', {'criminals': criminals})

@login_required
@user_passes_test(is_police)
def criminalsregister(request):
    if request.method == 'POST':
        form = CriminalRecordForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('police_dashboard')
    else:
        form = CriminalRecordForm()
    return render(request, "criminalsregister.html")
    # return render(request, 'criminalsregister.html', {'form': form})

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

@login_required
@user_passes_test(is_police)
def delete_criminal(request, pk):
    criminal = get_object_or_404(CriminalRecord, pk=pk)
    criminal.delete()
    return redirect('police_dashboard')


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
        html_message = render_to_string('regmail.html', context)
        send_mail('Case Registered Successfully', '', 'karthickmg2022@gmail.com', [data['email']], html_message=html_message)

    return render(request, "register.html")


def missing(request):
    query = request.GET.get('search', '')
    people = MissingPerson.objects.filter(aadhar_number__icontains=query) if query else MissingPerson.objects.all()
    return render(request, "missing.html", {'missingperson': people})


def update_person(request, person_id):
    person = get_object_or_404(MissingPerson, id=person_id)
    if request.method == 'POST':
        fields = ['first_name', 'last_name', 'father_name', 'date_of_birth', 'address', 'phone_number', 'aadhar_number', 'missing_from', 'gender']
        for field in fields:
            setattr(person, field, request.POST.get(field, getattr(person, field)))
        if new_image := request.FILES.get('image'):
            person.image = new_image
        person.save()
        return redirect('missing')
    return render(request, 'edit.html', {'person': person})


def delete_person(request, person_id):
    get_object_or_404(MissingPerson, id=person_id).delete()
    return redirect('missing')

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
            color = (0, 255, 0) if "Missing" in label else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return render(request, "surveillance.html")

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
            color = (0, 255, 0) if "Missing" in label else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return render(request, "surveillance.html")
