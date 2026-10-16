import os
from pathlib import Path
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.conf import settings
from Remote_User.models import ClientRegister_Model, deepfake_detection_type, detection_ratio, detection_accuracy
from deepfake_detection.ml_engine import predict_single
from deepfake_detection.deep_learning.forensic_analyzer import analyze_image_file, analyze_video_file


def login(request):
    """Handles Remote User Authentication."""
    if request.method == "POST":
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()

        try:
            user = ClientRegister_Model.objects.get(username=username, password=password)
            request.session["userid"] = user.id
            request.session["username"] = user.username
            messages.success(request, f"Welcome back, {user.username}!")
            return redirect('video_scanner')
        except ClientRegister_Model.DoesNotExist:
            messages.error(request, "Invalid username or password. Please try again.")

    return render(request, 'RUser/login.html')


def logout_view(request):
    """Logs out the active user and clears the session."""
    request.session.flush()
    messages.info(request, "You have been logged out.")
    return redirect('login')


def Register1(request):
    """Handles new client registration."""
    if request.method == "POST":
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '').strip()
        phoneno = request.POST.get('phoneno', '').strip()
        country = request.POST.get('country', '').strip()
        state = request.POST.get('state', '').strip()
        city = request.POST.get('city', '').strip()
        address = request.POST.get('address', '').strip()
        gender = request.POST.get('gender', '').strip()

        if not username or not password:
            messages.error(request, "Username and password are required.")
            return render(request, 'RUser/Register1.html')

        if ClientRegister_Model.objects.filter(username=username).exists():
            messages.error(request, f"Username '{username}' is already taken.")
            return render(request, 'RUser/Register1.html')

        ClientRegister_Model.objects.create(
            username=username,
            email=email,
            password=password,
            phoneno=phoneno,
            country=country,
            state=state,
            city=city,
            address=address,
            gender=gender
        )
        messages.success(request, "Registration successful! You can now log in.")
        return redirect('login')

    return render(request, 'RUser/Register1.html')


def ViewYourProfile(request):
    """Displays the currently logged in user's profile."""
    userid = request.session.get('userid')
    if not userid:
        messages.warning(request, "Please log in to view your profile.")
        return redirect('login')

    user = get_object_or_404(ClientRegister_Model, id=userid)
    return render(request, 'RUser/ViewYourProfile.html', {'object': user})


def Predict_deepfake_detection(request):
    """
    Sub-millisecond inference view powered by persistent ML Voting Classifier.
    """
    context = {}

    if request.method == "POST":
        url = request.POST.get('url', '').strip()
        ipaddress = request.POST.get('ipaddress', '').strip()
        videoname = request.POST.get('videoname', '').strip()
        original_width = request.POST.get('original_width', '').strip()
        original_height = request.POST.get('original_height', '').strip()
        country = request.POST.get('country', '').strip()
        locale = request.POST.get('locale', '').strip()
        latitude = request.POST.get('latitude', '').strip()
        longitude = request.POST.get('longitude', '').strip()

        val, pred_code = predict_single(videoname=videoname, url=url)

        record = deepfake_detection_type.objects.create(
            url=url,
            ipaddress=ipaddress,
            videoname=videoname,
            original_width=original_width,
            original_height=original_height,
            country=country,
            locale=locale,
            latitude=latitude,
            longitude=longitude,
            Prediction=val
        )

        context = {
            'objs': val,
            'prediction_code': pred_code,
            'record': record,
            'form_data': {
                'url': url,
                'ipaddress': ipaddress,
                'videoname': videoname,
                'original_width': original_width,
                'original_height': original_height,
                'country': country,
                'locale': locale,
                'latitude': latitude,
                'longitude': longitude,
            }
        }
        messages.success(request, f"Deepfake Prediction Completed: Video is classified as {val}!")

    return render(request, 'RUser/Predict_deepfake_detection.html', context)


def video_scanner(request):
    """
    State-of-the-Art Deep Learning Media Forensic Scanner.
    Accepts real video (.mp4, .avi, .mov) or image (.jpg, .png) uploads,
    extracts facial frames with OpenCV, and runs PyTorch MesoInception-4 CNN
    plus 2D FFT Frequency-Domain spectrum analysis.
    """
    context = {}

    if request.method == "POST":
        uploaded_file = request.FILES.get('media_file')
        demo_choice = request.POST.get('demo_choice')

        target_file_path = None
        filename = "media_input"

        # Handle demo media
        if demo_choice:
            samples_dir = Path(settings.MEDIA_ROOT) / 'samples'
            if demo_choice == 'demo_video':
                target_file_path = samples_dir / 'sample_interview_clip.mp4'
                filename = "sample_interview_clip.mp4"
            elif demo_choice == 'demo_fake':
                target_file_path = samples_dir / 'sample_deepfake_face.jpg'
                filename = "sample_deepfake_face.jpg"
            elif demo_choice == 'demo_real':
                target_file_path = samples_dir / 'sample_real_portrait.jpg'
                filename = "sample_real_portrait.jpg"
        elif uploaded_file:
            upload_dir = Path(settings.MEDIA_ROOT) / 'uploads'
            upload_dir.mkdir(parents=True, exist_ok=True)
            filename = uploaded_file.name
            target_file_path = upload_dir / filename

            with open(target_file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

        if target_file_path and target_file_path.exists():
            ext = target_file_path.suffix.lower()
            try:
                if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                    report = analyze_video_file(target_file_path)
                elif ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
                    report = analyze_image_file(target_file_path)
                else:
                    messages.error(request, f"Unsupported file format '{ext}'. Please upload MP4, AVI, or JPG/PNG.")
                    return render(request, 'RUser/video_scanner.html', context)

                # Log to database
                deepfake_detection_type.objects.create(
                    url=f"Local Media Forensic Upload: {filename}",
                    ipaddress="127.0.0.1",
                    videoname=filename,
                    original_width="Auto-detected",
                    original_height="Auto-detected",
                    country="Client Scanner",
                    locale="Neural Forensics",
                    latitude="0.0",
                    longitude="0.0",
                    Prediction=report['verdict']
                )

                context = {
                    'report': report,
                    'filename': filename,
                    'file_ext': ext,
                    'has_results': True
                }
                messages.success(request, f"Forensic CNN Analysis Complete: Media classified as {report['verdict']} ({report['confidence']}% confidence)")

            except Exception as e:
                messages.error(request, f"Forensic Processing Error: {str(e)}")
        else:
            messages.warning(request, "Please upload a media file or select a demo sample to scan.")

    return render(request, 'RUser/video_scanner.html', context)
