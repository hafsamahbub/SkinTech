from django.conf import settings
from django.shortcuts import render

# Create your views here.
# views.py
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .yolov8_detection import detect_objects
import os
import cv2

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        # Handle the uploaded image file
        image_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        uploaded_file_url = fs.url(filename)

        # Detect objects using YOLOv8
        image_path = fs.path(filename)
        annotated_image = detect_objects(image_path)

        # Save the annotated image to the static/images folder for display
        output_image_dir = os.path.join(settings.BASE_DIR, 'static', 'images')  # Create a folder for images
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)  # Ensure the directory exists

        output_image_path = os.path.join(output_image_dir, 'output_image.jpg')  # Image path in the static folder
        cv2.imwrite(output_image_path, annotated_image)

        # Create the output image URL for display in HTML
        output_image_url = os.path.join(settings.STATIC_URL, 'images', 'output_image.jpg')

        return render(request, 'upload_image.html', {
            'uploaded_file_url': uploaded_file_url,
            'output_image_url': output_image_url
        })

    return render(request, 'upload_image.html')
