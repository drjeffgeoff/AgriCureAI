import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from .recommendations import recommendations  
# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow as tf




# Suppress TensorFlow oneDNN custom operations message
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the model 
model_path = os.path.join(settings.BASE_DIR, 'model/model_InceptionV3.h5')
# model_path = os.path.join(settings.BASE_DIR, 'model/model_DenseNet121.h5')
# model_path = os.path.join(settings.BASE_DIR, 'model/model_MobileNetV2.h5')

model = tf.keras.models.load_model(model_path)



# class_names = [
#     'Corn___Common_Rust',
#     'Corn___Gray_Leaf_Spot',
#     'Corn___Healthy',
#     'Corn___Leaf_Blight',
#     'Potato___Early_Blight',
#     'Potato___Healthy',
#     'Potato___Late_Blight',
#     'Rice___Brown_Spot',
#     'Rice___Healthy',
#     'Rice___Hispa',
#     'Rice___Leaf_Blast',
#     'Wheat___Brown_Rust',
#     'Wheat___Healthy',
#     'Wheat___Yellow_Rust'
# ]

class_names = {0: 'Corn___Common_Rust', 1: 'Corn___Gray_Leaf_Spot',
                2: 'Corn___Healthy', 3: 'Corn___Leaf_Blight', 
                4: 'Potato___Early_Blight', 5: 'Potato___Healthy', 
                6: 'Potato___Late_Blight', 7: 'Rice___Brown_Spot', 
                8: 'Rice___Healthy', 9: 'Rice___Hispa', 10: 'Rice___Leaf_Blast', 
                11: 'Wheat___Brown_Rust', 12: 'Wheat___Healthy', 13: 'Wheat___Yellow_Rust'}

def home(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        img_path = default_storage.save(uploaded_file.name, uploaded_file)
        img_path = os.path.join(settings.MEDIA_ROOT, img_path)

        # Load and preprocess the image
        img = Image.open(img_path)
        img = img.resize((224, 224))  
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # Normalize the image
        img_array = img_array / 255.0  

        # Make predictions
        predictions = model.predict(img_array)
        score = predictions[0]

        # Get the highest confidence class
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        confidence_format = '{:.4f}'.format(confidence)

        recommendation = recommendations.get(predicted_class, {'solution': 'No recommendation available.', 'prevention': 'No prevention measure available.',
           
        
        })


        context = {
            'predicted_class': predicted_class,
            'confidence_format': confidence_format,
            'image_url': default_storage.url(img_path),
            'solution': recommendation['solution'],
            'prevention': recommendation['prevention'],
            'disease': recommendation.get('disease', ''),
            'text': recommendation.get('text', ''),
            'medicine_products': recommendation.get('medicine_products', [])
        }
       
        return render(request, 'prediction.html', context)

    return render(request, 'home.html')
