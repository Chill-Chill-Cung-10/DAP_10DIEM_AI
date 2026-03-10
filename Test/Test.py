import os
import numpy as np
from PIL import Image
import tensorflow as tf

# load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
interpreter = tf.lite.Interpreter(model_path=os.path.join(BASE_DIR, "plant_disease_model.tflite"))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ä‘á»c áº£nh
img = Image.open("test_leaf.jpeg")

# resize Ä‘Ãºng kÃ­ch thÆ°á»›c model
img = img.resize((160,160))

# chuyá»ƒn sang numpy
img = np.array(img)

# normalize náº¿u cáº§n
img = img / 255.0

# thÃªm batch dimension
img = np.expand_dims(img, axis=0).astype(np.float32)

print(img.shape)   # (1,160,160,3)

# Ä‘Æ°a vÃ o model
interpreter.set_tensor(input_details[0]['index'], img)

# cháº¡y model
interpreter.invoke()

# láº¥y káº¿t quáº£
output = interpreter.get_tensor(output_details[0]['index'])

print("Prediction:", output)
print("Predicted class:", np.argmax(output))

labels = [
"Apple___Apple_scab",
"Apple___Black_rot",
"Apple___Cedar_apple_rust",
"Apple___healthy",
"Blueberry___healthy",
"Cherry___Powdery_mildew",
"Cherry___healthy",
"Corn___Cercospora_leaf_spot Gray_leaf_spot",
"Corn___Common_rust",
"Corn___Northern_Leaf_Blight",
"Corn___healthy",
"Grape___Black_rot",
"Grape___Esca_(Black_Measles)",
"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
"Grape___healthy",
"Orange___Haunglongbing_(Citrus_greening)",
"Peach___Bacterial_spot",
"Peach___healthy",
"Pepper,_bell___Bacterial_spot",
"Pepper,_bell___healthy",
"Potato___Early_blight",
"Potato___Late_blight",
"Potato___healthy",
"Raspberry___healthy",
"Soybean___healthy",
"Squash___Powdery_mildew",
"Strawberry___Leaf_scorch",
"Strawberry___healthy",
"Tomato___Bacterial_spot",
"Tomato___Early_blight",
"Tomato___Late_blight",
"Tomato___Leaf_Mold",
"Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites Two-spotted_spider_mite",
"Tomato___Target_Spot",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus",
"Tomato___Tomato_mosaic_virus",
"Tomato___healthy"
]

pred = output[0]

for i, prob in enumerate(pred):
    print(f"{labels[i]} : {prob*100:.2f}%")

from FlagEmbedding import BGEM3FlagModel

embedding_model = BGEM3FlagModel("BAAI/bge-m3")

# after prediction
predicted_label = labels[np.argmax(output)]

print("Predicted:", predicted_label)

# embed label
embedding = embedding_model.encode([predicted_label])

vector = embedding["dense_vecs"][0]

print("Embedding length:", len(vector))