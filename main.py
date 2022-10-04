# FastAPI tut https://youtu.be/t6NI0u_lgNo?t=70

# click run python file to start server on port 8000 OR run: uvicorn main:app --reload
# to test http://localhost:8000/docs#/default/predict_predict_post

 
from itertools import count
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2

# init fastapi instance
app = FastAPI()

# origins from where to accept requests
origins = [
    "http://localhost",
    "http://localhost:3000",
]
# middleware to allow cross origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# https://youtu.be/t6NI0u_lgNo?t=1670 for tf serving

# return the count for only the first folder 

path = '../saved_models/'

files = folders = 0

for _, dirnames, filenames in os.walk(path):
  # ^ this idiom means "we won't be using this value"
    files += len(filenames)
    folders += len(dirnames)

# model from folder if not .h5 file 
model_folder = path+str(int(folders/3))+"/"

# '../model_img_gen.h5' 'strain_doc_model.h5'
model = "../strain_doc_model.h5"
# load model
MODEL = tf.keras.models.load_model(model)
print("\n model loaded from folder: ",model,"\n")
# class names array return to client
CLASS_NAMES = ['Algae', 'Aphids', 'Armored Scale Bugs', 'Boron Deficiency', 'Bud Rot or Mold', 'Calcium Deficiency', 'Corn Earworm', 'Crickets', 'Fungus Gnats', 'Grasshopper', 'Healthy', 'Heat Stress', 'Hemp Borer', 'Iron Deficiency', 'Leaf Miner', 'Leafhopper', 'Light Burn', 'Magnesium Deficiency', 'Mealybugs', 'Mutation', 'Nitrogen Deficiency', 'Nutrient Burn', 'Overwatering', 'Ph Fluctuation', 'Phosphorus Deficiency', 'Potassium Deficiency', 'Root Rot', 'Russet Mites', 'Slug', 'Snails', 'Spider Mites', 'Spray Burn', 'Stink Bug', 'Sulfur Deficiency', 'Thrips', 'Underwatering', 'Viruses', 'White Powdery Mildew', 'Whiteflies', 'Zinc Deficiency', 'Nitrogen Toxicity']

# test route
@app.get("/")
async def home():
    return "Hello, I am alive"

# read file from client function
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# predict route
@app.post("/predict")

# upload file from client
async def predict(
    file: UploadFile = File(...)
):
  
    # read file as image
    img = read_file_as_image(await file.read())
    
    # resize image
    image =cv2.resize(img,(256,256))
    
    # add batch dimension
    img_batch = np.expand_dims(image, 0)
    
    # predict
    predictions = MODEL.predict(img_batch)

    # get top 3 % in array
    sorted_predictions = sorted(zip(predictions[0]), reverse=True)[:3]

    # predict class name
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    # show confidence in prediction
    confidence = np.max(predictions[0])
    
    print('\n Top 3 --> ',sorted_predictions,'\n\n Predicted Class --> ', predicted_class,'\n\n Confidence % --> ',confidence, '\n')
    
    # return class name and confidence to client
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

# run server on port 8000 when file is run directly
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
