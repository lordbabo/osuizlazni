import numpy as np
from keras.models import load_model
from PIL import Image
import keras.utils as image


model = load_model("OSU_LV8/lv8_zad1_model_keras.keras")

img = image.load_img("OSU_LV8/test.png",target_size = (28, 28), color_mode = "grayscale")
img_array = image.img_to_array(img)
img_array = img_array.astype("float32") / 255
img_array=np.expand_dims(img_array,-1)
img_array = img_array.reshape(-1,784)

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

print("Predicted:",predicted_class)