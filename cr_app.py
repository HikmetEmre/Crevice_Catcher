### New App For Wall crack detection ###
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import streamlit as st

#### Page Config ###
st.set_page_config(
    page_title="Crevice Catcher",
    page_icon="https://imageio.forbes.com/specials-images/imageserve/63bdffc05989c30c33964a41/Artificial-Intelligence/960x0.png?format=png&width=960",
    menu_items={
        "Get help": "mailto:hikmetemreguler@gmail.com",
        "About": "For More Information\n" + "https://github.com/HikmetEmre/Crevice_Catcher"
    }
)

### Title of Project ###
st.title("**:red[Crevice Catcher: A model which is detecting cracks (crevices) in walls ]**")

### Markdown ###
st.markdown("**:red[CreviceCatcher] is a cutting-edge ML model adeptly trained on a diverse dataset consisting of 20,000 pristine wall images and 20 real-world cracked wall images, enabling it to accurately detect and flag even the subtlest wall cracks, empowering users to proactively address potential structural issues **.")

### Adding Image ###
st.image("https://raw.githubusercontent.com/HikmetEmre/Crevice_Catcher/main/img_def.jpg")

st.markdown("**The :red[CreviceCatcher] model was developed using TensorFlow and underwent 20 epochs of training. After training, the model achieved an impressive validation accuracy of :blue[0.9900], while other success parameters showed high values of approximately precision 0.9915, recall 0.9851, and accuracy 0.9883, indicating its exceptional performance in accurately detecting wall cracks.** ")


#### Header and definition of columns ###
st.header("**META DATA**")

st.markdown ("**The dataset used for training the CreviceCatcher model consists of a total of 40,000 images, with 20,000 images depicting healthy concrete walls and another 20,000 images showcasing damaged concrete walls. This diverse dataset ensures the model's ability to distinguish between normal and cracked wall conditions accurately.** ")

st.markdown("**Below are two example images from each class in the dataset: one depicting a healthy concrete wall, and the other showing a damaged concrete wall.** ")

st.image("https://raw.githubusercontent.com/HikmetEmre/Crevice_Catcher/main/meta%20data%20for%202%20class.jpg")







#---------------------------------------------------------------------------------------------------------------------

### Sidebar Markdown ###
st.sidebar.markdown("**UPLOAD** , **:red[Wall Image] Below & See The Condition Concrete wall!")

### Define Sidebar Input's ###
Image = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])

if Image is not None:
    # Read and preprocess the uploaded image
    img = cv2.imdecode(np.fromstring(Image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    input_data = img.astype(np.float32) / 255.0  # Normalize the image to [0, 1]


#---------------------------------------------------------------------------------------------------------------------

### Recall Model ###
import tensorflow as tf
import cv2
import numpy as np

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='quantized_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = cv2.imread(Image)
input_data = img.astype(np.float32) / 255.0
# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], np.expand_dims(input_data, axis=0))

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])



yhat = output_data







#---------------------------------------------------------------------------------------------------------------------

st.header("Results")

### Result Screen ###
if st.sidebar.button("Submit"):

    ### Info message ###
    st.info("You can find the result below.")

    ### Inquiry Time Info ###
    from datetime import date, datetime

    today = date.today()
    time = datetime.now().strftime("%H:%M:%S")

    ### For showing results create a df ###
    results_df = pd.DataFrame({
    'Date': [today],
    'Time': [time],
    'Image': [Image],
    'Wall Condition': [yhat]
    })

   


    st.table(results_df)

    if yhat > 0.5:
        st.markdown('**:Predicted class is :red[DAMAGED]**')
        st.image("https://raw.githubusercontent.com/HikmetEmre/Crevice_Catcher/main/if_crack.jpg")

    else:
        st.markdown('**:Predicted class is :blue[HEALTHY]**')
        st.image('https://raw.githubusercontent.com/HikmetEmre/Crevice_Catcher/main/if_healthy.jpg')

