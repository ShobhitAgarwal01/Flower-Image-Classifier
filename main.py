from keras.models import load_model
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import mysql.connector

vgg_model = load_model('vgg19model.h5')
custom_model = load_model('custommodel.h5')
resnet_model = load_model('resnetkaggle.h5')
xception_model = load_model('xceptionmodel.h5')
densenet_model = load_model('densenetmodel.h5')

# Setting the Page Config
st.set_page_config(
    page_title="Plant Species Identification",
    page_icon="icons\PlantIcon.png"
)

# Creating two columns with display flex and align items center
col1, col2 = st.columns([0.2, 0.7])

# Add icon to first column
with col1:
    st.image("icons\PlantIcon_2.png", width=50, use_column_width=False)

with col2:
    st.title("Plants Image Classifier")

# Connecting to the MySQL database
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Shobhit23",
    database="predictied"
)


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1528834342297-fdefb9a5a92b?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxleHBsb3JlLWZlZWR8Nnx8fGVufDB8fHx8&w=1000&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()


# Defining the Functions and the Classes that are going to be Inputed
class_names = ["Daisy","Lavender","Lily","Rose","Sunflower"]

def predict_vggnet(image):
    img = Image.open(image)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = vgg_model.predict(img)
    return prediction

def predict_custommodel(image):
    img = Image.open(image)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = custom_model.predict(img)
    return prediction

def predict_resnetmodel(image):
    img = Image.open(image)
    img = img.resize((224,224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = resnet_model.predict(img)
    return prediction

def predict_xceptionnetmodel(image):
    img = Image.open(image)
    img = img.resize((224,224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = xception_model.predict(img)
    return prediction

def predict_densenetmodel(image):
    img = Image.open(image)
    img = img.resize((224,224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = densenet_model.predict(img)
    return prediction


st.set_option('deprecation.showfileUploaderEncoding', False)

# Defining a function to insert the prediction results into the database
def insert_prediction(model_choice, class_name, class_prob):

    mycursor = mydb.cursor()
    # Convert class_prob from numpy.float32 to native Python float
    class_prob = float(class_prob)
    sql = "INSERT INTO predicted (model_name, class_name, class_prob) VALUES (%s, %s, %s)"
    val = (model_choice, class_name, class_prob)
    mycursor.execute(sql, val)
    mydb.commit()
    print("Prediction inserted successfully")


# Adding a selection box
model_choice = st.selectbox("Choose a model", ("VGG19","Custom Model","ResNet","Xception","DenseNet"))
image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
prediction = None

if image is not None:
    if model_choice == "VGG19":
        prediction = predict_vggnet(image)
    elif model_choice == "Custom Model":
        prediction = predict_custommodel(image)
    elif model_choice == "ResNet":
        prediction = predict_resnetmodel(image)
    elif model_choice == "Xception":
        prediction = predict_xceptionnetmodel(image)
    elif model_choice == "DenseNet":
        prediction = predict_densenetmodel(image)

    col1, col2, col3 = st.columns([1, 1, 2])

    if image is not None:
        # Opening and resizing the image
        img = Image.open(image)
        img = img.resize((350, 340))
        # Display the uploaded image in col1
        col1.image(img, caption='Image Uploaded', use_column_width=False )

    if prediction is not None:
        class_index = np.argmax(prediction[0])
        class_name = class_names[class_index]
        class_prob = prediction[0][class_index]
        insert_prediction(model_choice, class_name, class_prob)


        # Displaying the bar graph in col3
        probabilities = prediction[0]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.bar(class_names, probabilities)
        ax.set_xlabel("Class")
        ax.set_ylabel("Probability")
        col3.write(fig)

        # Displaying the progress bar and predicted class below the image and bar graph
        col1.write("")
        progress_bar_container = st.container()
        with progress_bar_container:
            st.write("")
            st.write(f"Predicted Class: {class_name} with Probability: {class_prob:.2%}")
            st.progress(float(class_prob))
