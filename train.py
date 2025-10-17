from subprocess import call
import tkinter as tk
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
from tkinter import ttk
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

root = tk.Tk()
root.title("train")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
# ++++++++++++++++++++++++++++++++++++++++++++

image2 = Image.open('4.jpg')

image2 = image2.resize((w, h), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)


background_label = tk.Label(root, image=background_image)
background_label.image = background_image
background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)


label_l2 = tk.Label(root, text="___Parkison's disease Detection___",font=("times", 30, 'bold','italic'),
                    background="black", fg="white", width=60, height=1)
label_l2.place(x=0, y=10)


# _+++++++++++++++++++++++++++++++++++++++++++++++++++++++
data = pd.read_csv("D:/Ravina data/parkisons disease/dataset.csv")



data = data.dropna()

le = LabelEncoder()


def Data_Preprocessing():
    data = pd.read_csv("D:/Ravina data/parkisons disease/dataset.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

   


    """Feature Selection => Manual"""
    x = data.drop(['id','class'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['class']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    

    load = tk.Label(root, font=("Tempus Sans ITC", 15, "bold"), width=50, height=2, background="gray",
                    foreground="white", text="Data Loaded=>Splitted into 80% for Training & 20% for Testing")
    load.place(x=250, y=130)


def Model_Training():
    data = pd.read_csv("D:/Ravina data/parkisons disease/dataset.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""


    """Feature Selection => Manual"""
    x = data.drop(['id','class'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['class']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=123456789)

    from sklearn.svm import SVC
    svcclassifier = SVC(kernel='linear')
    svcclassifier.fit(x_train, y_train)

    y_pred = svcclassifier.predict(x_test)
    print(y_pred)

    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, y_pred)))
    print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))
    
    label4 = tk.Label(root,text =str(repo),width=45,height=20,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=400,y=130)
    
    label5 = tk.Label(root,text ="Accracy : "+str(ACC)+"%\nModel saved as PD_model.joblib",width=45,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=400,y=560)
    from joblib import dump
    dump (svcclassifier,"PD_model.joblib")
    print("Model saved as PD_model.joblib")

def predication():
    from subprocess import call
    call(["python","Check_Prediction.py"])
    root.destroy()




def window():
    root.destroy()
def cnn():
    from subprocess import call
    call(["python","GUI_Master_plant.py"])
    

button2 = tk.Button(root, foreground="white", background="maroon", font=("Times", 14, "bold"),
                    text="Data_Preprocessing", command=Data_Preprocessing, width=20, height=2)
button2.place(x=10, y=120)

button3 = tk.Button(root, foreground="white", background="brown", font=("Times", 14, "bold"),
                    text="Model Training", command=Model_Training, width=20, height=2)
button3.place(x=10, y=220)

button4 = tk.Button(root, foreground="white", background="brown", font=("Times", 14, "bold"),
                    text="Prediction", command=predication, width=20, height=2)
button4.place(x=10, y=320)
exit = tk.Button(root, text="CNN Model", command=cnn, width=20, height=2, font=('times', 14, ' bold '),bg="brown",fg="white")
exit.place(x=10, y=420)

root.mainloop()

'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''