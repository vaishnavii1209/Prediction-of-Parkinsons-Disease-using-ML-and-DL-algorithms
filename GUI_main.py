import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox as ms
#import cv2
import sqlite3
import os
import numpy as np
import time
#from tkvideo import tkvideo
#import video_capture as value
#import lecture_details as detail_data
#import video_second as video1

#import lecture_video  as video

global fn
fn = ""
##############################################+=============================================================
root = tk.Tk()
root.configure(background="skyblue")
# root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Parkison's disease Detection")


# ++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 = Image.open('8.jpg')
image2 = image2.resize((700, 700))

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=650, y=0)  # , relwidth=1, relheight=1)

# image2 =Image.open('logo2.jpg')
# image2 =image2.resize((630,120))




image2 = Image.open('b1.jpg')
image2 = image2.resize((630, 250))

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=315) 

image2 = Image.open('logo3.png')
image2 = image2.resize((200, 200))

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=10, y=335)
#
# label_l1 = tk.Label(root, text="__Parkison's disease Detection__", font=("Times New Roman", 25, 'bold'),
#                     background="skyblue", fg="white", width=30, height=1)
# label_l1.place(x=0, y=20)


label_l1 = tk.Label(root, text="__Welcome To__\n Parkinson's Disease Detection System",font=("Times New Roman", 25, 'bold'),
                    background="skyblue", fg="white", width=30, height=3)
label_l1.place(x=10, y=20)

label_l1 = tk.Label(root, text="EARLY DETECTION AND TREATMENT \n \n CAN LEAD TO BETTER TREATMENT OUTCOMES",font=("Times New Roman", 15, 'bold'),
                    background="skyblue", fg="blue", width=40, height=3)
label_l1.place(x=60, y=200)

#T1.tag_configure("center", justify='center')
#T1.tag_add("center", 1.0, "end")

################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#def clear_img():
#    img11 = tk.Label(root, background='bisque2')
#    img11.place(x=0, y=0)


#################################################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#

def reg():
    from subprocess import call
    call(["python","registration.py"])

def log():
    from subprocess import call
    call(["python","login.py"])
    
def window():
  root.destroy()


button1 = tk.Button(root, text="Login", command=log, width=13, height=1,font=('times', 15, ' bold '), bg="skyblue", fg="white")
button1.place(x=300, y=350)

button2 = tk.Button(root, text="Sign Up",command=reg,width=13, height=1,font=('times', 15, ' bold '), bg="skyblue", fg="white")
button2.place(x=300, y=420)

button3 = tk.Button(root, text="Exit",command=window,width=13, height=1,font=('times', 15, ' bold '), bg="skyblue", fg="white")
button3.place(x=300, y=490)

root.mainloop()