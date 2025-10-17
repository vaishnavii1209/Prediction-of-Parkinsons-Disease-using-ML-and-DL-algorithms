from tkinter import *



######################
def Train():
    """GUI"""
    import tkinter as tk
    from tkinter import ttk
    import numpy as np
    import pandas as pd

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder
    #import webview 
    import webbrowser

    root = tk.Tk()

    root.geometry("1500x850+0+5")
    root.title("Parkison's disease Detection")
    root.configure(background="#CD5C5C")
    
    
    gender = IntVar()
    PPE = DoubleVar()
    DFA = DoubleVar()
    RPDE = DoubleVar()
    numPulses = IntVar()
    numPeriodsPulses = IntVar()
    meanPeriodPulses = DoubleVar()
    minIntensity = DoubleVar()
    maxIntensity = DoubleVar()
    app_LT_entropy_log_coef = DoubleVar()
    tqwt_kurtosisValue_dec = DoubleVar()
   
    

    # ===================================================================================================================

    def Detect():
        e1 = gender.get()
        print(e1)
        e2 = PPE.get()
        print(e2)
        e3 = DFA.get()
        print(e3)
        e4 = RPDE.get()
        print(e4)
        e5 = numPulses.get()
        print(e5)
        e6 = numPeriodsPulses.get()
        print(e6)
        e7 = meanPeriodPulses.get()
        print(e7)
        e8 = minIntensity.get()
        print(e8)
        e9 = maxIntensity.get()
        print(e9)
        e10 = app_LT_entropy_log_coef.get()
        print(e10)
        e11 = tqwt_kurtosisValue_dec.get()
        print(e11)
        
        #########################################################################################

        from joblib import dump, load
        a1 = load('D:/Ravina data/parkisons disease/PD_model.joblib')
        v = a1.predict([[e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11]])
        print(v)
        if v[0]==1:
            print("PD")
            PD = tk.Label(root, text="Parkison's disease detected", background="red", foreground="white", font=('times', 20, ' bold '), width=35,height=2)
            PD.place(x=400, y=450)
            
            PD = tk.Button(root, text="Tap", command=window,background="black",font=('times', 20, ' bold '), width=10, fg="white")
            PD.place(x=600, y=550)
            
        else:
            print("NPD")
            NPD = tk.Label(root, text="Parkison's disease not detected", background="#2F4F4F", foreground="white",font=('times', 20, ' bold '),width=40,height=2)
            NPD.place(x=700, y=300)  
            
            
            
    url = "https://sahyadrihospital.com/blog/prevent-parkinson-disease/"
            
        
               
    def window():
        webbrowser.open(url)
        
              
              
              
    l1 = tk.Label(root, text="Gender", background="Bisque", font=('times', 20, ' bold '), width=15)
    l1.place(x=150, y=100)
    gender=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=gender)
    gender.place(x=500,y=100)

  

    l2= tk.Label(root, text="PPE", background="Bisque", font=('times', 20, ' bold '), width=15)
    l2.place(x=150, y=150)
    PPE = tk.Entry(root, bd=2, width=5, font=("TkDefaultFont", 20), textvar=PPE)
    PPE.place(x=500, y=150)

    l3 = tk.Label(root, text="DFA", background="Bisque", font=('times', 20, ' bold '), width=15)
    l3.place(x=150, y=200)
    DFA = tk.Entry(root, bd=2, width=5, font=("TkDefaultFont", 20), textvar=DFA)
    DFA.place(x=500, y=200)

    l4 = tk.Label(root, text="RPDE", background="Bisque",font=('times', 20, ' bold '), width=15)
    l4.place(x=150, y=250)
    RPDE = tk.Entry(root, bd=2, width=5, font=("TkDefaultFont", 20), textvar=RPDE)
    RPDE.place(x=500, y=250)

    l5 = tk.Label(root, text="numPulses", background="Bisque", font=('times', 20, ' bold '), width=15)
    l5.place(x=150, y=300)
    numPulses = tk.Entry(root, bd=2, width=5, font=("TkDefaultFont", 20), textvar=numPulses)
    numPulses.place(x=500, y=300)

    l6 = tk.Label(root, text="numPeriodsPulses", background="Bisque", font=('times', 20, ' bold '), width=15)
    l6.place(x=150, y=350)
    numPeriodsPulses = tk.Entry(root, bd=2, width=5, font=("TkDefaultFont", 20), textvar=numPeriodsPulses)
    numPeriodsPulses.place(x=500, y=350)

    l7 = tk.Label(root, text="meanPeriodPulses", background="Bisque",font=('times', 20, ' bold '), width=15)
    l7.place(x=700, y=100)
    meanPeriodPulses = tk.Entry(root, bd=2, width=5, font=("TkDefaultFont", 20), textvar=meanPeriodPulses)
    meanPeriodPulses.place(x=1050, y=100)
    
    l8 = tk.Label(root, text="minIntensity", background="Bisque",font=('times', 20, ' bold '), width=15)
    l8.place(x=700, y=150)
    minIntensity = tk.Entry(root, bd=2, width=5, font=("TkDefaultFont", 20), textvar=minIntensity)
    minIntensity.place(x=1050, y=150)
    
    l8 = tk.Label(root, text="maxIntensity", background="Bisque",font=('times', 20, ' bold '), width=15)
    l8.place(x=700, y=200)
    maxIntensity = tk.Entry(root, bd=2, width=5, font=("TkDefaultFont", 20), textvar=maxIntensity)
    maxIntensity.place(x=1050, y=200)
    
    l8 = tk.Label(root, text="app_LT_entropy_log_coef", background="Bisque",font=('times', 20, ' bold '), width=19)
    l8.place(x=700, y=250)
    app_LT_entropy_log_coef = tk.Entry(root, bd=2, width=5, font=("TkDefaultFont", 20), textvar=app_LT_entropy_log_coef)
    app_LT_entropy_log_coef.place(x=1050, y=250)
    
    l8 = tk.Label(root, text="tqwt_kurtosisValue_dec", background="Bisque",font=('times', 20, ' bold '), width=19)
    l8.place(x=700, y=300)
    tqwt_kurtosisValue_dec = tk.Entry(root, bd=2, width=5, font=("TkDefaultFont", 20), textvar=tqwt_kurtosisValue_dec)
    tqwt_kurtosisValue_dec.place(x=1050, y=300)


  

    button1 = tk.Button(root, text="Submit", command=Detect,background="black",font=('times', 20, ' bold '), width=10, fg="white")
    button1.place(x=965, y=350)
    
    

    root.mainloop()


Train()
