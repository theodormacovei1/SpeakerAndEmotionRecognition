#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf
import tkinter as tk
from tkinter import filedialog

import numpy as np

from tensorflow.keras import datasets, layers, models, Sequential
import sklearn.model_selection as sk
import glob
import os
import sounddevice as sd
import librosa
import librosa.display
from scipy.io.wavfile import write

import pickle

from keras.regularizers import l2



import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg







from PIL import Image, ImageTk



person_model = models.load_model('person_model')
emotion_model = models.load_model('emotion_model')




filename=None
fs = 44100
no_mfcc=20
no_mfcc_emotion=40
max_len_person=300
max_len_emotion=400


def emotion_val2name(emotion):
    if (emotion==0):
        return "Neutral"
    elif (emotion==1):
        return "Happy"
    elif (emotion==2):
        return "Angry"


iconPath="C:\\Users\\tmacovei001\\Desktop\\Speaker Recognition\\Code\\icon.ico"
helpPageText="Speaker Recognition is an application used to recognise people and emotions based on their voice. \
\n\n1.Test \n In order to recognise a person or an emotion click on the Test Button. This opens the Test Window. \
You can choose to load a file from memory or to record a file. After you load the file check the options bellow \
to select whether you want to perform a speaker recognition, an emotion recognition or both of them.\
\n\n2. Add or Remove Users \
\nIn the Add or Remove Users window you can see all the users which are loaded in the model. If you  want to add\
another user press Add User, enter your name and select the appropiate wav files. If you want to delete a user select\
him from the users list and then press the Delete User button\
\n\n\n\n Developer: Theodor Macovei \n Version: 1 \n Date: 10.06.2020"

def predict(VoiceState,EmotionState):
    if (filename==None):
        PredictPage = tk.Toplevel(root)
        PredictPage.title("Test")
        PredictPage.iconbitmap(iconPath)
        PredictPage.geometry("260x150")
        MessagePredict = tk.Message(PredictPage,width=250, text = "Please select a file")
        MessagePredict.place(x=0,y=0)
    else:

        if(VoiceState.get()==0 and EmotionState.get()==0):
            PredictPage = tk.Toplevel(root)
            PredictPage.title("Test")
            PredictPage.iconbitmap(iconPath)
            PredictPage.geometry("260x150")
            MessagePredict = tk.Message(PredictPage,width=250, text = "Please select an option")
            MessagePredict.place(x=0,y=0)
        
        elif(VoiceState.get()==1 and EmotionState.get()==0):
            file_temp,fs_temp=librosa.load(filename, sr=fs)
            mfcc_test=librosa.feature.mfcc(file_temp,sr=fs, S=None, n_mfcc=no_mfcc)
            if (max_len_person > mfcc_test.shape[1]):
                pad_width = max_len_person - mfcc_test.shape[1]
                mfcc_test = np.pad(mfcc_test, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc_test = mfcc_test[:, :max_len_person]
            mfcc_test=mfcc_test.reshape((1,20,300,1)) 
            probability_model = Sequential([person_model, layers.Softmax()])
            predictions = probability_model.predict(mfcc_test)

            
            PredictPage = tk.Toplevel(root)
            PredictPage.title("Test")
            PredictPage.iconbitmap(iconPath)
            PredictPage.geometry("200x210")
            LabelPersonName = tk.Label(PredictPage, text = "Person")
            LabelPersonName.place(x=0,y=10)    
            LabelPersonVal = tk.Label(PredictPage, text = np.argmax(predictions[0]))
            LabelPersonVal.place(x=40,y=10)          
            
            
            if (np.argmax(predictions[0])==0):
                load = Image.open("theodor.jpg")
            elif (np.argmax(predictions[0])==1):
                load = Image.open("andreea.jpg")
            elif (np.argmax(predictions[0])==2):
                load = Image.open("ioana.jpg") 
            elif (np.argmax(predictions[0])==3):
                load = Image.open("anonim.jpg")
            else:
                load = Image.open("adriana.jpg") 
            photo = ImageTk.PhotoImage(load) 
            label = tk.Label(PredictPage,image=photo)
            label.image = photo # keep a reference!
            label.place(x=0,y=40)    

            
            
            
        elif(VoiceState.get()==0 and EmotionState.get()==1):
            file_temp,fs_temp=librosa.load(filename, sr=fs)
            mfcc_test=librosa.feature.mfcc(file_temp,sr=fs, S=None, n_mfcc=no_mfcc_emotion)
            if (max_len_emotion > mfcc_test.shape[1]):
                pad_width = max_len_emotion - mfcc_test.shape[1]
                mfcc_test = np.pad(mfcc_test, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc_test = mfcc_test[:, :max_len_emotion]
            mfcc_test=mfcc_test.reshape((1,40,400,1)) 
            probability_model = Sequential([emotion_model, layers.Softmax()])
            predictions = probability_model.predict(mfcc_test)
            
            
            PredictPage = tk.Toplevel(root)
            PredictPage.title("Test")
            PredictPage.iconbitmap(iconPath)
            PredictPage.geometry("290x190")
            LabelEmotionName = tk.Label(PredictPage, text = "Emotion")
            LabelEmotionName.place(x=0,y=10)    
            emotion=np.argmax(predictions[0])
            emotion=emotion_val2name(emotion)
            LabelEmotionVal = tk.Label(PredictPage, text = emotion )
            LabelEmotionVal.place(x=50,y=10)
            
            figure = plt.Figure(figsize=(4,2), dpi=70)
            ax = figure.add_subplot(111)
            chart_type = FigureCanvasTkAgg(figure, PredictPage)
            chart_type.get_tk_widget().place(x=0,y=30)
            thisplot = ax.bar(["Neutral","Happy","Angry"], 100*predictions[0])            
            ax.set_ylabel('Percentage')
            
        elif(VoiceState.get()==1 and EmotionState.get()==1):
            file_temp,fs_temp=librosa.load(filename, sr=fs)
            mfcc_test=librosa.feature.mfcc(file_temp,sr=fs, S=None, n_mfcc=no_mfcc)
            if (max_len_person > mfcc_test.shape[1]):
                pad_width = max_len_person - mfcc_test.shape[1]
                mfcc_test = np.pad(mfcc_test, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc_test = mfcc_test[:, :max_len_person]
            mfcc_test=mfcc_test.reshape((1,20,300,1)) 
            probability_model = Sequential([person_model, layers.Softmax()])
            predictions = probability_model.predict(mfcc_test)

            PredictPage = tk.Toplevel(root)
            PredictPage.title("Test")
            PredictPage.iconbitmap(iconPath)
            PredictPage.geometry("450x200")


            LabelPersonName = tk.Label(PredictPage, text = "Person")
            LabelPersonName.place(x=0,y=10)    
            LabelPersonVal = tk.Label(PredictPage, text = np.argmax(predictions[0]))
            LabelPersonVal.place(x=40,y=10) 
            
            if (np.argmax(predictions[0])==0):
                load = Image.open("theodor.jpg")
            elif (np.argmax(predictions[0])==1):
                load = Image.open("andreea.jpg")
            elif (np.argmax(predictions[0])==2):
                load = Image.open("ioana.jpg") 
            elif (np.argmax(predictions[0])==3):
                load = Image.open("anonim.jpg")
            else:
                load = Image.open("adriana.jpg")
            photo = ImageTk.PhotoImage(load) 
            label = tk.Label(PredictPage,image=photo)
            label.image = photo 
            label.place(x=0,y=30)   
            
            file_temp,fs_temp=librosa.load(filename, sr=fs)
            mfcc_test=librosa.feature.mfcc(file_temp,sr=fs, S=None, n_mfcc=no_mfcc_emotion)
            if (max_len_emotion > mfcc_test.shape[1]):
                pad_width = max_len_emotion - mfcc_test.shape[1]
                mfcc_test = np.pad(mfcc_test, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc_test = mfcc_test[:, :max_len_emotion]
            mfcc_test=mfcc_test.reshape((1,40,400,1)) 
            probability_model = Sequential([emotion_model, layers.Softmax()])
            predictions = probability_model.predict(mfcc_test)

        

            LabelEmotionName = tk.Label(PredictPage, text = "Emotion")
            LabelEmotionName.place(x=150,y=10)    
            emotion=np.argmax(predictions[0])
            emotion=emotion_val2name(emotion)
            LabelEmotionVal = tk.Label(PredictPage, text = emotion )
            LabelEmotionVal.place(x=200,y=10)
            
            figure = plt.Figure(figsize=(4,2), dpi=70)
            ax = figure.add_subplot(111)
            chart_type = FigureCanvasTkAgg(figure, PredictPage)
            chart_type.get_tk_widget().place(x=155,y=30)
            thisplot = ax.bar(["Neutral","Happy","Angry"], 100*predictions[0])            
            ax.set_ylabel('Percentage')
    
def Record():
    global filename
    seconds = 3 
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait() 
    filename="voice_test.wav"
    write(filename, fs, recording)
    #sd.play(recording)
    
    
    
    
def TestFromFile():
    global filename
    filename=filedialog.askopenfilename(initialdir="C:\\Users\\tmacovei001\\Desktop\\Speaker Recognition\\Datasets",title="Select wav files",filetypes=[("wav files","*.wav")])
    
def AddUserFromFile():
    selectfile=filedialog.askopenfilenames(initialdir="C:\\Users\\tmacovei001\\Desktop\\Speaker Recognition\\Datasets",title="Select wav files",filetypes=[("wav files","*.wav")])

    

def OpenHelpPage():
    HelpPage = tk.Toplevel(root)
    HelpPage.title("Help")
    HelpPage.iconbitmap(iconPath)
    HelpPage.geometry("280x450")
    
    
    MessageHelp = tk.Message(HelpPage,width=250, text = helpPageText)
    MessageHelp.place(x=0,y=0)
    
    
    
def OpenTestPage():
    TestPage = tk.Toplevel(root)
    TestPage.title("Test")
    TestPage.iconbitmap(iconPath)
    TestPage.geometry("260x300")
    
                        
    ButtonTestFile=tk.Button(TestPage,text="Load file",width=10,height=1,command=TestFromFile)
    
    
    ButtonTestRecord=tk.Button(TestPage,text="Record",width=10,height=1,command=Record)
    
    
    
    VoiceState = tk.IntVar()
    EmotionState = tk.IntVar()
    VoiceCheckButton=tk.Checkbutton(TestPage, text="Voice Recognition",variable=VoiceState)
    EmotionCheckButton=tk.Checkbutton(TestPage, text="Emotion Recognition",variable=EmotionState)
    
    ButtonTest=tk.Button(TestPage,text="Test",width=10,height=1,command=lambda:predict(VoiceState,EmotionState))
    
    ButtonClose=tk.Button(TestPage,text="Close",width=10,height=1,command=TestPage.destroy)
    
    ButtonTestFile.place(x=10,y=10)
    ButtonTestRecord.place(x=10,y=50)
    VoiceCheckButton.place(x=10,y=90)
    EmotionCheckButton.place(x=10,y=110)
    ButtonTest.place(x=10,y=150)
    
    ButtonClose.place(x=10,y=250)
    
    
    
    
def OpenAddOrRemovePage():
    with open('Saved Persons','rb') as f:
        y1=pickle.load(f)
    
    AddOrRemovePage = tk.Toplevel(root)
    AddOrRemovePage.title("Add or Remove")
    AddOrRemovePage.iconbitmap(iconPath)
    AddOrRemovePage.geometry("260x300")
    
    
    AvailableUsersTitleLabel = tk.Label(AddOrRemovePage,text = "Available Users") 
    UniqueList=[]
    for item in y1: 
        if item not in UniqueList: 
            UniqueList.append(item) 
    UsersListbox = tk.Listbox(AddOrRemovePage)
    for item in UniqueList:
        UsersListbox.insert(tk.END, item)
        
    ButtonAddUser=tk.Button(AddOrRemovePage,text="Add User",width=10,height=1,command=AddUserPage)
    ButtonDeleteUser=tk.Button(AddOrRemovePage,text="Delete User",width=10,height=1,command=DeleteUserPage)

    
    AvailableUsersTitleLabel.place(x=10,y=10) 
    UsersListbox.place(x=10,y=40)      
    ButtonAddUser.place(x=10,y=250)
    ButtonDeleteUser.place(x=170,y=250)
    
    
def AddUserPage():
    AddUserPage = tk.Toplevel(root)
    AddUserPage.title("Add User")
    AddUserPage.iconbitmap(iconPath)
    AddUserPage.geometry("300x180")
    
    #selectfile=filedialog.askopenfilenames(initialdir="C:\\Users\\tmacovei001\\Desktop\\Speaker Recognition\\Datasets",title="Select wav files",filetypes=[("wav files","*.wav")])
    
    NameLabel=tk.Label(AddUserPage, width=20,text ="Name",anchor="w")
    NameEntry =tk.Entry(AddUserPage, width=20)
    
    ButtonSelectFiles=tk.Button(AddUserPage,text="Select audio files",width=15,height=1,command=AddUserFromFile)
    PhotoCheckButton=tk.Checkbutton(AddUserPage, text="Profile picture")
    ButtonAddPhoto=tk.Button(AddUserPage,text="Select file",width=10,height=1)
    ButtonUpdateModel=tk.Button(AddUserPage,text="Add user to model",width=20,height=2)
    
    NameLabel.place(x=0,y=10)
    NameEntry.place(x=50,y=10)
    
    ButtonSelectFiles.place(x=180,y=5)
    
    PhotoCheckButton.place(x=0,y=60)
    ButtonAddPhoto.place(x=5,y=90)
    
    
    
    ButtonUpdateModel.place(x=140,y=120)

    
    
    
    
    
def DeleteUserPage():
    DeleteUserPage = tk.Toplevel(root)
    DeleteUserPage.title("Delete User")
    DeleteUserPage.iconbitmap(iconPath)
    DeleteUserPage.geometry("260x150")
    
    labelDescription = tk.Label(DeleteUserPage, width=50,height=2,text ="Are you sure you want to delete this user? ",anchor="w")
    ButtonYes=tk.Button(DeleteUserPage,text="Yes",width=10,height=1)
    ButtonNo=tk.Button(DeleteUserPage,text="No",width=10,height=1,command=DeleteUserPage.destroy)
    labelDescription.place(x=20,y=30)        
    ButtonYes.place(x=10,y=100)
    ButtonNo.place(x=170,y=100)    
    
    
    

root=tk.Tk()
root.title("Main Menu")
root.iconbitmap(iconPath)
root.geometry("260x300")



ButtonTest=tk.Button(root,text="Test",width=20,height=1,command=OpenTestPage)
ButtonAddOrRemove=tk.Button(root,text="Add or Remove User",width=20,height=1,command=OpenAddOrRemovePage)
ButtonHelp=tk.Button(root,text="Help",width=20,height=1,command=OpenHelpPage)
ButtonExit=tk.Button(root,text="Exit",command=root.destroy,width=20,height=1)



ButtonTest.place(x=55,y=40)
ButtonAddOrRemove.place(x=55,y=80)
ButtonHelp.place(x=55,y=120)
ButtonExit.place(x=55,y=160)


root.mainloop()





