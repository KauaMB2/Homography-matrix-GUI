import tkinter as tk
from tkinter import *
import os
from tkinter import messagebox
from tkinter import font
import cv2
import numpy as np
import pandas as pd

df=None
CSVAlreadyReaded=False
columnNamesList=[""]
selectModelMenu=None
DIR = os.path.dirname(os.path.abspath(__file__))
inputEntries=[]

percentageList=["50%", "55%", "60%", "65%", "70%", "75%", "80%", "85%", "90%"]

def predictTargets():
    global inputEntries, DIR, df
    if df is None:
        messagebox.showerror("ERROR", "Firstly, you need to read a valid table.")
        return
    inputData = []
    for entry in inputEntries:  # Collect and validate the input values
        value = entry.get()
        if not value.isdigit():  # Check if the input is a valid integer
            messagebox.showwarning("WARNING", "The inputs are not in the correct format.")
            return
        inputData.append(int(value))  # Convert to integer and add to input_data
    # Create the input point for perspective transform
    ponto_original = np.array([inputData], dtype=np.float32).reshape(-1, 1, 2)

    # Ensure that homographyMatrix is defined and has the correct shape (3, 3)
    if homographyMatrix is None or homographyMatrix.shape != (3, 3):
        messagebox.showerror("ERROR", "Homography matrix is not defined or has incorrect shape.")
        return

    # Transform the point using the homography matrix
    ponto_estimado = cv2.perspectiveTransform(ponto_original, homographyMatrix)
    inputList=list(inputListBox.get(0, END))
    # Extrair pos_x e pos_y estimados
    #pos_x_est = ponto_estimado[0][0][0]
    #pos_y_est = ponto_estimado[0][0][1]
    predictionsMessage=""
    for i in range(len(ponto_estimado[0][0])):
        predictionsMessage+=f"{inputList[i]}: {ponto_estimado[0][0][i]}\n"
    print(predictionsMessage)
    messagebox.showinfo("RESULT", predictionsMessage)

def readCSVFile():
    global columnNamesList, df, CSVAlreadyReaded
    fileName=inputName.get()
    if fileName=="":
        messagebox.showerror("ERROR", "Please, firstly inform the file name.")
        return
    try:
        df=pd.read_csv(f"{fileName}.csv", low_memory=False)
        df[['x', 'y', 'pos_x', 'pos_y']] = df[['x', 'y', 'pos_x', 'pos_y']].apply(pd.to_numeric, errors='coerce')
    except:
        messagebox.showerror("Error", "It wasn't possible read the file. Please, be sure the file exist.")
        return
    columnNamesList = df.columns
    inputColumnNamesMenu['menu'].delete(0, END)  # Clear the current menu
    for column in columnNamesList:  # Add new column names
        inputColumnNamesMenu['menu'].add_command(label=column, command=lambda value=column: currentInputOption.set(value))
    outputColumnNamesMenu['menu'].delete(0, END)  # Clear the current menu
    for column in columnNamesList:  # Add new column names
        outputColumnNamesMenu['menu'].add_command(label=column, command=lambda value=column: currentOutputOption.set(value))
    if CSVAlreadyReaded:
        inputListBox.delete(0, END)
        outputListBox.delete(0, END)
        currentInputOption.set("")
        currentOutputOption.set("")
    CSVAlreadyReaded=True

def calculateMatrix():
    global df, homographyMatrix, originPoints, inputEntries, outputColumnNamesMenu
    if df is None:
        messagebox.showerror("ERROR", "Firstly, you need to read a valid table.")
        return
    outputList=list(outputListBox.get(0, END))
    inputList=list(inputListBox.get(0, END))
    if len(outputList)!=2 or len(inputList)!=2:
        messagebox.showerror("ERROR", "Exactly two features and two targets must be selected.")
        return
    # Extrair pontos de origem e destino, removendo NaN
    originPoints = df[inputList].dropna().values.astype(np.float32)
    targetPoints = df[outputList].dropna().values.astype(np.float32)
    homographyMatrix, status = cv2.findHomography(originPoints, targetPoints, cv2.RANSAC)
    if homographyMatrix is None:
        messagebox.showerror("ERROR", "It was not possible to calculate the homography matrix.")
        return
    for i in range(len(inputList)):# Create new entries
        predictLabel = tk.Label(root, text=f"{inputList[i]} value:", font=("calibri", 9), bg=PURPLE_COLOR, fg=WHITE_COLOR)
        predictLabel.place(x=(60 * i + 10), y=265)
        entry = tk.Entry(root, width=5)
        entry.place(x=(60 * i + 15), y=280)
        inputEntries.append(entry)  # Store the entry reference
    predictButton = Button(root, text="Predict", command=predictTargets, bg=PURPLE_COLOR, fg=WHITE_COLOR)# Create the Predict button
    predictButton.place(x=330, y=310)
    
def updateColumnNameList():
    global columnNamesList
    inputColumnNamesMenu['menu'].delete(0, END)  # Clear the current menu
    outputColumnNamesMenu['menu'].delete(0, END)  # Clear the current menu
    for column in columnNamesList:  # Add new column names
        inputColumnNamesMenu['menu'].add_command(label=column, command=lambda value=column: currentInputOption.set(value))
        outputColumnNamesMenu['menu'].add_command(label=column, command=lambda value=column: currentOutputOption.set(value))

def addToInputListBox():
    global columnNamesList
    item = currentInputOption.get()
    if item=="":
        messagebox.showwarning("WARNING", "The input option is not selected. Please, firstly select the input option.")
        return
    if item in inputListBox.get(0, END):  # Verifica se o item já está no Listbox
        messagebox.showwarning("WARNING", f"The item '{item}' was already added.")
    else:
        inputListBox.insert(END, item)
        currentInputOption.set("")
        if type(columnNamesList)!=list:
            columnNamesList = columnNamesList.tolist()
        columnNamesList.remove(item)
        updateColumnNameList()

def addToOutputListBox():
    global columnNamesList
    item = currentOutputOption.get()
    if item=="":
        messagebox.showwarning("WARNING", "The output option is not selected. Please, firstly select the output option.")
        return
    if item in outputListBox.get(0, END):# Verifica se o item já está no Listbox
        messagebox.showwarning("WARNING", f"O item '{item}' já foi adicionado.")
    else:
        outputListBox.insert(END, item)
        currentOutputOption.set("")
        if type(columnNamesList)!=list:
            columnNamesList = columnNamesList.tolist()
        columnNamesList.remove(item)
        updateColumnNameList()

def removeFromInputListBox():# Function to remove selected feature
    global columnNamesList
    # Get the index of the selected item
    selectedIndex = inputListBox.curselection()
    if (len(selectedIndex)==0):
        messagebox.showwarning("WARNING", "Please, select a feature to be removed.")
        return
    selectedValue=inputListBox.get(selectedIndex[0])
    if selectedIndex:  # Check if an item is selected
        inputListBox.delete(selectedIndex)
        if type(columnNamesList)!=list:
            columnNamesList = columnNamesList.tolist()
        columnNamesList.append(selectedValue)
        updateColumnNameList()
    else:
        messagebox.showinfo("INFO", "Before remove a feature, you must select the feature clicking on it")

def removeFromOutputListBox():# Function to remove selected output
    global columnNamesList
    selectedIndex = outputListBox.curselection()# Get the index of the selected item
    if (len(selectedIndex)==0):
        messagebox.showwarning("WARNING", "Please, select a target to be removed.")
        return
    selectedValue=outputListBox.get(selectedIndex[0])
    if selectedIndex:  # Check if an item is selected
        outputListBox.delete(selectedIndex)
        if type(columnNamesList)!=list:
            columnNamesList = columnNamesList.tolist()
        columnNamesList.append(selectedValue)
        updateColumnNameList()
    else:
        messagebox.showinfo("INFO", "Before remove an output, you must select the output clicking on it")

DIR=os.path.dirname(os.path.abspath(__file__))
PURPLE_COLOR="#6a0dad"
WHITE_COLOR="white"
RED_COLOR="red"

root = tk.Tk()
img = PhotoImage(file=os.path.join(os.path.dirname(__file__), 'icons', 'icon.png'))
root.iconphoto(False, img)
root.title("AutoML")
root.geometry("400x350")
root.configure(bg=PURPLE_COLOR)  # Cor de fundo roxo

currentInputOption=StringVar()
currentInputOption.set(columnNamesList[0])
currentOutputOption=StringVar()
currentOutputOption.set(columnNamesList[0])
currentPercentageOption=StringVar()
currentPercentageOption.set(percentageList[0]) 

# Block resizing
root.resizable(False, False)  # (width, height)

addOutputButton = Button(root, text="Read CSV file", command=readCSVFile, bg=PURPLE_COLOR, fg=WHITE_COLOR)
addOutputButton.place(x=158, y=60)

trainModelButton = Button(root, text="CALCULATE MATRIX", command=calculateMatrix, bg=PURPLE_COLOR, fg=RED_COLOR, font=font.Font(weight="bold", size=10))
trainModelButton.place(x=130, y=95)

fileNameLabel = tk.Label(root, text="Type the file name(csv): ", font=("calibri", 12), bg=PURPLE_COLOR, fg="white")
fileNameLabel.pack(side=tk.TOP)

inputName = tk.Entry(root, width=40)
inputName.pack(side=tk.TOP)

inputFeaturesLabel = tk.Label(root, text="Select the\ninput features: ", font=("calibri", 9), bg=PURPLE_COLOR, fg="white")
inputFeaturesLabel.place(x=10, y=60)
inputColumnNamesMenu = OptionMenu(root, currentInputOption, *columnNamesList)
inputColumnNamesMenu.config(bg=PURPLE_COLOR, fg=WHITE_COLOR, highlightbackground=PURPLE_COLOR, highlightcolor=PURPLE_COLOR, width=5)
inputColumnNamesMenu.place(x=10, y=90)

inputListLabel = tk.Label(root, text="Inputs list: ", font=("calibri", 9), bg=PURPLE_COLOR, fg="white")
inputListLabel.place(x=10, y=125)

inputListBox = Listbox(root, bg=WHITE_COLOR, fg=PURPLE_COLOR, height=3, width=12)
inputListBox.place(x=10, y=145)

addInputButton = Button(root, text="Add feature", command=addToInputListBox, bg=PURPLE_COLOR, fg=WHITE_COLOR)
addInputButton.place(x=10, y=205)

addInputButton = Button(root, text="Remove feature", command=removeFromInputListBox, bg=PURPLE_COLOR, fg=WHITE_COLOR)
addInputButton.place(x=10, y=235)

outputLabel = tk.Label(root, text="Select the output\n target: ", font=("calibri", 9), bg=PURPLE_COLOR, fg="white")
outputLabel.place(x=295, y=60)
outputColumnNamesMenu = OptionMenu(root, currentOutputOption, *columnNamesList)
outputColumnNamesMenu.config(bg=PURPLE_COLOR, fg=WHITE_COLOR, highlightbackground=PURPLE_COLOR, highlightcolor=PURPLE_COLOR, width=5)
outputColumnNamesMenu.place(x=295, y=90)

outputListLabel = tk.Label(root, text="Outputs list: ", font=("calibri", 9), bg=PURPLE_COLOR, fg="white")
outputListLabel.place(x=295, y=125)

outputListBox = Listbox(root, bg=WHITE_COLOR, fg=PURPLE_COLOR, height=3, width=12)
outputListBox.place(x=295, y=145)

addOutputButton = Button(root, text="Add output", command=addToOutputListBox, bg=PURPLE_COLOR, fg=WHITE_COLOR)
addOutputButton.place(x=295, y=205)

addOutputButton = Button(root, text="Remove output", command=removeFromOutputListBox, bg=PURPLE_COLOR, fg=WHITE_COLOR)
addOutputButton.place(x=295, y=235)

root.mainloop()