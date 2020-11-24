import tkinter as tk
import pickle
import numpy as np
import re

from tkinter import messagebox
from sklearn.svm import SVC
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
#from model import *

bm = tk.Tk()
bm.title("Bank Marketing")
bm.geometry("600x400")

## FUNCIONS
##------------------
sc_X = StandardScaler()

global arr
arr = 0

global string 
string = "text"

global like 
like = 0

# clf = load('model1.joblib') 
# print('load model complete!')

# with open('sc1.pickle', 'rb') as f:
# 	scaler = pickle.load(f)

def ErorCallBack():
   messagebox.showinfo( "Sorry, an error occurred! ")


# def hienthi():
# 	greeting = funcion1()
# 	greeting_display = tk.Text(master = bm, height = 10 , width = 10)
# 	greeting_display.grid(column = 1 , row = 4)
# 	greeting_display.insert(tk.END, greeting)

def processing():
	# age
	x = str(entry1.get())
	global string
	string = x

	# Marial
	global arr
	if(b.get() == 1): 
		arr = 0
	if(b2.get() == 1): 
		arr = 1	
	if(b3.get() == 1): 
		arr = 2

	x2 = int(entry2.get())
	global like
	like = x2

	print("Input OK")
	print("--------")

	

def log():
	#arr_display = tk.Text(master = bm, height = 4 , width = 30)
	#arr_display.grid(columnspan = 3 , row = 59 )
	#arr_display.insert(tk.END, arr)
	print(arr)
	print(string)
	print(like)
	print("------------")


def prediction():
	if(arr ==1):
		count = len(string)
		display_kq.insert(tk.END, string)

	if(arr ==2):
		display_kq.insert(tk.END,"ec2")

	# x = np.array([arr])
	# x_sc = scaler.transform(x)
	# u = clf.predict(x_sc)

	#display_kq.insert(tk.END,string)
	#messagebox.showinfo("Model predict: ", string)
	#arr = []

def new():
	global arr
	global string
	global like 
	like = 0
	arr = 0
	string = "text"



#-------------------------------
# label
title = tk.Label(text = "Hello World!", font = ("Times New Roman", 11))
title.grid(column = 0, row = 0)

# #entry field
# entry1 = tk.Entry()
# entry1.grid(column = 1, row = 2)

# # button 
# button1 = tk.Button(text = "Click me!", command = hienthi)
# button1.grid(column = 1, row = 3)




## Text
#---------------------------------
title1 = tk.Label(text = "Input Text Here ", font = ("Time New Roman", 11))
title1.grid(column = 2, row = 6)
entry1 = tk.Entry()
entry1.grid(column = 3, row = 6)



## Model
#----------------------------------
title1 = tk.Label(text = "Model? ", font = ("Time New Roman", 11))
title1.grid(column = 2, row = 4)
b = tk.IntVar()
ckb1 = tk.Checkbutton(bm, text = "Model1", variable = b)
ckb1.grid(column = 3, row = 4)
b2 = tk.IntVar()
ckb2 = tk.Checkbutton(bm, text = "Model2", variable = b2)
ckb2.grid(column = 4, row = 4)
b3 = tk.IntVar()
ckb3 = tk.Checkbutton(bm, text = "Model3", variable = b3)
ckb3.grid(column = 5, row = 4)


### Like
##-------------------------------
title2 = tk.Label(text = "Like ", font = ("Time New Roman", 11))
title2.grid(column = 2, row = 7)
entry2 = tk.Entry()
entry2.grid(column = 3, row = 7)





title = tk.Label(text = "Ket qua ", font = ("Time New Roman", 11))
title.grid(column = 3, row = 9)
#### Hien thi ket qua
display_kq = tk.Text(master = bm, height = 2 , width = 10)
display_kq.grid(column = 3, row = 11)




##--------------------------------

B1 = tk.Button(bm, text ="Click me",height = 2, width = 10, command = processing)
B1.grid(column = 2, row = 8)


Bd = tk.Button(bm, text = 'Log', height = 2, width = 10, command = log )
Bd.grid(column = 3, row = 8)


pd = tk.Button(bm, text = 'Predict', height = 2, width = 10, command = prediction )
pd.grid(column = 4, row = 8)

new = tk.Button(bm, text = 'New', height = 2, width = 10, command = new)
new.grid(column = 5, row = 8)



# B = tk.Button(window, text ="Hello", command = helloCallBack)
# B.pack()

# text field
# textf1 = tk.Text(master = window, height = 10, width = 30)
# textf1.grid(column = 1, row = 5)

bm.mainloop()