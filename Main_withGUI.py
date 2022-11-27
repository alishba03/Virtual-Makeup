import cv2
from tkinter import*
from PIL import Image, ImageTk
import mediapipe as mp
from cv2 import cvtColor
import numpy as np

upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 408, 415, 272, 271, 268, 12, 38, 41, 42, 191, 78, 76]
lower_lip = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


#:::::::::::::::::::::::::::(functions)::::::::::::::::::::::::::::::::
# detection of 468 landmarks of face in the range (0,1)
def detect_landmarks(src):
    global rgb
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh= mp_face_mesh.FaceMesh() # facemesh object from facemesh solution
    rgb = cvtColor(src, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    return None #none type error when landmarks are not detected

# filtering detected 468 landmarks for lips and normalizing in the range of frame
def landmarks(landmarks, height, width,mask):
    lip_landmarks = np.array([(int(landmark.x * width), \
        int(landmark.y * height)) for landmark in landmarks])
    #landmark is not iterable in landmarks if landmarks are  nonetype.
    if mask:
        lip_landmarks = lip_landmarks[mask]
    return lip_landmarks
# Mask screen for lipstick application
def lip_mask(src, points , color):
    mask = np.zeros_like(src)  # Create a mask (array of zeros)
    mask = cv2.fillPoly(mask, [points], color)  # Mask for the lip color
    # Blurring the color
    mask = cv2.GaussianBlur(mask, (7, 7), 5)
    return mask
# function to apply lipstick
def apply_lipstick(src,clr="NO"):

    global mask
    ret_landmarks = detect_landmarks(src) #if returns none then in normalize_landmark func..
    height, width, _ = src.shape
    feature_landmarks = None
    feature_landmarks = landmarks(ret_landmarks, height, width, upper_lip+lower_lip)
    #lip color selection
    if clr=='orange':
        mask = lip_mask(src, feature_landmarks, [0, 143, 255])
    elif clr=='purple':
        mask = lip_mask(src, feature_landmarks, [255, 0, 0])
    elif clr=='NO':
        mask = lip_mask(src, feature_landmarks, [0, 0, 0])
    elif clr=='pink':
        mask = lip_mask(src, feature_landmarks, [153, 0, 157])
    elif clr=='green':
        mask = lip_mask(src, feature_landmarks, [0, 255, 0])
    elif clr=='berry':
        mask = lip_mask(src, feature_landmarks, [40, 0, 100])
    elif clr=='caramel':
        mask = lip_mask(src, feature_landmarks, [50, 70, 70])
    elif clr=='yellow':
        mask = lip_mask(src, feature_landmarks, [0,255,255])
    elif clr=='aqua':
        mask = lip_mask(src, feature_landmarks, [255,255,0])
    elif clr=='peach':
        mask = lip_mask(src, feature_landmarks, [35,35,139])
    elif clr=='red':
        mask = lip_mask(src, feature_landmarks, [2, 1, 159])
    # combining source and mask screens 
    src1 = cvtColor(src, cv2.COLOR_BGR2RGB)
    mask1 = cvtColor(mask, cv2.COLOR_BGR2RGB)
    output = cv2.addWeighted(src1, 1.0, mask1, 0.4, 0.0)
    return output

   
#--------------------------(tkinter Window)--------------------------------------
win = Tk()
win.geometry("1350x720+0+5") # dimensions of tkinter window
win.title("MakeUp-Lipsticks")
# tkinter variables
shade=StringVar()
shade.set("NO")
v_address=StringVar()
v_address.set("Videos/v1.mp4")
live_v=IntVar()
live_v.set(1)
# overlaying frame widget on tkinter window 
frame_1 = Frame(win, width=730, height=740, bg="#fe74a2").place(x=0, y=0)
frame_2 = Frame(win, width=685, height=740, bg="#f3f3f7").place(x=730, y=0)
var1 = IntVar()
# Title/logo Label and placement
T_image = Image.open("logo.png")
resize_Timage = T_image.resize((200, 200))
T_img = ImageTk.PhotoImage(resize_Timage)
T_label = Label(image=T_img)
T_label.image = T_img
T_label.place(x=1110,y=30)
N_label=Label(win,text="By A&H",font=("Times",15)).place(x=1170,y=220)
# Frame image and placement
F_image = Image.open("img6.jpg")
resize_Fimage = F_image.resize((730, 740))
F_img = ImageTk.PhotoImage(resize_Fimage)
F_label = Label(image=F_img)
F_label.image = F_img
F_label.place(x=0,y=0)
# Shade Click_Shade buttons and placement
Label(win,text="Pick your favorite Lip color",font=("Segoe Print",25),bg="pink").place(x=90,y=660)
Button(win,text='red',padx=20,pady=10,bg="red",fg="black",command=lambda:shade.set("red")).place(x=70,y=450)
Button(win,text='orange',padx=10,pady=10,bg="orange",fg="black",command=lambda:shade.set("orange")).place(x=170,y=450)
Button(win,text='aqua',padx=20,pady=10,bg="aqua",fg="black",command=lambda:shade.set("aqua")).place(x=280,y=450)
Button(win,text='purple',padx=20,pady=10,bg="purple",fg="black",command=lambda:shade.set("purple")).place(x=395,y=450)
Button(win,text='pink',padx=20,pady=10,bg="#fca7f2",fg="black",command=lambda:shade.set("pink")).place(x=520,y=450)
Button(win,text='green',padx=20,pady=10,bg="green",fg="black",command=lambda:shade.set("green")).place(x=240,y=540)
Button(win,text='yellow',padx=20,pady=10,bg="yellow",fg="black",command=lambda:shade.set("yellow")).place(x=365,y=540)

# Recorded videos switching and placement
Label(win,text="Select a Video",font=("Segoe Print",25),bg="#f3f3f7").place(x=800,y=180)
VIDEOS=[("Video 1","Videos/v1.mp4",820,240),
        ("Video 2","Videos/v2.mp4",820,280),
        ("Video 3","Videos/v3.mp4",930,240),
        ("Video 4","Videos/v4.mp4",930,280),
        ("Video 5","Videos/v5.mp4",870,320),
]
def video_click(address):
    global cap
    #Label(win,text=address).place(x=700,y=500) # To check adress
    if address==0:
        cap = cv2.VideoCapture(0)
        v_address.set(" ")
    else:
        v_address.set(address)
        cap = cv2.VideoCapture(v_address.get())
  
for name,address,x,y in VIDEOS:
    Radiobutton(win,text=name,variable=v_address,value=address).place(x=x,y=y)
Button(win,text="Select",bg="pink",command=lambda:video_click(v_address.get())).place(x=820,y=360)
# Live and Recorded video switching and placement
Label(win,text="Select Mode",font=("Segoe Print",25),bg="#f3f3f7").place(x=800,y=440)
Radiobutton(win,text="Live Stream",variable=live_v,value=0).place(x=820,y=500)
Button(win,text="Select",bg="pink",command=lambda:video_click(live_v.get())).place(x=820,y=550)


#.*.*.*.*.*.*.*.*.*.*.*.*  Main Program  .*.*.*.*.*.*.*.*.*.*.*.*

w = 700 # video width
h = 400 # video height
label1 = Label(frame_1, width=w, height=h) # video lable
label1.place(x=10, y=10) # video lable placement

cap = cv2.VideoCapture(v_address.get())
while True:
    ret,img = cap.read()
    if ret:
        img = cv2.resize(img, (w, h))
        img = cv2.flip(img, 1)# flip code = 1 _for horizontal
        output=apply_lipstick(img,shade.get())
        image = Image.fromarray(output)
        finalImage = ImageTk.PhotoImage(image=image)
        label1.configure(image=finalImage)
        label1.image = finalImage
    else:
        continue
    win.update()

#.*.*.*.*.*.*.*.*.*.*.*.*  End of Program  .*.*.*.*.*.*.*.*.*.*.*.*