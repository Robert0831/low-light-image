import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from collections import defaultdict
import tkinter
from tkinter import *
import numpy as np
from numpy.linalg import inv
from PIL import Image, ImageTk
import time
import math
import warnings
from zoom import MainWindow
R, G, B = 0, 1, 2


def boxfilter(I, r):
    """Fast box filter implementation.
    Parameters
    ----------
    I:  a single channel/gray image data normalized to [0.0, 1.0]
    r:  window radius
    Return
    -----------
    The filtered image data.
    """
    M, N = I.shape
    dest = np.zeros((M, N))
    # print(I)

    # cumulative sum over Y axis (tate-houkou no wa)
    sumY = np.cumsum(I, axis=0)
    # print('sumY:{}'.format(sumY))
    # difference over Y axis
    dest[:r + 1] = sumY[r:2*r + 1]  # top r+1 lines
    dest[r + 1:M - r] = sumY[2*r + 1:] - sumY[:M - 2*r - 1]
    # print(sumY[2*r + 1:]) # from 2*r+1 to end lines
    # print(sumY[:M - 2*r - 1]) # same lines of above, from start
    # tile replicate sumY[-1] and line them up to match the shape of (r, 1)
    dest[-r:] = np.tile(sumY[-1], (r, 1)) - sumY[M - 2 *
                                                 r - 1:M - r - 1]  # bottom r lines

    # cumulative sum over X axis
    sumX = np.cumsum(dest, axis=1)
    # print('sumX:{}'.format(sumX))
    # difference over X axis
    dest[:, :r + 1] = sumX[:, r:2*r + 1]  # left r+1 columns
    dest[:, r + 1:N - r] = sumX[:, 2*r + 1:] - sumX[:, :N - 2*r - 1]
    dest[:, -r:] = np.tile(sumX[:, -1][:, None], (1, r)) - \
        sumX[:, N - 2*r - 1:N - r - 1]  # right r columns

    # print(dest)

    return dest



def guided_filter(I, p, r=15, eps=1e-3):
    """Refine a filter under the guidance of another (RGB) image.
    Parameters
    -----------
    I:   an M * N * 3 RGB image for guidance.
    p:   the M * N filter to be guided. transmission is used for this case.
    r:   the radius of the guidance
    eps: epsilon for the guided filter
    Return
    -----------
    The guided filter.
    """
    M, N = p.shape
    base = boxfilter(np.ones((M, N)), r)  # this is needed for regularization

    # each channel of I filtered with the mean filter. this is myu.
    means = [boxfilter(I[:, :, i], r) / base for i in range(3)]

    # p filtered with the mean filter
    mean_p = boxfilter(p, r) / base

    # filter I with p then filter it with the mean filter
    means_IP = [boxfilter(I[:, :, i]*p, r) / base for i in range(3)]

    # covariance of (I, p) in each local patch
    covIP = [means_IP[i] - means[i]*mean_p for i in range(3)]

    # variance of I in each local patch: the matrix Sigma in ECCV10 eq.14
    var = defaultdict(dict)
    for i, j in combinations_with_replacement(range(3), 2):
        var[i][j] = boxfilter(I[:, :, i]*I[:, :, j], r) / \
            base - means[i]*means[j]

    a = np.zeros((M, N, 3))
    for y, x in np.ndindex(M, N):
        #         rr, rg, rb
        # Sigma = rg, gg, gb
        #         rb, gb, bb
        Sigma = np.array([[var[R][R][y, x], var[R][G][y, x], var[R][B][y, x]],
                          [var[R][G][y, x], var[G][G][y, x], var[G][B][y, x]],
                          [var[R][B][y, x], var[G][B][y, x], var[B][B][y, x]]])
        cov = np.array([c[y, x] for c in covIP])
        a[y, x] = np.dot(cov, inv(Sigma + eps*np.eye(3)))  # eq 14

    # ECCV10 eq.15
    b = mean_p - a[:, :, R]*means[R] - \
        a[:, :, G]*means[G] - a[:, :, B]*means[B]

    # ECCV10 eq.16
    q = (boxfilter(a[:, :, R], r)*I[:, :, R] + boxfilter(a[:, :, G], r) *
         I[:, :, G] + boxfilter(a[:, :, B], r)*I[:, :, B] + boxfilter(b, r)) / base

    return q


def get_illumination_channel(I, w):
    M, N, _ = I.shape
    # padding for channels
    padded = np.pad(
        I, ((int(w/2), int(w/2)), (int(w/2), int(w/2)), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    brightch = np.zeros((M, N))

    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :]
                              )  # dark channel rgb中最小的
        brightch[i, j] = np.max(padded[i:i + w, j:j + w, :])  # bright channel  
    return darkch, brightch


def get_atmosphere(I, brightch, p=0.1):
    M, N = brightch.shape
    flatI = I.reshape(M*N, 3)  # reshaping image array
    flatbright = brightch.ravel()  # flattening image array

    searchidx = (-flatbright).argsort()[:int(M*N*p)]  # sorting and slicing
    A = np.mean(flatI.take(searchidx, axis=0), dtype=np.float64, axis=0)
    return A


def get_initial_transmission(A, brightch):
    A_c = np.max(A)
    init_t = (brightch-A_c)/(1.-A_c)  # finding initial transmission map
    # normalized initial transmission map
    return (init_t - np.min(init_t))/(np.max(init_t) - np.min(init_t))


def get_corrected_transmission(I, A, darkch, brightch, init_t, alpha, omega, w):
    im = np.empty(I.shape, I.dtype)
    for ind in range(0, 3):
        # divide pixel values by atmospheric light
        im[:, :, ind] = I[:, :, ind] / A[ind]
    dark_c, _ = get_illumination_channel(
        im, w)  # dark channel transmission map
    dark_t = 1 - omega*dark_c  # corrected dark transmission map
    # initializing corrected transmission map with initial transmission map
    corrected_t = init_t
    diffch = brightch - darkch  # difference between transmission maps

    for i in range(diffch.shape[0]):
        for j in range(diffch.shape[1]):
            if (diffch[i, j] < alpha):
                corrected_t[i, j] = dark_t[i, j] * init_t[i, j]
    return np.abs(corrected_t)


def get_final_image(I, A, refined_t, tmin):
    # duplicating the channel of 2D refined map to 3 channels
    refined_t_broadcasted = np.broadcast_to(
        refined_t[:, :, None], (refined_t.shape[0], refined_t.shape[1], 3))
    J = (I-A) / (np.where(refined_t_broadcasted < tmin,
                          tmin, refined_t_broadcasted)) + A  # finding result

    return (J - np.min(J))/(np.max(J) - np.min(J))  # normalized image


def reduce_init_t(init_t):
    init_t = (init_t*255).astype(np.uint8)
    xp = [0, 32, 255]
    fp = [0, 32, 48]
    x = np.arange(256)  # creating array [0,...,255]
    # interpreting fp according to xp in range of x
    table = np.interp(x, xp, fp).astype('uint8')
    init_t = cv2.LUT(init_t, table)  # lookup table
    init_t = init_t.astype(np.float64)/255  # normalizing the transmission map
    return init_t


def dehaze(I, tmin=0.1, w=15, alpha=0.4, omega=0.75, p=0.1, eps=1e-3, reduce=True):
    I = np.asarray(I, dtype=np.float64)  # Convert the input to a float array.
    I = I[:, :, :3] / 255
    m, n, _ = I.shape
    Idark, Ibright = get_illumination_channel(I, w)
    A = get_atmosphere(I, Ibright, p)

    init_t = get_initial_transmission(A, Ibright)
    if reduce:
        init_t = reduce_init_t(init_t)
    corrected_t = get_corrected_transmission(
        I, A, Idark, Ibright, init_t, alpha, omega, w)

    normI = (I - I.min()) / (I.max() - I.min())
    refined_t = guided_filter(normI, corrected_t, w,
                              eps)  # applying guided filter
    J_refined = get_final_image(I, A, refined_t, tmin)

    enhanced = (J_refined*255).astype(np.uint8)
    f_enhanced = enhanced
    f_enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
    f_enhanced = cv2.edgePreservingFilter(f_enhanced, flags=1, sigma_s=64, sigma_r=0.2)
    return f_enhanced



global tt
global gett
root = Tk()
root.geometry("500x500")
root.title('Image ')
mystring = StringVar(root)
# Function for closing window


def Close():
    global tt
    global gett
    tt = os.path.dirname(os.path.realpath(__file__))
    # tt='C:/imagefinal/'
    tt = tt+'/'+str(mystring.get())
    gett=str(mystring.get())
    gett=gett[:-4]
    root.destroy()


# Button for closing
la = Label(root, text="Image to get")
la.pack(pady=20)
en = Entry(root, textvariable=mystring)
en.pack(pady=20)
exit_button = Button(root, text="Confirm", command=Close)
exit_button.pack(pady=20)

root.mainloop()


img = cv2.imread(tt)
global wei
global hei
hei, wei, _ = img.shape


global xxx
global yyy
global tmp
global res
global key
global li
global co
global show
global las
global cout
cout=0
show=0
li=0
co=0
key=0
w1 = Tk()

def motion(event):
    xx, yy = event.x, event.y
    label2 = Label(frame1, text=f" X:{xx} ,Y:{yy} ",background='#FFFFE4',font=("Arial"))
    label2.grid(row=1, column=0)

#readimage########################################
w1.title('Image Process')
w1.config(bg='#FFFFE4')
w1.geometry('1024x550')
w1.resizable(width=False, height=False)
#img1 = cv2.imread(tt)
tmp = np.array(img, dtype=np.int8)
#res=np.asarray(img, dtype=np.int8)
res=np.array(img)
res=res.astype(np.uint8)
las=[]
a=res.copy()
a=a.astype(np.uint8)
las.append(a)
img1 = cv2.resize(img, (512, 512))
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
tk_img = ImageTk.PhotoImage(Image.fromarray(img1))
##################################################
def sshow():
    global gett
    global show
    if show==0:
        path=tt
    else:
        path=os.path.dirname(os.path.realpath(__file__))
        path=os.path.join(path,gett+"-temp.gif")
    app = MainWindow(Toplevel(), path=path)
    app.mainloop()


def illu():
    global res
    global key
    global las
    global cout
    global show
    show=1
    key=0
    for i in range(cout+1,len(las)):
        las.pop()
    cout+=1
    res=dehaze(res)
    stmp=cv2.resize(res,(512,512))
    stmp = cv2.cvtColor(stmp, cv2.COLOR_RGB2BGR)
    stmp=ImageTk.PhotoImage(Image.fromarray(stmp))
    label1.configure(image=stmp)
    label1.image=stmp

    pre= cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    pre=Image.fromarray(pre)
    pre.save(gett+"-temp.gif")
    las.append((res.copy()).astype(np.uint8))

frame1 = Frame(w1,bg='#FFFFE4')
frame1.grid(row=0, column=0)
global label1
label1 = Label(frame1, image=tk_img, width=512, height=512, anchor='nw')

label1.grid(row=0, column=0)
label1.bind('<Motion>', motion)

r2 = Frame(w1, width=512, height=512, padx=0, pady=0,bg='#FFFFE4')
r2.grid(row=0, column=1)

b1=Button(r2,text='image illuminate',command=illu, font=("Arial", 14, "bold"), padx=5, pady=5, bg="yellow")
b1.place(x=10, y=10)

b2=Button(r2,text='show image',command=sshow, font=("Arial", 14, "bold"), padx=5, pady=5, bg="yellow")
b2.place(x=350, y=10)

def deep():
    global res
    global key
    global las
    global cout
    global gett
    global show
    show=1
    key=0
    for i in range(cout+1,len(las)):
        las.pop()
    cout+=1
    res=res.astype(np.uint8)
    pre= cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    pre=Image.fromarray(pre)
    pre.save('./datasets/'+"temp.jpg")

    datasets='datasets'
    oout='datasets'
    ww='./HWMNet-main/model/LOL_enhancement_HWMNet.pth'
    # datasets='C:/imagefinal/datasets'
    # oout='C:/imagefinal'
    # ww='C:/imagefinal/HWMNet-main/model/LOL_enhancement_HWMNet.pth'
    os.system('C:/Users/...../python ./HWMNet-main/demo.py --input_dir "%s" --result_dir "%s" --weights "%s"' %(datasets,oout,ww))       
    path=os.path.dirname(os.path.realpath(__file__))
    path=path+'/datasets/'+'temp.jpg'

    img = cv2.imread(path)
    res=np.asarray(img)
    res=res.astype(np.uint8)
    pre= cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    pre=Image.fromarray(pre)
    pre.save(gett+"-temp.gif")
    
    stmp=cv2.resize(img,(512,512))
    stmp = cv2.cvtColor(stmp, cv2.COLOR_RGB2BGR)
    stmp=ImageTk.PhotoImage(Image.fromarray(stmp))
    label1.configure(image=stmp)
    label1.image=stmp
    las.append((res.copy()).astype(np.uint8))

    

b6=Button(r2,text="Deepway",font=("Arial", 14, "bold"),padx=5, pady=5,background="yellow",command=deep)
b6.place(x=220, y=10)

#light###################################
l2 = Label(r2, text='Light', font=("Arial", 10, "bold"), padx=5, pady=5, bg="yellow")
l2.place(x=130, y=80)
def light(value):
    global tmp
    global res
    global li
    global co
    global key
    global show
    global  las
    global cout
    for i in range(cout+1,len(las)):
        las.pop()
    cout+=1
    show=1
    if key!=1:
        tmp=res
    li=int(value)
    value=int(value)
    res = np.asarray(tmp, dtype=np.uint8)
    #constrast
    if co!=0.0 and key==1:
        res= cv2.cvtColor(res, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(res)
        clahe = cv2.createCLAHE(clipLimit=float(co), tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        res = cv2.merge((cl,a,b))
        res= cv2.cvtColor(res, cv2.COLOR_LAB2BGR)
    key=1
    #light
    res = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(res)
    v = cv2.add(v,int(value*2))
    v[v > 255] = 255
    v[v < 0] = 0
    res = cv2.merge((h, s, v))
    res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    stmp=cv2.resize(res,(512,512))
    stmp = cv2.cvtColor(stmp, cv2.COLOR_RGB2BGR)
    stmp=ImageTk.PhotoImage(Image.fromarray(stmp))

    pre= cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    pre=Image.fromarray(pre)
    pre.save(gett+"-temp.gif")
    '''
    aa=os.path.dirname(os.path.realpath(__file__))
    aa=aa+'/qw.jpg'
    asd=cv2.imread(aa)
    #asd = np.asarray(asd, dtype=np.float64)
    asd=cv2.resize(asd,(512,512))
    asd = cv2.cvtColor(asd, cv2.COLOR_RGB2BGR)
    asd=ImageTk.PhotoImage(Image.fromarray(asd))
    '''   
    label1.configure(image=stmp)
    label1.image=stmp
    las.append((res.copy()).astype(np.uint8))

scale1 = Scale(r2, from_=-100, to=100, font=("Arial", 14, "bold"),
              width=15, orient=HORIZONTAL, length=200, command=light,background='white')
scale1.place(x=200, y=60)
#########################################
#contrast######################################
def contrast(value):
    global tmp
    global res
    global li
    global co
    global key
    global show
    global las
    global cout
    for i in range(cout+1,len(las)):
        las.pop()
    cout+=1
    show=1
    if key!=1:
        tmp=res

    co=float(value)

    res = np.asarray(tmp, dtype=np.uint8)
    #contrast  Contrast Limiting Adaptive Histogram Equalization (CLAHE)
    if co!=0.0:
        res= cv2.cvtColor(res, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(res)
        clahe = cv2.createCLAHE(clipLimit=float(value), tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        res = cv2.merge((cl,a,b))
        res= cv2.cvtColor(res, cv2.COLOR_LAB2BGR)
    #light
    if li!=0.0 and key==1:
        res = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(res)
        v = cv2.add(v,int(li*2))
        v[v > 255] = 255
        v[v < 0] = 0
        res = cv2.merge((h, s, v))
        res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    key=1

    pre= cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    pre=Image.fromarray(pre)
    pre.save(gett+"-temp.gif")

    #res = cv2.convertScaleAbs(res, alpha=float(value), beta=0)
    stmp=cv2.resize(res,(512,512))
    stmp = cv2.cvtColor(stmp, cv2.COLOR_RGB2BGR)
    stmp=ImageTk.PhotoImage(Image.fromarray(stmp))
    label1.configure(image=stmp)
    label1.image=stmp
    las.append((res.copy()).astype(np.uint8))

l3 = Label(r2, text='Contrast', font=("Arial", 10, "bold"), padx=5, pady=5, bg="yellow")
l3.place(x=110, y=150)
scale2 = Scale(r2, from_=0.0, to=40.0,resolution=0.1, font=("Arial", 14, "bold"),
              width=15, orient=HORIZONTAL, length=200, command=contrast,background='white')
scale2.place(x=200, y=130)
###############################################

#local###########################################################################################
##################################################################################################
l4 = Label(r2, text='- - -Local \t Adjustment- - -', font=("Arial", 14, "bold"), padx=5, pady=5,background='#FFFFE4' )
l4.place(x=180, y=190)
r22=Frame(w1, width=300, height=150, padx=0, pady=0,background="#EEEED5")  
r22.place(x=650, y=240)
global e1
global e2
def motionxy(event):
    global xxx
    global yyy
    global e1
    global e2
    xxx, yyy = event.x, event.y
    e1.after(100, e1.destroy())
    e2.after(100, e2.destroy())
    e1=Label(r22,text=f"X:{xxx}", font=("Arial", 10, "bold"), padx=5, pady=5, bg="white")
    e1.place(x=115, y=5)
    e2=Label(r22,text=f"Y:{yyy}", font=("Arial", 10, "bold"), padx=5, pady=5, bg="white")
    e2.place(x=170, y=5)

e1=Label(r22,text="X:0", font=("Arial", 10, "bold"), padx=5, pady=5, bg="white")
e1.place(x=115, y=5)
e2=Label(r22,text="Y:0", font=("Arial", 10, "bold"), padx=5, pady=5, bg="white")
e2.place(x=170, y=5)
label1.bind('<Button-1>', motionxy)

global which
which=1
var =StringVar()
def check():
    global which
    which=int(var.get())
def sharpen(img,sigma=50):
    blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    return usm

def loctran():
    global key,show,xxx,yyy,which,wei,hei
    global res,las,cout
    for i in range(cout+1,len(las)):
        las.pop()
    cout+=1
    show=1
    key=0
    ix=wei/512
    iy=hei/512
    r=min(ix,iy)

    if r>1:
        r=int(r*4)
    elif r>2:
        r=int(r*8)
    if r<1:
        r=1/r
        r=int(r/2)
    if r<3:
        r=3
    ix=int(xxx*ix)
    iy=int(yyy*iy)
    lx=ix-r
    ux=ix+r
    ly=iy-r
    uy=iy+r
    if lx<0:
        lx=0
    if ux>wei:
        ux=wei-1
    if ly<0:
        ly=0
    if uy>hei:
        uy=hei
    if which==1:
        ct=np.array(res[ly:int(uy+1),lx:int(ux+1),...],dtype=np.uint8)
        ct=cv2.blur(ct,(5,5))
        res[ly:int(uy+1),lx:int(ux+1),...]=ct
        res=res.astype(np.uint8)
    if which==2:
        ct=np.array(res[ly:int(uy+1),lx:int(ux+1),...],dtype=np.uint8)
        #ct=cv2.edgePreservingFilter(ct, flags=1, sigma_s=8, sigma_r=0.2)
        ct=sharpen(ct)
        res[ly:int(uy+1),lx:int(ux+1),...]=ct
        res=res.astype(np.uint8)
    if which==3:
        now=res[iy,ix]
        ct=np.array(res[ly:int(uy+1),lx:int(ux+1),...],dtype=np.uint8)
        ct=now
        res[ly:int(uy+1),lx:int(ux+1),...]=ct
        res=res.astype(np.uint8)

    stmp=cv2.resize(res,(512,512))
    stmp = cv2.cvtColor(stmp, cv2.COLOR_RGB2BGR)
    stmp=ImageTk.PhotoImage(Image.fromarray(stmp))
    label1.configure(image=stmp)
    label1.image=stmp
    pre= cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    pre=Image.fromarray(pre)
    pre.save(gett+"-temp.gif")
    las.append((res.copy()).astype(np.uint8))


ch1 = Radiobutton(r22,text="Blur", value=1,variable=var,  command=check,background='#EEEED5')
ch2 = Radiobutton(r22,text="Sharpen", value=2,variable=var, command=check,background='#EEEED5')
ch3 =Radiobutton(r22,text="Painting", value=3, variable=var, command=check,background='#EEEED5')
ch1.place(x=80, y=40)
ch1.select()
ch2.place(x=80, y=60)
ch3.place(x=80, y=80)
b3=Button(r22,text="confirm",font=("Arial", 10, "bold"),background="white",command=loctran)
b3.place(x=180, y=60)

def sharp():
    global key,show
    global res,las,cout
    for i in range(cout+1,len(las)):
        las.pop()
    cout+=1
    show=1
    key=0
    res = cv2.detailEnhance(res, sigma_s=10, sigma_r=0.15)
    res = cv2.edgePreservingFilter(res, flags=1, sigma_s=64, sigma_r=0.2)

    res=res.astype(np.uint8)
    stmp=cv2.resize(res,(512,512))
    stmp = cv2.cvtColor(stmp, cv2.COLOR_RGB2BGR)
    stmp=ImageTk.PhotoImage(Image.fromarray(stmp))
    label1.configure(image=stmp)
    label1.image=stmp
    pre= cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    pre=Image.fromarray(pre)
    pre.save(gett+"-temp.gif")
    las.append((res.copy()).astype(np.uint8))

def undo():
    global las
    global res
    global cout
    if cout>0:
        cout-=1
        res= np.array(las[int(cout)], dtype=np.uint8)
        stmp=cv2.resize(res,(512,512))
        stmp = cv2.cvtColor(stmp, cv2.COLOR_RGB2BGR)
        stmp=ImageTk.PhotoImage(Image.fromarray(stmp))
        label1.configure(image=stmp)
        label1.image=stmp
        pre= cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        pre=Image.fromarray(pre)
        pre.save(gett+"-temp.gif")

def redo():
    global las
    global res
    global cout
    if cout+1<len(las):
        cout+=1
        res= np.array(las[cout], dtype=np.uint8)
        stmp=cv2.resize(res,(512,512))
        stmp = cv2.cvtColor(stmp, cv2.COLOR_RGB2BGR)
        stmp=ImageTk.PhotoImage(Image.fromarray(stmp))
        label1.configure(image=stmp)
        label1.image=stmp
        pre= cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        pre=Image.fromarray(pre)
        pre.save(gett+"-temp.gif")

b4=Button(r2,text="undo",font=("Arial", 10, "bold"),background="yellow",command=undo)
b4.place(x=160, y=400)
b5=Button(r2,text="redo",font=("Arial", 10, "bold"),background="yellow",command=redo)
b5.place(x=210, y=400)
b7=Button(r2,text="End",font=("Arial", 20, "bold"),background="yellow",command=w1.destroy)
b7.place(x=380, y=460)
b8=Button(r2,text="Sharpen",font=("Arial", 14, "bold"),background="yellow",command=sharp)
b8.place(x=350, y=400)

w1.mainloop()

