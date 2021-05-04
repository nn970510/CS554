# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 04:28:34 2021

@author: tc324
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn import model_selection,linear_model,metrics,ensemble,svm

def encode(ageofdriver,ageofv,speedlimit,vehicle_Type,lefthand,sex,datec,timec,rclass
	,rtype,lc,weather,rs,ps,ur,driverhome,jp,dow,pc):
	pdx=np.zeros(70)
	pdx[0]=ageofdriver
	pdx[1]=ageofv
	pdx[2]=speedlimit
	pdx[vehicle_Type+2]=1
	pdx[lefthand+8]=1
	pdx[sex+10]=1
	pdx[datec+12]=1
	pdx[timec+17]=1
	pdx[rclass+18]=1
	pdx[rtype+24]=1
	pdx[lc+29]=1
	pdx[weather+34]=1
	pdx[rs+38]=1
	pdx[ps+43]=1
	pdx[ur+47]=1
	pdx[driverhome+49]=1
	pdx[jp+51]=1
	pdx[dow+56]=1
	pdx[pc+64]=1
	return pdx


displaygen = ("male", "female")
optionsgen = list(range(len(displaygen)))

displayhand = ("right", "left")
optionshand = list(range(len(displayhand)))

displayur = ("Urban", "Rural")
optionsur = list(range(len(displayur)))

displaydh = ("Urban", "Rural")
optionsdh = list(range(len(displaydh)))

displayjp = ("Driver", "to work", "to school", "other")
optionsjp = list(range(len(displayjp)))

displayseason = ("spring", "summer","fall","winter")
optionsseason = list(range(len(displayseason)))

displaytime = ("night", "day")
optionstime = list(range(len(displaytime))) #not plus 1

displayrc = ("Mortorway","A(M)","A","B","C","Unclassified")
optionsrc = list(range(len(displayrc)))

displayrt = ("Roundabout","One way street","Dual Carriageway","Single Carriageway","Slip Road")
optionsrt = list(range(len(displayrt)))

displaylc = ("Daylight","Dark-lights lit","Dark-lights unlit","Dark-no light")
optionslc = list(range(len(displaylc)))

displayvt = ("Bicycle","Motorcycle","car", "Bus","Truck","Other" )
optionsvt = list(range(len(displayvt)))

displayweather=("Sunny","Rain","Snow","Other")
optionsweather=list(range(len(displayweather)))

displayrs = ("Dry","Wet","Snow", "frozen","flood")
optionsrs = list(range(len(displayrs)))

displayps = ("Petrol","Diesel/Heavy Oil","Electric","Other")
optionsps = list(range(len(displayps)))

disspeed=(20,30,40,50,60,70)
optionsspeed = list(range(len(disspeed)))

displaydow = ("Mon","Tue","Wed","Thur","Fri","Sat","Sun")
optionsdow = list(range(len(displaydow)))

displaypc = ("No corssing facilities","Zebra","non-junction pedestrian light crossing","Signal","Footbridge","Central Refuge")
optionspc = list(range(len(displaypc)))


st.title("Drivers Information")
ageofdriver = st.slider("your age",min_value=1,max_value=70)
sex = st.selectbox("gender of driver", optionsgen, format_func=lambda x: displaygen[x])+1
lefthand=st.selectbox("your strong hand", optionshand, format_func=lambda x: displayhand[x])+1
driverhome=st.selectbox("your home in urban or rural", optionsdh, format_func=lambda x: displaydh[x])+1
jp=st.selectbox("journal purpose", optionsjp, format_func=lambda x: displayjp[x])+1

st.title("Vehicle Information")
ageofv=st.slider("vehicle age",min_value=1,max_value=30)
vehicle_Type=st.selectbox("vehicle type", optionsvt, format_func=lambda x: displayvt[x])+1
ps=st.selectbox("vehicle fuel", optionsps, format_func=lambda x: displayps[x])+1

st.title("Genearal Information")
datec=st.selectbox("What is the season", optionsseason, format_func=lambda x: displayseason[x])+1
timec=st.selectbox("Day or Night", optionstime, format_func=lambda x: displaytime[x])
weather=st.selectbox("Weather", optionsweather, format_func=lambda x: displayweather[x])+1
dow=st.selectbox("What day is today", optionsdow, format_func=lambda x: displaydow[x])+1

st.title("Road Information")
speedlimit = st.selectbox("speed limit of road",disspeed)
rclass=st.selectbox("Road Class", optionsrc, format_func=lambda x: displayrc[x])+1
rtype=st.selectbox("Road Type", optionsrt, format_func=lambda x: displayrt[x])+1
lc=st.selectbox("Road light condition", optionslc, format_func=lambda x: displaylc[x])+1
rs=st.selectbox("Road Situation", optionsrs, format_func=lambda x: displayrs[x])+1
ur=st.selectbox("Drive in Urban or Rural", optionsur, format_func=lambda x: displayur[x])+1
pc=st.selectbox("Road facilities on road", optionspc, format_func=lambda x: displaypc[x])



if st.button('Predict'):
    if ageofdriver<25:
        ageofdriver=((ageofdriver-1)/5)+1
    else:
        ageofdriver=(ageofdriver+34)/10
    ageofdriver=int(ageofdriver)
    pdx=encode(ageofdriver,ageofv,speedlimit,vehicle_Type,lefthand,sex,datec,timec,rclass
	,rtype,lc,weather,rs,ps,ur,driverhome,jp,dow,pc)
    modelas=joblib.load("modelAS.pkl")
    modelcc=joblib.load("modelCC.pkl")
    modelfi=joblib.load("modelfi.pkl")
    modeljl=joblib.load("modeljl.pkl")
    
    pdjl=np.array(pdx[2])
    pdjl=np.append(pdjl,pdx[19:35])
    pdjl=np.append(pdjl,pdx[48:50])
    pdjl=np.append(pdjl,pdx[65:])

    pdfp=np.array(pdx[3:9])
    pdfp=np.append(pdfp,pdx[11:13])
    pdfp=np.append(pdfp,pdx[25:30])

    pdjl=pdjl.reshape(1,-1)
    pdfp=pdfp.reshape(1,-1)
    pdx=pdx.reshape(1,-1)
    yas=modelas.predict(pdx)
    ycc=modelcc.predict(pdx)
    print(pdjl)
    yjl=modeljl.predict(pdjl)
    yfp=modelfi.predict(pdfp)
    if yas==1:
        st.write("Be caution on the road!!! According to you information, it might be a serious accident if you have accident on the road")        
    else:
        st.write("you are not likely to meet a serious accident on this trip. Drive safe")
    if ycc==3:
        st.write("Beware of pedestrian when you drive!!!")
    else:
        st.write("You are not likely to meet too much pedestrian on this trip. Drive safe.")
    if yjl==0:
        st.write("You are not likely to have accident on junction. Drive safe.")
    else:
        st.write("Beware of the junction when you drive!!!")
    if yfp==1:
        st.write("Beware of your front!!!")
    elif yfp==2:
        st.write("Beware of your back!!!")
    elif yfp==3:
        st.write("Beware of your side!!!")
    else:
        st.write("You are not likely to have serious impect. Drive safe.")
    
    