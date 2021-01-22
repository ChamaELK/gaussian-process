import csv
import numpy as np
from matplotlib import pyplot
from collections import defaultdict

def ra_dec_data(n= 2945,filename = 'logs/cvastrophoto_guidelog_20200614T020302.txt', plot= True):

  inc=0
  GUIDING_SECTION=False
  DITHER_SECTION = False
  DITHERING = False
  #n= 2945
  m=2
  guiding_columns=[]
  time = np.zeros(n, dtype= float)
  guiding_data= np.zeros(shape= (n,m), dtype= float)
  dither = np.zeros(shape= n) 
  pulse= np.zeros(shape=(n,m),dtype= float)
  pulse_dir= np.zeros(shape=(n,m),dtype=str)
  dt=0
  
  with open(filename) as csvfile:
    pixelscale = None
    raguidespeed = None
    decguidespeed = None
    phd_reader= csv.reader(csvfile)
    for row in phd_reader:
      if row!= []:
        if GUIDING_SECTION and row[0].isdigit():
          time[inc] = row[22] 
          guiding_data[inc]= row[5:7]
          pulse[inc,0]=row[9]
          pulse_dir[inc,0]= row[10]
          pulse[inc,1]= row[11]
          pulse_dir[inc,1]= row[12]
          if DITHERING : 
            dither[inc] = 1
          inc+=1 
        if 'DITHER' in row[0]:
          DITHER_SECTION = True
        if 'Settling started' in row[0]:
          DITHERING = True
        if 'Settling complete' in row[0]:
          DITHERING = False
        if row[0]=='Frame':
          print("guiding section ... ")
          GUIDING_SECTION=True
          guiding_columns=row
        if "Pixel scale"  in row[0]:
            strpixelscale = row[0].split("Pixel scale =")[1].split("arc-sec/px")[0]
            if not strpixelscale.istitle():
                try:
                    pixelscale = float(strpixelscale)
                except:
                    print("error converting pixelscale to float :" + pixelscale)
        if "RA Guide Speed" in row[0] and "Dec Guide Speed" in row[1]:
            strraguidespeed = row[0].split("RA Guide Speed =")[1].split("a-s/s")[0]
            strdecguidespeed = row[1].split("Dec Guide Speed =")[1].split("a-s/s")[0]
            try:
                if not strraguidespeed.istitle():
                    raguidespeed = float(strraguidespeed)
                if not strdecguidespeed.istitle():
                    decguidespeed= float(strdecguidespeed)
            except: 
                print("error converting ra dec guide speed" + strraguidespeed + " " + strdecguidespeed)
  rasecperpixel = pixelscale/raguidespeed
  decsecperpixel = pixelscale/decguidespeed
  pew = pulse_dir[:,0]
  pns =  pulse_dir[:,1]
  ew = (pew == 'E') + (pew == 'W')* (-1)
  ns = (pns == 'N') + (pns == 'S')* (-1)
  pulse_ra = pulse[:,0]*ew
  pulse_dec = pulse[:,1]*n
  data_ra = rasecperpixel * guiding_data[:,0]
  data_dec = decsecperpixel * guiding_data[:,1]
  if plot: 
    pyplot.subplot(3,1,1)
    pyplot.plot(pulse_ra)
    pyplot.plot(pulse_dec)
    pyplot.subplot(3,1,2)
    pyplot.plot(time,10*dither)
    pyplot.plot(time,data_ra ,'ro')
    pyplot.subplot(3,1,3)
    pyplot.plot(time,10*dither)
    pyplot.plot(time,data_dec,'ro')
    pyplot.show()      

  return time, dither, pulse_ra, pulse_dec, data_ra, data_dec
def pulse_log(filename):
    HEADER = 9
    inc= 0
    keys =[]
    pulse_dict = defaultdict(list)
    with open(filename) as csvfile:
        pulse_reader= csv.reader(csvfile)
        for row in pulse_reader:
            if row!=[]:
                if inc == HEADER :
                    keys = row
                if inc > HEADER : 
                    for k,r in zip(keys,row):
                        pulse_dict[k].append(r)
            inc+=1
    time = np.array(pulse_dict["AbsTime"]).astype(float)
    raduration = np.array(pulse_dict["RADuration"]).astype(float)
    pew=  np.array(pulse_dict["RADirection"])
    decduration = np.array(pulse_dict["DECDuration"]).astype(float)
    pns =  np.array(pulse_dict["DECDirection"]) 
    ew,ns = directions(pew,pns)
    ra = np.multiply(raduration,ew)
    dec = np.multiply(decduration,ns)
    return pulse_dict, time, ra, dec

def directions(pew,pns):
    ew = (pew == 'E') + (pew == 'W')* (-1)
    ns = (pns == 'N') + (pns == 'S')* (-1)            
    return ew,ns

def ak(x,u):
  return x - u

def pulse_guide():
    guide_file = "logs/cvastrophoto_guidelog_20201030T043254.txt"       
    pulse_file = "logs/cvastrophoto_pulselog_20201030T043254.txt"
    guide_time, dither,  ur, ud, xr, xd = ra_dec_data(2854,guide_file,plot=False)
    pulse_dict , pulse_time, pr,pd = pulse_log(pulse_file)

    npulse= pulse_time.shape[0]
    nguide = guide_time.shape[0]
    n = npulse + nguide
    # time, pr, pd, xr, xd, dither
    _index = {"time": 0, "pr":1, "pd":2, "xr":3,  "xd":4, "dither":5 }  
    data = np.nan * np.ones(shape=(n,6))
    data[:,0] = np.concatenate([pulse_time,guide_time])
    data[:npulse,1] = pr
    data[:npulse,2] = pd
    data[npulse:n,3] = xr
    data[npulse:n,4] = xd
    data[npulse:n,5] = dither
    return _index, data[np.argsort(data[:, 0])]

