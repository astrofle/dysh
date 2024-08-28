#!/usr/bin/env python3

"""
Description:
------------
This is a script developed in spyder to develop nodding.

"""

import numpy as np
import numpy.ma as ma
from scipy.stats import norm
import astropy.units as u
from astropy.io import fits

import matplotlib.pyplot as plt
from dysh.fits.sdfitsload import SDFITSLoad
from dysh.fits.gbtfitsload import GBTFITSLoad
import dysh.util as util
from dysh.util.files import dysh_data
from dysh.util.selection import Selection

#  useful keys for a mult-beam observation
k=['DATE-OBS','SCAN', 'IFNUM', 'PLNUM', 'FDNUM', 'INTNUM', 'PROCSCAN','FEED', 'SRFEED', 'FEEDXOFF', 'FEEDEOFF', 'SIG', 'CAL', 'PROCSEQN', 'PROCSIZE']

#  some more liberal panda dataframe display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# pd.options.display.max_columns = None
 
#%%  classic tp/ps
   
f1 = dysh_data(test="getps")       # OnOff, small one
#f1 = dysh_data(example="getps")   # OnOff, long one (4 IFs, ...)
#f1 = dysh_data(test="AGBT18B_354_03/AGBT18B_354_03.raw.vegas") # OffOn

print("Using",f1.parts[-1])     # isn't there a better name for this?


sdf1 = GBTFITSLoad(f1)
sdf1.summary()
sdf1.summary(verbose=True)
sdf1._index[k]

#%%
p1 = sdf1.gettp()
sp0=p1[0].finalspectrum()
sp1=p1[1].finalspectrum()
print(sp0,sp1)
print("why are the finalspectrum's None")

#%%
sp0 = p1[0].timeaverage()
sp1 = p1[1].timeaverage()
a = (sp0-sp1)/sp0
a.smooth('box',256).plot()

#%%

p2 = sdf1.getps()
sp2 = p2[0].timeaverage()
sp2.smooth('box',256).plot()
p2 = sdf1.getps()



#%% WORKING DIRECTORY


# cd /home/teuben/GBT/dysh_data/acceptance_testing/data/
cd /home/teuben/GBT/dysh/notebooks/developer


#%% EXAMPLE-0  fs/nod

#  example from
#  https://gbtdocs.readthedocs.io/en/latest/how-tos/data_reduction/gbtidl.html
#  TGBT22A_503_02.raw.vegas
#  in here one frequency-switched scan (#64), 
#  and two nodding scans (#62 and #63)
#  7 SDFITS files, one for each beam, are stored in the data directory

# this is a huge file, we preload the minimum number of scans

f2 = dysh_data(example="nod-KFPA/data/TGBT22A_503_02.raw.vegas")
sdf2 = GBTFITSLoad(f2)
sdf2.summary()

# Loaded 7 FITS files
# CPU times: user 17.9 s, sys: 5.54 s, total: 23.4 s
# Wall time: 1min 22s
# 18GB VIRT, 5G RES 4G SHR

#     SCAN OBJECT VELOCITY   PROC  PROCSEQN   RESTFREQ    DOPFREQ # IF # POL # INT # FEED     AZIMUTH   ELEVATIO
# 0     60   W3_1    -40.0  OnOff         1  23.959156  23.694495    6     2    31      7  324.227878  38.705977
# 1     61   W3_1    -40.0  OnOff         2  23.959156  23.694495    6     2    31      7  324.152607  39.012111
# 2     62   W3_1    -40.0    Nod         1  23.959156  23.694495    6     2    31      7  324.274257  38.419406
# 3     63   W3_1    -40.0    Nod         2  23.959156  23.694495    6     2    31      7  324.367171  38.285767


sdf2.write('junk62.fits',scan=[62,63])

#%%


f2 = 'junk62/junk625.fits'   # beam 3, the sig in scan 62
sdf2 = GBTFITSLoad(f2)
sdf2.summary()
sdf2.summary(verbose=True)
sdf2._index[k]

f3 = 'junk62/junk626.fits'   # beam 7, the sig in scan 63
sdf3 = GBTFITSLoad(f3)
sdf3.summary()
sdf3.summary(verbose=True)
sdf3._index[k]


p62 = sdf2.gettp(scan=62, ifnum=0, plnum=0)[0]
p63 = sdf2.gettp(scan=63, ifnum=0, plnum=0)[0]

#%%

f2 = 'junk62.fits'    # all 7 beams
sdf2 = GBTFITSLoad(f2)
sdf2.summary()
for b in range(7):
    print(sdf2.gettp(scan=62, ifnum=0, plnum=0, hdu=b+1)[0]._index[k][:1])


for b in range(7):
    print(GBTFITSLoad(f'junk62{b}.fits').gettp(scan=62, ifnum=0, plnum=0)[0]._index[k][:1])

# beams:   2, 4, 5, 6, 1, 3, 7
# FEED  SRFEED  FEEDXOFF  FEEDEOFF
#    3       0       0.0       0.0  junk626 
#    7       0  0.045644       0.0  junk625  
#    1       0  0.022822 -0.013178    
#    6       0  0.045644 -0.026356 
#    5       0  0.022822 -0.039533 
#    4       0       0.0 -0.026356 
#    2       0  0.022822  0.013178  junk620



#%%
#     now with the mutifile trick
sdf2 = GBTFITSLoad('junk62')
sdf2.summary()
sdf2._index[k]    # 10416 rows
    
#%% EXAMPLE-1   tp_nocal

f1 = dysh_data(accept='AGBT22A_325_15/AGBT22A_325_15.raw.vegas/')
sdf=GBTFITSLoad(f1)
# 8 files, each has 2 beams - 4 scans, VANE/SKY/Nod/Nod
sdf.summary()
# extract 290 and 289 (note order is odd in sdfits)
sdf.write('junk289.fits',scan=[289,290])

sdf2890 = GBTFITSLoad('junk2890.fits')
sdf2890.summary()
sdf2890._index[k]    # 24 rows

#%% EXAMPLE-2 tp

f2 = dysh_data(accept='TREG_050627/TREG_050627.raw.acs/TREG_050627.raw.acs.fits')
sdf=GBTFITSLoad(f2)
sdf.summary()
# extracts 182 and 183 for testing
sdf.write('junk182.fits', scan=[182,183])

sdf182 = GBTFITSLoad('junk182.fits')
sdf182.summary()
sdf182._index[k]   # 128 rows

# data[beam=2][scan=2][if=2][time=4][pol=2][cal=2]


#%% EXAMPLE-3 tp_nocal


f3 = dysh_data(accept='AGBT15B_244_07/AGBT15B_244_07.raw.vegas')
sdf=GBTFITSLoad(f3)
# 8 fits files,   2 for beams, 4 for IF  - 12 scans (2 CALSEQ)
sdf.summary()
# 11072 rows
sdf.write('junk131.fits', scan=[131,132])

 
sdf1310 = GBTFITSLoad('junk1310.fits')
sdf1310.summary()
sdf1310._index[k]    # 244 rows

# where did the IF go?   1310 is for one beam (there are 7)
# 


#%% EXAMPLE-4  tp

f4 = dysh_data(accept='TGBT18A_500_06/TGBT18A_500_06.raw.vegas')
sdf = GBTFITSLoad(f4)
# 2 files - 10 scans written in odd order (64,65,66 first)
sdf.summary()
sdf.write('junk57.fits', scan=[57,58])

sdf570 = GBTFITSLoad('junk570.fits')
sdf570.summary()
sdf570._index[k]   # 240 rows

sdf571 = GBTFITSLoad('junk571.fits')
sdf571.summary()
sdf571._index[k]   # 240 rows
# data[int=30,30][pol=2][cal=2] 

#%% EXAMPLE-5  tp

f5 = dysh_data(accept='TSCAL_19Nov2015/TSCAL_19Nov2015.raw.acs/TSCAL_19Nov2015.raw.acs.fits')
sdf=GBTFITSLoad(f5)
# 1 file, 1 scan
sdf.summary()
sdf._index[k]    # 64 rows

# data[scan=1][beam=2][if=2][time=4][pol=2][cal=2]

# NOTE: this data incomplete? only has procseqn=1

#%% EXAMPLE-6  tp

f6 = dysh_data(accept='AGBT17B_319_06/AGBT17B_319_06.raw.vegas')
sdf = GBTFITSLoad(f6)
# 3 files
sdf.summary()
sdf.write('junk9.fits', scan=[9,10])

sdf90 = GBTFITSLoad('junk90.fits')
sdf90.summary()
sdf90._index[k]   # 248 rows

#  data     [scan=2][int=31][pol=2][cal=2]

#%% EXAMPLE-7  tp

f7 = dysh_data(accept='TGBT21A_501_10/TGBT21A_501_10.raw.vegas')
sdf=GBTFITSLoad(f7)
# 2 files
sdf.summary()
sdf.write('junk36.fits', scan=[36,37])

sdf360 = GBTFITSLoad('junk360.fits')
sdf360.summary()
sdf360._index[k]  # 128, beam 3

sdf361 = GBTFITSLoad('junk361.fits')
sdf361.summary()
sdf361._index[k]  # 128, beam 7

#   data   [ scan=2] [int=16] [pol=2] [cal=2]

 
