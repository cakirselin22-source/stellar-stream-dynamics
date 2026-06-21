import os
from pathlib import Path
import numpy as np

# Falls back to ./data if the STREAM_DATA_DIR env var isn't set
DEFAULT_DATA_DIR = Path(os.environ.get("STREAM_DATA_DIR", "./data"))

def readsnap(dir, ifile, data_dir=DEFAULT_DATA_DIR):
  filename = Path(data_dir) / dir / f"snapshot_{ifile:04}"

  hd = np.fromfile(filename,dtype='uint32',count=7)
  nd = hd[2]
  ns = hd[5]
  nc = hd[3]
  nb = hd[4]
 
  head = np.fromfile(filename,dtype='double',count=7,offset=4+6*4)
  mdark=head[1]*1.e10
  mstar=head[4]*1.e10
  
  time=head[6]

  
  start_d = 0
  start_c = nd
  start_s = nd + nc + nb
  start_b = nd + nc
  block = nd + nc + ns + nb
  
  
  skipx = 256 + 3*4 + start_c*3*8
  skipv = 256 + 5*4 + (block*3 + start_c*3)*8
  skipi = 256 + 7*4 + block*3*2*8 + start_c*4
  rc = np.fromfile(filename,dtype='double',count=3*nc,offset=skipx)
  vc = np.fromfile(filename,dtype='double',count=3*nc,offset=skipv)
  idc = np.fromfile(filename,dtype='uint32',count=nc,offset=skipi)
  idc -= nd
  ki = np.zeros(nc,dtype='uint32')
  for j in range(nc):
      ix = idc[j]
      ki[ix] = j
      jrc3 = np.reshape(rc,(nc,3))
      jvc3 = np.reshape(vc,(nc,3))
      rc3 = jrc3[ki]
      vc3 = jvc3[ki]



  #stars
  skipx = 256 + 3*4 + start_s*3*8
  skipv = 256 + 5*4 + block*3*8 + start_s*3*8 #skip over r, and dm+c v's
  skipi = 256 + 7*4 + block*3*2*8 + start_s*4 #skip over r and v and dm+c id
  skipc = 256 + 9*4 + block*(2*3*8+4) + start_s*4 #skip over r, v and id and dm+c ic
  rs= np.fromfile(filename,dtype='double',count=3*ns,offset=skipx)
  vs = np.fromfile(filename,dtype='double',count=3*ns,offset=skipv)
  ids = np.fromfile(filename,dtype='uint32',count=ns,offset=skipi)
  jics = np.fromfile(filename,dtype='uint32',count=ns,offset=skipc)

  jrs3 = np.reshape(rs,(ns,3))
  jvs3 = np.reshape(vs,(ns,3))
  ids -= nd + nc + nb
  rs3 = np.zeros_like(jrs3)
  vs3 = np.zeros_like(jvs3)
  ics = np.zeros_like(jics)

  rs3[ids] = jrs3 #[ids]
  vs3[ids] = jvs3 #[ids]
  ics[ids] = jics #[ids]

  return rs3, vs3, rc3, vc3, time

def readtout(dir,ifile,data_dir=DEFAULT_DATA_DIR):
  toutfile =  Path(data_dir) /  dir /f"TOUT{ifile:04}" 
  thead = np.fromfile(toutfile,dtype='double',count=1)
  Tend = thead[0]
  nnn = np.fromfile(toutfile,dtype='uint32',count=3,offset=8)
  nstart = nnn[0]
  ntc = nnn[1]
  nts = nnn[2]
  Toffset = 14.1 - Tend
  #print(Tend,nstart,ntc,nts,Toffset)
  noff = 8 + 3*4
  Nco = np.zeros(ntc,dtype='uint32')
  outid = []
  ctout = []
  
  for i in range(ntc):
      Nco[i] = np.fromfile(toutfile,dtype='uint32',count=1,offset=noff)
      noff += 4
      readoutN = np.fromfile(toutfile,dtype='uint32',count=Nco[i],offset=noff)
      outid.append(readoutN)
      noff += 4*Nco[i]
      readoutT= np.fromfile(toutfile,dtype='double',count=Nco[i],offset=noff)
      ctout.append(readoutT)
      noff += 8*Nco[i]
    
#  print(len(ctout),len(outid))
  return ctout, outid # or in []

