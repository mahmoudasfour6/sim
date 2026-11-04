"""Fig. 3 of arXiv:2102.08927 (p.4 JPEG, 642x614): colour separation of the seven forward J-V curves."""
from PIL import Image; import numpy as np, json, colorsys
a=np.asarray(Image.open('p3_2.jpeg').convert('RGB')).astype(float)/255
H,W,_=a.shape
hsv=np.array([[colorsys.rgb_to_hsv(*a[y,x]) for x in range(W)] for y in range(H)])
h,s,v=hsv[...,0]*360,hsv[...,1],hsv[...,2]
# axes from the detected ticks: x 0.0..2.0 V at 112.5..618.5 px; decades 10^1..10^-8 at y 48.5..504.5
xt=np.array([112.5,239,366,492.5,618.5]); vt=np.array([0,0.5,1.0,1.5,2.0]); px=np.polyfit(vt,xt,1)
yt=np.array([48.5,99.5,150,200.5,251,302,352.5,403.5,454,504.5]); lt=np.arange(1,-9,-1); py=np.polyfit(lt,yt,1)
col={'T25':(v<0.3),
     'T40':((h>340)|(h<15))&(s>0.6)&(v>0.5),
     'T55':(h>100)&(h<140)&(s>0.6),
     'T70':(h>215)&(h<250)&(s>0.6),
     'T85':(h>165)&(h<195)&(s>0.6),
     'T100':(h>285)&(h<320)&(s>0.6),
     'T115':(h>40)&(h<80)&(s>0.6)&(v<0.8)}
mask=np.zeros((H,W),bool); mask[18:515,117:360]=True        # V < 0.98 V: the region the windows use; no inset
mask[225:265,110:215]=False                                    # '115 C' label
mask[410:450,200:280]=False                                    # '25 C' label
yy,xx=np.mgrid[0:H,0:W]
x0,y0,x1,y1=236,398,163,258                                    # arrow (black only)
d=np.abs((y1-y0)*xx-(x1-x0)*yy+x1*y0-y1*x0)/np.hypot(y1-y0,x1-x0)
on_seg=((xx-x0)*(x1-x0)+(yy-y0)*(y1-y0)>=-8)&((xx-x1)*(x0-x1)+(yy-y1)*(y0-y1)>=-8)
arrow=(d<6)&on_seg
out={}
for n,m in col.items():
    m=m&mask
    if n=='T25': m=m&~arrow; m[:, :135]=False      # left-axis tick marks are black too
    ys,xs=np.nonzero(m); pts=[]
    for x in np.unique(xs):
        pts.append((x,np.median(ys[xs==x])))
    pts=np.array(pts)
    V=(pts[:,0]-px[1])/px[0]; L=(pts[:,1]-py[1])/py[0]
    out[n]={'V':V.tolist(),'log10J':L.tolist()}
    print(n,len(pts),round(V.min(),3),round(V.max(),3),round(L.min(),2),round(L.max(),2))
json.dump(out,open('raw.json','w'))
print('px/V',px,'decade',py)
