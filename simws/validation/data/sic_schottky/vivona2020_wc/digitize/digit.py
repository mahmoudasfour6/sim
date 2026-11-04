from PIL import Image; import numpy as np, json
a=np.asarray(Image.open('p4_1.jpeg').convert('RGB')).astype(float)
H,W,_=a.shape
R,G,B=a[...,0],a[...,1],a[...,2]
# axis calibration from the detected ticks
xt=np.array([121,189,256.5,324,392,459.5,527.5,595]); vt=np.arange(8)*0.2
px=np.polyfit(vt,xt,1)
yt=np.array([534.5-39.04*k for k in range(14)])   # decades -11..2
py=(534.5, 39.04)
cls={
 'black': (R<90)&(G<90)&(B<90),
 'red': (R>170)&(G<90)&(B<90),
 'green': (G>90)&(R<90)&(B<90)&(G>R+40),
 'magenta': (R>170)&(B>170)&(G<120),
 'blue': (B>170)&(R<90)&(G<120),
}
mask=np.zeros((H,W),bool); mask[60:531,126:626]=True   # J < 10^0.8 A/cm2: above every curve
mask[320:,440:]=False          # legend
mask[:80,:340]=False           # label text
out={}
for name,m in cls.items():
    m=m&mask
    ys,xs=np.nonzero(m)
    pts=[]
    for x in np.unique(xs):
        yy=ys[xs==x]
        if len(yy)==0: continue
        # curve thickness ~2-4 px; steep parts give long runs: keep the median of each column
        pts.append((x,np.median(yy),len(yy)))
    pts=np.array(pts)
    V=(pts[:,0]-px[1])/px[0]; lJ=-11+(py[0]-pts[:,1])/py[1]
    out[name]={'V':V.tolist(),'log10J':lJ.tolist(),'npx':pts[:,2].tolist()}
    print(name,len(pts),'V range',V.min().round(3),V.max().round(3),'logJ',lJ.min().round(2),lJ.max().round(2))
json.dump(out,open('raw.json','w'))
print('x scale px/V',px)
