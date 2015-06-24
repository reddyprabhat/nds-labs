#use identify to get image size
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.cm as cm
import os,sys
import heatmap
from matplotlib.patches import Ellipse
import time

import pandas as pd

dft=pd.read_csv('test.csv')

w=np.where((dft.COLOR >=-1) & (dft.COLOR <=3))[0]

df=dft.ix[w]


w= 256
h = 256

#N=500
#x2=np.random.rand(N)*100
#y2=np.random.rand(N)*100
#s2=np.random.rand(N)*3+0
#s3=np.random.rand(N)*3+0
#ang=np.random.rand(N)*360
#col=np.random.rand(N,3)


x2=df.RA.values
y2=df.DEC.values
s2=df.A_IMAGE.values*0.263/3600.*2
s3=df.B_IMAGE.values*0.263/3600.*2
ang=df.THETA_IMAGE.values+90.
colg=(df.COLOR.values-min(df.COLOR.values))
colg=colg/max(colg)
ebv=(df.EBV.values-min(df.EBV.values))
ebv=ebv/max(ebv)


winx=df.RA.max()-df.RA.min()
winy=df.DEC.max()-df.DEC.min()
minx=df.RA.min()
miny=df.DEC.min()
maxx=df.RA.max()
maxy=df.DEC.max()

sys.exit()

hm=heatmap.Heatmap()
img = hm.heatmap(zip(x2,y2),size=(8192,8192),dotsize=120,scheme='fire',opacity=200)
img.save('test_heatmap.png') 



#ells = [Ellipse(xy=np.random.rand(2)*100, width=4, height=4, angle=np.random.rand()*360)
#        for i in range(N)]

nzoom=9
dpiN=np.linspace(10,256,nzoom)
dsN=np.linspace(10,30,nzoom)
dsN=dsN[::-1]

dpit=400
figsize = w*2/(1.*dpit), h*2/(1.*dpit)
fig = Figure(figsize=figsize, dpi=dpit, frameon=False)
canvas = FigureCanvas(fig)
ax = fig.add_axes([0, 0, 1, 1])
nn=60
x = np.linspace(minx, maxx, nn+1)
y = np.linspace(miny, maxy, nn+1)
X, Y = np.meshgrid(x, y)
BBD = np.zeros((X.shape[0]-1, X.shape[1]-1))
for ix in xrange(nn):
    for iy in xrange(nn):
        win=np.where((x2>=x[ix])&(x2<x[ix+1])&(y2>=y[iy])&(y2<y[iy+1]))[0]
        BBD[ix,iy]=np.mean(ebv[win])

ax.pcolor(X, Y, BBD, antialiased=False, cmap=cm.jet, edgecolors='none')
ax.axis( [x[0], x[-1], y[0], y[-1]] )
ax.axis('off')
fig.savefig('test_ebv.png', dpi=dpit)

def cr_image(zoom,xt,yt,dpi):
    t1=time.time()
    #print zoom,xt,yt, dpi
    cutsX=np.linspace(minx,maxx,(2**zoom)+1)
    dX=cutsX[1]-cutsX[0]
    cutsY=np.linspace(miny,maxy,(2**zoom)+1)
    dY=cutsY[1]-cutsY[0]
    win=np.where((x2>=cutsX[xt]-dX/4.) & (x2<=cutsX[xt+1]+dX/4.) &  (y2>=cutsY[-2-yt]-dY/4.) & (y2<=cutsY[-1-yt]+dY/4.))[0]
    dd=int(round(dsN[zoom]))
    dd = (2**zoom)*1
    path = 'tiles2/'+str(zoom)
    path2 = 'tiles3/'+str(zoom)
    if not os.path.exists(path): os.makedirs(path)
    if not os.path.exists(path2): os.makedirs(path2)
    path = 'tiles2/'+str(zoom)+'/'+str(xt)
    if not os.path.exists(path): os.makedirs(path)
    path2 = 'tiles3/'+str(zoom)+'/'+str(xt)
    if not os.path.exists(path2): os.makedirs(path2)
    filez = 'tiles2/'+str(zoom)+'/'+str(xt)+'/'+str(yt)+'.png'
    filem = 'tiles3/'+str(zoom)+'/'+str(xt)+'/'+str(yt)+'.png'
    figsize = w/(1.*dpi), h/(1.*dpi)
    fig = Figure(figsize=figsize, dpi=dpi, frameon=False)
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    x = np.linspace(cutsX[xt], cutsX[xt+1], 10)
    y = np.linspace(cutsY[-2-yt], cutsY[-1-yt], 10)
    X, Y = np.meshgrid(x, y)
    D = np.zeros((X.shape[0]-1, X.shape[1]-1))
    ax.pcolor(X, Y, D, antialiased=False, cmap=cm.binary_r)
    #ax.scatter(x2,y2,s2,color='white', edgecolor='none')
    for ij in xrange(len(win)):
        j=win[ij]
        e=Ellipse([x2[j],y2[j]],width=s2[j], height=s3[j], angle=ang[j], fc=cm.RdBu_r(colg[j]), edgecolor='none')
        ax.add_artist(e)
    ax.axis( [x[0], x[-1], y[0], y[-1]] )
    ax.axis('off')
    fig.savefig(filez, dpi=dpi)
    eb=True
    if eb:
        fig = Figure(figsize=figsize, dpi=dpi, frameon=False)
        canvas = FigureCanvas(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        nn=60/(2**zoom)
        #if nn == 64: nn=60
        x = np.linspace(cutsX[xt], cutsX[xt+1], nn+1)
        y = np.linspace(cutsY[-2-yt], cutsY[-1-yt], nn+1)
        X, Y = np.meshgrid(x, y)
        #D = np.zeros((X.shape[0]-1, X.shape[1]-1))
        D = BBD[(2**zoom-1-yt)*nn:(2**zoom-1-yt+1)*nn,xt*nn:(xt+1)*nn]
        #D= D [:,::-1]   
        #for ix in xrange(nn):
            #for iy in xrange(nn):
                #win=np.where((x2>=x[ix])&(x2<x[ix+1])&(y2>=y[iy])&(y2<y[iy+1]))[0]
                #D[iy,ix]=np.mean(ebv[win])

        ax.pcolor(X, Y, D, antialiased=False, cmap=cm.jet, edgecolors='none',vmin=0,vmax=1)
        ax.axis( [x[0], x[-1], y[0], y[-1]] )
        ax.axis('off')
        fig.savefig(filem, dpi=dpi)
    
    #print '%.5f seconds' % (time.time()-t1)
    #hm=heatmap.Heatmap()
    #img = hm.heatmap(zip(x2[win],y2[win]),size=(256,256),area=((cutsX[xt],cutsY[-2-yt]),(cutsX[xt+1],cutsY[-1-yt])),dotsize=dd)
    #img.save(filem)            
            
            
          
for z in xrange(4,5):
    nx=2**z
    ny=2**z
    #dpi=int(round(dpiN[zoom]))
    dpi = 2**(z+(8-nzoom+1))
    t1=time.time()
    print z
    for xt in range(nx):
        for yt in range(ny):
            cr_image(z,xt,yt,dpi)
    print '%.5f seconds' % (time.time()-t1)






# Test image
#from PIL import Image
#im = Image.open('tiles2/testfile.png')
#im.show()
