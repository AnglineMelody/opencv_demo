'''import cv2
img=cv2.imread("aaa.jpg")
img1=cv2.pyrDown(img)
img2=cv2.pyrDown(img1)
img3=cv2.pyrDown(img2)
print("img.shape",img.shape)
print("img1.shape",img1.shape)
print("img2.shape",img2.shape)
cv2.imshow("a",img)
cv2.imshow("b",img1)
cv2.imshow("c",img2)
cv2.waitKey()
cv2.destroyAllWindows()'''

'''import cv2
img=cv2.imread("bbb.jpg")
img1=cv2.pyrUp(img)
img2=cv2.pyrUp(img1)
img3=cv2.pyrUp(img2)
print("img.shape",img.shape)
print("img1.shape",img1.shape)
print("img2.shape",img2.shape)
cv2.imshow("a",img)
cv2.imshow("b",img1)
cv2.imshow("c",img2)
cv2.waitKey()
cv2.destroyAllWindows()'''

'''import cv2
img=cv2.imread("ccc.jpg")
img1=cv2.pyrUp(img)
img2=cv2.pyrDown(img1)
cv2.imshow("a",img)
cv2.imshow("b",img1)
cv2.imshow("c",img2)
print("img.shape",img.shape)
print("img1.shape",img1.shape)
print("img2.shape",img2.shape)
cv2.waitKey()
cv2.destroyAllWindows()'''

'''import cv2
import numpy as np
g0=cv2.imread("aaa.jpg")
g1=cv2.pyrDown(g0)
g2=cv2.pyrDown(g1)
g3=cv2.pyrDown(g2)
l1=g0-cv2.pyrUp(g1)#bug
l2=g1-cv2.pyrUp(g2)#bug
l3=g2-cv2.pyrUp(g3)#bug
print("l1",l1.shape)
print("l2",l2.shape)
print("l3",l3.shape)
cv2.imshow("a",g0)
cv2.imshow("b",l1)
cv2.imshow("c",l2)
cv2.imshow("d",l3)
cv2.waitKey()
cv2.destroyAllWindows()'''

'''import cv2
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/bbb.jpg")
rst=cv2.Sobel(img,-1,1,0)
cv2.imshow("a",img)
cv2.imshow("b",rst)
cv2.waitKey()
cv2.destroyAllWindows()'''

'''import cv2
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/bbb.jpg",cv2.IMREAD_GRAYSCALE)
rst=cv2.Sobel(img,-1,1,0)
cv2.imshow("a",img)
cv2.imshow("b",rst)
cv2.waitKey()
cv2.destroyAllWindows()'''

'''import cv2
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/bbb.jpg",cv2.IMREAD_GRAYSCALE)
rst=cv2.Sobel(img,cv2.CV_64F,1,0)
cv2.imshow("a",img)
cv2.imshow("b",rst)
cv2.waitKey()
cv2.destroyAllWindows()'''

'''import cv2
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/ggg.png")
Scharrx=cv2.Scharr(img,cv2.CV_64F,1,0)
Scharrx=cv2.convertScaleAbs(Scharrx)
cv2.imshow("a",img)
cv2.imshow("b",Scharrx)
cv2.waitKey()
cv2.destroyAllWindows()'''

'''import cv2
img=cv2.imread("Users/yy/PycharmProjects/小张的opencv/ggg.png")
Scharry=cv2.Scharr(img,cv2.CV_64F,0,1)
Scharry=cv2.convertScaleAbs(Scharry)
cv2.imshow("a",img)
cv2.imshow("b",Scharry)
cv2.waitKey()
cv2.destroyAllWindows()'''
#水平方向和垂直方向的和(在粘贴一些路径时，是不允许有中文的，所以依情况而定）
'''import cv2
img=cv2.imread("ggg.png")
Scharrx=cv2.Scharr(img,cv2.CV_64F,1,0)
Scharry=cv2.Scharr(img,cv2.CV_64F,0,1)
Scharrx=cv2.convertScaleAbs(Scharrx)
Scharry=cv2.convertScaleAbs(Scharry)
Scharra=cv2.addWeighted(Scharrx,0.5,Scharry,0.5,0)
cv2.imshow("a",img)
cv2.imshow("b",Scharra)
cv2.waitKey()
cv2.destroyAllWindows()'''

'''import cv2
img=cv2.imread("ggg.png")
x=cv2.Sobel(img,cv2.CV_64F,1,0,-1)
y=cv2.Sobel(img,cv2.CV_64F,0,1,-1)
x=cv2.convertScaleAbs(x)
y=cv2.convertScaleAbs(y)
xy=cv2.addWeighted(x,0.5,y,0.5,0)
cv2.imshow("a",img)
cv2.imshow("b",img)
cv2.imshow("c",x)
cv2.imshow("d",y)
cv2.waitKey()
cv2.destroyAllWindows()'''

'''import cv2
img=cv2.imread("bbb.jpg")
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,5)
sobely=cv2.Sobel(img,cv2.CV_64F,0,1,3)
sobelx=cv2.convertScaleAbs(sobelx)
sobely=cv2.convertScaleAbs(sobely)
sobelxy=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
scharrx=cv2.Scharr(img,cv2.CV_64F,1,0)
scharry=cv2.Scharr(img,cv2.CV_64F,0,1)
scharrx=cv2.convertScaleAbs(scharrx)
scharry=cv2.convertScaleAbs(scharry)
scharrxy=cv2.addWeighted(scharrx,0.5,scharry,0.5,0)
cv2.imshow("a",img)
cv2.imshow("b",sobelx)
cv2.imshow("c",sobely)
cv2.imshow("d",sobelxy)
cv2.imshow("e",scharrx)
cv2.imshow("f",scharry)
cv2.imshow("g",scharrxy)
cv2.waitKey()
cv2.destroyAllWindows()'''

'''import cv2
img=cv2.imread("ggg.png")

import cv2
import numpy as np
o=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/fff.png",cv2.IMREAD_UNCHANGED)
kernel=np.ones((5,5),np.uint8)
rst=cv2.erode(o,kernel)
cv2.imshow("a",o)
cv2.imshow("b",rst)
cv2.waitKey()
cv2.destroyAllWindows()'''
'''
import cv2
import cv2
import numpy as np
o=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/fff.png",cv2.IMREAD_UNCHANGED)
kernel=np.ones((9,9),np.uint8)
rst=cv2.erode(o,kernel ,iterations=5)
cv2.imshow("a",o)
cv2.imshow("b",rst)
cv2.waitKey()
cv2.destroyAllWindows()'''
'''
import cv2
import numpy as np
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/bbb.jpg")
knerl=np.ones((5,5),dtype=np.uint8)
rst=cv2.dilate(img,knerl)
cv2.imshow("a",img)
cv2.imshow("b",rst)
cv2.waitKey()
cv2.destroyAllWindows()

#通用形态学函数
#开运算
import cv2
import numpy as np
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/fff.png")
knerl=np.ones((5,5),dtype=np.uint8)
rst=cv2.morphologyEx(img,cv2.MORPH_OPEN,knerl)
cv2.imshow("a",img)
cv2.imshow("b",rst)
cv2.waitKey()
cv2.destroyAllWindows()

import cv2
import numpy as np
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/fff.png")
knerl=np.ones((5,5),dtype=np.uint8)
rst=cv2.morphologyEx(img,cv2.MORPH_CLOSE,knerl)
cv2.imshow("a",img)
cv2.imshow("b",rst)
cv2.waitKey()
cv2.destroyAllWindows()'''
'''
import cv2
import numpy as np
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/fff.png")
knerl=np.ones((5,5),dtype=np.uint8)
rst=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,knerl)
cv2.imshow("a",img)
cv2.imshow("b",rst)
cv2.waitKey()
cv2.destroyAllWindows()'''
'''
import cv2
import numpy as np
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/fff.png")
knerl=np.ones((5,5),dtype=np.uint8)
rst=cv2.morphologyEx(img,cv2.MORPH_TOPHAT,knerl)
cv2.imshow("a",img)
cv2.imshow("b",rst)
cv2.waitKey()
cv2.destroyAllWindows()'''
'''
import cv2
import numpy as np
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/fff.png")
knerl=np.ones((5,5),dtype=np.uint8)
rst=cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,knerl)
cv2.imshow("a",img)
cv2.imshow("b",rst)
cv2.waitKey()
cv2.destroyAllWindows()'''

'''import cv2
import numpy as np
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/fff.png")
k=cv2.getStructuringElement(cv2.MORPH_RECT,(50,50))
rst=cv2.dilate(img ,k,)
cv2.imshow("a",img)
cv2.imshow("b",rst)
cv2.waitKey()
cv2.destroyAllWindows()'''
'''
import cv2
import numpy as np
img=np.zeros([2,4,3],dtype=np.uint8)
rst=img.shape[:2]
abc=cv2.resize(img,rst)
print("打印img",img)
print("打印rst",rst)

import cv2
import numpy as np
img=np.zeros([2,4,3],dtype=np.uint8)
cols,rows=img.shape[:2]
size=(int(rows*2),int (cols*0.5))#cols是列，rows是行
rst=cv2.resize(img,size)
print("打印img.shape:",img.shape)
print("打印size",size)
print("打印rst.shape",rst.shape)
'''
'''
import cv2
import numpy as np
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/aaa.jpg")
rst=cv2.flip(img,0)
dst=cv2.flip(img,1)
est=cv2.flip(img,-1)
cv2.imshow("a",img)
cv2.imshow("b",rst)
cv2.imshow("c",dst)
cv2.imshow("d",est)
cv2.waitKey(100)
cv2.destroyAllWindows()'''

'''import cv2
import numpy as np
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/aaa.jpg")
width,height=img.shape[:2]
x=100
y=200
M=np.float32([[1,0,x],[0,1,y]])
rst=cv2.warpAffine(img,M,(width,height))
cv2.imshow("a",img)
cv2.imshow("b",rst)
cv2.waitKey()
cv2.destroyAllWindows()'''
'''
import cv2
import numpy as np
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/aaa.jpg")
width,height=img.shape[:2]
M=cv2.getRotationMatrix2D((width/3,height/2),30,0.8)
rst=cv2.warpAffine(img,M,(width,height))
cv2.imshow("a",img)
cv2.imshow("b",rst)
cv2.waitKey()
cv2.destroyAllWindows()'''
'''
import cv2
import numpy as np
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/aaa.jpg")
rows,cols,ch=img.shape
p1=np.float32([[0,0],[cols-1,0],[0,rows-1]])
p2=np.float32([[0,rows*0.33],[cols*0.85,rows*0.25],[cols*0.15,rows*0.7]])
M=cv2.getAffineTransform(p1,p2)
dst=cv2.warpAffine(img,M,(cols,rows))
cv2.imshow("a",img)
cv2.imshow("b",dst)
cv2.waitKey()
cv2.destroyAllWindows()'''

'''import cv2
import numpy as np
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/aaa.jpg")
cols,rows=img.shape[:2]
print(rows,cols)
pst1=np.float32([[100,50],[200,50],[60,200],[60,300]])
pst2=np.float32([[20,20],[rows-20,20],[20,cols-20],[rows-20,cols-20]])
M=cv2.getPerspectiveTransform(pst1,pst2)
dst=cv2.warpAffine(img,M,(cols,rows))
cv2.imshow("a",img)
cv2.imshow("b",rst)
cv2.waitKey()
cv2.destroyAllWindows()'''

'''import cv2
import numpy as np
img=np.random.randint(0,256,size=[5,5],dtype=np.uint8)
rows,cols=img.shape
mapx=np.ones(img.shape,np.float32)*0
mapy=np.ones(img.shape,np.float32)*3
rst=cv2.remap(img,mapx ,mapy ,cv2.INTER_LINEAR)
print("img",img)
print("mapx",mapx)
print("mapy",mapy)
print("rst",rst)

import cv2
import numpy as np
img=np.random.randint(0,256,size=[5,5],dtype=np.uint8)
rows,cols=img.shape
mapx=np.zeros(img.shape,np.float32)
mapy=np.zeros(img.shape,np.float32)
for i in range(cols):
    for j in range (rows):
        mapx.itemset((i,j),i)
        mapy.itemset((i,j),j)
rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
print("img",img)
print("mapx",mapx)
print("mapy",mapy)
print("rst",rst)'''
'''
import cv2
import numpy as np
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/aaa.jpg")
cols,rows=img.shape[:2]
mapx=np.zeros(img.shape[:2],np.float32)
mapy=np.zeros(img.shape[:2],np.float32)
for j in range (rows):#行
    for i in range(cols):#列
        mapx.itemset((i, j), j)
        mapy.itemset((i, j), i)
rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
cv2.imshow("a",img)
cv2.imshow("b",rst)
cv2.waitKey()
cv2.destroyAllWindows()

''import cv2
import numpy as np
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/aaa.jpg")
cols,rows=img.shape[:2]
mapx=np.zeros(img.shape[:2],np.float32)
mapy=np.zeros(img.shape[:2],np.float32)
for i  in range (cols):#列
    for j in range(rows):#行
        mapx.itemset((i, j), i)
        mapy.itemset((i, j), j)
rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
cv2.imshow("a",img)
cv2.imshow("b",rst)
cv2.waitKey()
cv2.destroyAllWindows()'''

'''import cv2
import numpy as np
img=cv2.imread("/Users/yy/PycharmProjects/小张的opencv/ddd.webp")
cols,rows=img.shape[:2]
mapx=np.zeros(img.shape[:2],np.float32)
mapy=np.zeros(img.shape[:2],np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i,j),i)
        mapy.itemset((i,j),j)
rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
cv2.imshow("a",img)'''