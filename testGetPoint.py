import numpy as np
import pickle
import cameraMatrix as cm

## Load pickle
with open("data.pickle","rb") as fr:
    data = pickle.load(fr)

# data.printfd()
# data.printDist()
# data.printExVector()

#### 2d -> 3d 변환 데이터 출력
temp = cm.find3dPoint(data.opt_mtx, data.dist)

# temp.printinputData()

#2차원 영상좌표
points_2D = np.array([
                        (1169, 2093),  #좌 하단 
                        (2638, 2071),  #우 하단
                        (1184, 648),  #좌 상단
                        (2612, 635),  #우 상단
                      ], dtype="double")
                      
# #3차원 월드좌표
points_3D = np.array([
                      (-2.5, -2.5, 11),       #좌 하단
                      (2.5, -2.5, 11),        #우 하단
                      (-2.5, 2.5, 11),        #좌 상단
                      (2.5, 2.5, 11)          #우 상단
                     ], dtype="double")

# temp.printinputData()
temp.findRdata(points_2D, points_3D)
# temp.printTransData()

# temp.worldPoint (2009, 1482, 5, 1460)
# temp.worldPoint(1976, 1417, 7.6, 457)
x, y = temp.worldPoint(1924, 1323, 5, 1431) # 입력값: x좌표(px), y좌표(px), 실제크기, px크기
print('x좌표: ', x)
print('y좌표: ', y)
