# -- 체커보드를 이용한 카메라 정보 저장 
# -- pickle  사용

import cameraMatrix as cm
import numpy as np
import pickle

CHECKER_ROW = 9
CHECKER_COL = 7
MAX_ITER = 100

# ---- 체커보드 이용
checkerWhere = "./sample"   # checker board 이미지 폴더 경로
allchecker = "*"       # 모든 이미지
checkerImgExt = "jpg"  # checker board 이미지 확장자

cali = cm.Calibration(MAX_ITER)
checkerImgsPath = cm.getIMGPath(checkerWhere, allchecker, checkerImgExt)
print(checkerImgsPath, ": Collecting Data ....")
cali.findCamMtx(CHECKER_COL, CHECKER_ROW, checkerImgsPath)
# -----

print()

# ----- 최적화
testFilePath = "./origin_img"    # 왜곡 제거할 이미지 폴더 경로
testImgName = "dist_sample2"          # 왜곡 제거할 이미지 이름
testImgExt = "jpg"              # 왜곡 제거할 이미지 확장자

testImgPath = cm.getIMGPath(testFilePath, testImgName, testImgExt)

print("get optimal camera matrix:", testImgPath)
cali.optimalMtx(testImgPath)

# print("finish calibration:", storeImgPath)

# cali.printfd()
# ------

print()

## Save pickle
with open("data.pickle","wb") as fw:
    pickle.dump(cali, fw)

print("Finish save data:: data.pickle")