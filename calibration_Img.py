import cameraMatrix as cm
import pickle

with open("data.pickle","rb") as fr: # 저장된 데이터 가져오기
    cali = pickle.load(fr)

# cali.printfd()
# cali.printDist()
# cali.printExVector()

# 이미지 왜곡 제거 결과 확인용

testFilePath = "./origin_img"    # 왜곡 제거할 이미지 폴더 경로
testImgName = "img_point2"          # 왜곡 제거할 이미지 이름
testImgExt = "jpg"              # 왜곡 제거할 이미지 확장자

storeFilePath = "./after_test"    # 이미지 저장 폴더 경로
storeImgName = "img_point2"          # 저장할 이미지 이름
storeImgExt = "jpg"              # 저장할 이미지 확장자

testImgPath = cm.getIMGPath(testFilePath, testImgName, testImgExt)
storeImgPath = cm.getIMGPath(storeFilePath, storeImgName, storeImgExt)

print("test remove distort:", testImgPath)

cali.removeImgDistort(testImgPath, storeImgPath)
print("finish calibration:", storeImgPath)

# cali.printfd()
# cali.printDist()
# cali.printExVector()