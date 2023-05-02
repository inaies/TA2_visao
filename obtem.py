import numpy as np
import cv2 as cv
import glob

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (8, 8)
frameSize = (2048, 2048)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0],
                       0:chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 30
objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

cap = cv.VideoCapture(0)

images = glob.glob('./*.png')
j=1

while True:
    # Captura um frame
    r, img = cap.read()

    # Verifica se a captura foi bem-sucedida
    if not r:
        break
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Mostra o frame na janela "Video"
    cv.imshow('Video', gray)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imwrite(f'./imgs/obt{j}.png', img)
        j+=1
        cv.waitKey(1000)

    # Verifica se a tecla "q" foi pressionada
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


j=1
images = glob.glob('./imgs/*.png')
for image in images:
    ############## CALIBRATION #######################################################

    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)


    ############## UNDISTORTION #####################################################
    print(image)
    img = cv.imread(image)
    h,  w = img.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))


    # Undistort
    dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(f'./calibration/imgCalibrada{int(j/2)+1}-1.png', dst)
    cv.imwrite(f'./calibration/imgCalibrada{int(j/2)+1}-Original.png', img)
    j+=1


    # Undistort with Remapping
    mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(f'./calibration/imgCalibrada{int(j/2)}-2.png', dst)
    j+=1


# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )
