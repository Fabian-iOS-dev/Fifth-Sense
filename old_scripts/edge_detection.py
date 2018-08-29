import cv2
import numpy as np
import imutils
from old_scripts.transform import four_point_transform
from skimage.measure import compare_ssim, compare_mse


# #resize picture while keeping aspect ratio
# r = 300.0 / w
# dim = (300, int(h * r))
# resized = cv2.resize(image, dim)
#
img = cv2.imread('../test_pics/testnote_nocrop.JPG')
# img = cv2.imread('../test_pics/testnote_nocrop_I.JPG')
# img = cv2.imread('perspective.JPG')

ratio = img.shape[0]/500.0
orig = img.copy()
img = imutils.resize(img, height=500)
cv2.imshow('img',img)

# img = cv2.imread('letter.JPG')
# yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# gbI = cv2.medianBlur(yuv, 5)
# gb = cv2.blur(yuv,(6,6))

# cv2.imshow('yuv',yuv)
# cv2.imshow('medianblur',gbI)
# cv2.imshow('blur',gb)
# cv2.imshow('gray', gray)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray',gray)

# gblurred = cv2.GaussianBlur(gray, (5,5), 3)
blurred = cv2.bilateralFilter(gray, 7, 60, 60)
# cv2.imshow('blur',blurred)
# cv2.imshow('gblur',gblurred)


threshold = cv2.threshold(blurred, np.average(blurred), 255, cv2.THRESH_BINARY)[1]
# cv2.imshow('tr',threshold)

th3 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# cv2.imshow('t3',th3)

for_edges = th3

sigma = 1
v = np.median(for_edges)
lower = int(max(0, (1.0 -sigma)*v))
upper = int(min(255, (1.0+sigma)*v))

# lower = 15
# upper = 300

edges = cv2.Canny(for_edges, lower,upper)
cv2.imshow('edges', edges)

cv2.waitKey(0)

kernel = np.ones((5,5), np.uint8)
dilation = cv2.dilate(edges, kernel, iterations = 1)
cv2.imshow('dilation', dilation)


im_temp, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

max_l = 0.0
max_c = None
for contour in contours:
    length = cv2.arcLength(contour, closed=1)

    if length > max_l:
        max_l = length
        max_c = contour

print(max_c)
# c = max(contours, key = cv2.arcLength(contour)) #get largets contour

cont = cv2.drawContours(img.copy(), max_c, -1, (128,255,0), 3)
# cv2.imshow('cont', cont)

hull = cv2.convexHull(max_c)
hull_img=cv2.drawContours(img.copy(), [hull], -1, (204,0,0), 3)
# cv2.imshow('hull', hull_img)

epsilon = 0.02 * cv2.arcLength(hull, True)
approx = cv2.approxPolyDP(hull, epsilon, False)
aprox_img=cv2.drawContours(img.copy(), [approx], -1, (204,255,0), 3)
# cv2.imshow('approx', aprox_img)

rect = cv2.minAreaRect(approx)
box = cv2.boxPoints(rect)
box = np.int0(box)
im = cv2.drawContours(img.copy(),[box],0,(0,0,255),2)
# cv2.imshow('rect', im)

#perform transform on original sized image
warped, rect = four_point_transform(orig, box*ratio)

#(tl, tr, br, bl) = rect
if rect[1][0]-rect[0][0] < rect[3][1]-rect[0][1]:
    rotated = imutils.rotate_bound(warped, 270)

cv2.imshow('warped', rotated)

#rotation testing
org = cv2.imread('back.JPG')

#needs same shape
# org = imutils.resize(org, height=rotated.shape[0], width=rotated.shape[1])
org = cv2.resize(org, (rotated.shape[1], rotated.shape[0]), interpolation=cv2.INTER_AREA)
mse = compare_mse(rotated, org)
ssim = compare_ssim(rotated, org, multichannel=True) #-1 worst, 1 best

print('mse: ',mse, 'ssim: ', ssim)


cv2.waitKey(0)

# approx_list = []
# for contour in contours:
#     epsilon = 0.1 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, False)
#     approx_list.append(approx)



# edgesg = cv2.Canny(gray, 100,200, apertureSize = 3)
# cv2.imshow('edges_G',edgesg)
# edgesy = cv2.Canny(yuv, 50,200, apertureSize = 3, L2gradient=True)
# cv2.imshow('edges_Y',edgesy)
# edges = cv2.Canny(threshold, lower,upper)
# lines = cv2.HoughLines(edges,1,np.pi/180,200)

for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)



cv2.imshow('edges',edges)
cv2.imshow('lines',img)

cv2.waitKey(0)

#contours
im_temp, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# im_temp, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (128,255,0), 3)
cv2.imshow('Keypoints',img)



hull_list = []
for contour in contours:
    hull = cv2.convexHull(contour)
    hull_list.append(hull)

approx_list = []
for contour in contours:
    epsilon = 0.1 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(hull, epsilon, False)
    approx_list.append(approx)

hull_img=cv2.drawContours(img, hull_list, -1, (204,0,0), 3)
aprox_img=cv2.drawContours(img, approx_list, -1, (204,255,0), 3)
cv2.imshow('hull:', hull_img)
# cv2.imshow('aprox:', aprox_img)



min_x, min_y = img.shape[1],img.shape[0]
max_x, max_y = 0,0
c1, c2, c3, c4 = [0,0],[0,0],[0,0],[0,0]

def get_avg(x_or_y, addition):
    return round(((x_or_y*2+addition)/2))

for approx in approx_list:
    (x,y,w,h) = cv2.boundingRect(approx)
    factor = 0.1

    if min_x > x:
        c1 = [x, get_avg(y,h)]
        min_x = x

    if min_y > y:
        c2 = [get_avg(x,w), y]
        min_y = y

    if max_y < (y+h):
        c3 = [get_avg(x,w), y+h]
        max_y = (y+h)

    if max_x < (x+w):
        c4 = [x+w, get_avg(y,h)]
        max_x = (x+w)


    # min_x, max_x = min(x, min_x), max(x+w, max_x)
    # min_y, max_y = min(y, min_y), max(y+h, max_y)

l_con = cv2.rectangle(img_copy, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

# cv2.imshow('l_con:', l_con)
corners = [c1,c2,c3,c4]
print(corners)
print(max_x,max_y, min_x, min_y)

#r_con = np.array([[min_x,min_y],[max_x,min_y],[min_x,max_y],[max_x,max_y]], dtype=np.int32)
r_con = np.array(corners, dtype=np.int32)

# dummy_drawing = np.zeros([img.shape[1], img.shape[0]],np.uint8)
# for x in [min_x,max_x]:
#     for y in [min_y, max_y]:
#         dummy_drawing[x][y] = 1
#
# im_tempo, contours_I, hierarchy = cv2.findContours(dummy_drawing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# r_con_img = cv2.drawContours(img_copy, contours_I, -1, (128,255,0), 3)
# cv2.imshow('rect:', r_con_img)

rect = cv2.minAreaRect(r_con)
box = cv2.boxPoints(rect)
box = np.int0(box)
im = cv2.drawContours(img,[box],0,(0,0,255),2)
cv2.imshow('rect:', im)
cv2.waitKey(0)
c_sums = np.sum(corners, axis=1)

#max sum is lr and min sum is ul if (0,0) is top-left
lr = corners[np.argmax(c_sums)]
ul = corners[np.argmin(c_sums)]
corners= np.delete(corners, np.argmax(c_sums),0)
corners = np.delete(corners, np.argmin(c_sums),0)

#if A's y-value is larger than B's it must be upper and only ur is right
if corners[0][1] < corners[1][1]:
    ur = corners[0]
    ll = corners[1]
else:
    ur = corners[1]
    ll = corners[0]

corners_det = np.array([lr, ll, ul, ur], np.float32)

width = ur[0]-ul[0]
height = ll[1]-ul[1]

dest_img = np.array([
    [0,0],#ul
    [width,0],#ur
    [width,height],#lr
    [0,height]#ll
], np.float32)

# dest_img = np.array([
#     [0,0],#ul
#     [8000,0],#ur
#     [800,1000],#lr
#     [0,1000]#ll
# ], np.float32)

M = cv2.getPerspectiveTransform(corners_det, dest_img) #obtain transformation matrix
warp = cv2.warpPerspective(img_copy, M, (width, height))
cv2.imshow('warp',warp)
cv2.imwrite('testI.png', warp)



