import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_local

# Resize image
def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


# approximate the contour by a more primitive polygon shape
def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.032 * peri, True)

def get_receipt_contour(contours):
    # loop over the contours
    for c in contours:
        approx = approximate_contour(c)
        # if our approximated contour has four points, we can assume it is receipt's rectangle
        if len(approx) == 4:
            return approx


def contour_to_rect(contour,original):
    resize_ratio = 500 / original.shape[0]

    if contour is None:
      print("ZJEBALO SIE COS ")
      return 1

    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    # top-left point has the smallest sum
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # compute the difference between the points:
    # the top-right will have the minumum difference
    # the bottom-left will have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect / resize_ratio


def wrap_perspective(img, rect):
    # unpack rectangle points: top left, top right, bottom right, bottom left
    (tl, tr, br, bl) = rect
    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    # destination points which will be used to map the screen to a "scanned" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    # warp the perspective to grab the screen
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))


def bw_scanner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 21, offset = 5, method = "gaussian")
    return (gray > T).astype("uint8") * 255


def contours_and_wrap_perspective(closed_image,original,idx):

  #Detect edges with Canny
  edged = cv2.Canny(closed_image, 90, 150, apertureSize=3)
  # plot_gray(edged)

  # Detect all contours in Canny-edged image
  contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  image_with_contours = cv2.drawContours(original.copy(), contours, -1, (0,255,0), 3)
  # plot_rgb(image_with_contours)

  # Get 10 largest contours
  largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
  image_with_largest_contours = cv2.drawContours(original.copy(), largest_contours, -1, (0,255,0), 3)

  print("Wyodrebnienie jedynie kontorów paragonu",idx)

  #Wyodrebnienie jedynie kontorów paragonu
  receipt_contour = get_receipt_contour(largest_contours)
  # image_with_receipt_contour = cv2.drawContours(original.copy(), [receipt_contour], -1, (0, 255, 0), 2)
  # plot_rgb(image_with_receipt_contour)

  print("Perspektywa lotu ptaka",idx)

  #Perspektywa lotu ptaka
  scanned = wrap_perspective(original.copy(), contour_to_rect(receipt_contour,original))
  plt.figure(figsize=(16,10))
  # plt.imshow(scanned)
  cv2.imwrite(f"/content/drive/MyDrive/img_kerf_warp_persp/k_wrap{idx}.jpg",scanned)

  print("Przejście na czarno-biały",idx)

  #Przejście na czarno-biały
  result = bw_scanner(scanned)
  cv2.imwrite(f"/content/drive/MyDrive/img_kref_scan/k_scan{idx}.jpg",result)



def img_to_wrap_perspective2(image_filepath,idx=1):

  image = cv2.imread(image_filepath)

  # Downscale image as finding receipt contour is more efficient on a small image
  resize_ratio = 500 / image.shape[0]
  original = image.copy()
  image = opencv_resize(image, resize_ratio)

  #Change image to gray scale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # plot_gray(gray)

  #Delete noise
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  # plot_gray(blurred)

  # Detect white regions
  rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (22,22))
  dilated = cv2.dilate(blurred, rectKernel)
  # plot_gray(dilated)

  contours_and_wrap_perspective(dilated,original,idx)