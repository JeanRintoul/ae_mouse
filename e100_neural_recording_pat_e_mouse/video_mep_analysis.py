'''

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
import cv2
import numpy as np
from PIL import ImageGrab


def motion_detector_gist():

  #    
  framecount = 0 
  previous_frame = None
  cap = cv2.VideoCapture('D:\\mouse_aeti\\e100_neural_recording_pat_e_mouse\\IMG_9406.mov')
  total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
  print ('total_frames',total_frames)

  cap.set(cv2.CAP_PROP_FRAME_COUNT, 100)
  ret, frame = cap.read()

  while framecount < total_frames:
    # Load in the total frames.  
    cap.set(cv2.CAP_PROP_FRAME_COUNT, framecount)
    ret, frame = cap.read()
    framecount = framecount + 1 

    # 1. Load image; convert to RGB
    img_brg = np.array(frame)
    # print ('frame shape',img_brg.shape)  1080, 1920, 3
    img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)

    # 2. Prepare image; grayscale and blur
    prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

    # 2. Calculate the difference
    if (previous_frame is None):
      # First frame; there is no previous one yet
      previous_frame = prepared_frame
      continue

    # 3. calculate difference and update previous frame
    diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
    previous_frame = prepared_frame

    # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
    kernel = np.ones((5, 5))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)

    # 5. Only take different areas that are different enough (>20 / 255)
    thresh_frame = cv2.threshold(src=diff_frame, thresh=50, maxval=255, type=cv2.THRESH_BINARY)[1]

    # 6. Find and optionally draw contours
    contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # Comment below to stop drawing contours
    cv2.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    #  
    # Uncomment 6 lines below to stop drawing rectangles
    # for contour in contours:
    #   if cv2.contourArea(contour) < 50:
    #     # too small: skip!
    #       continue
    #   (x, y, w, h) = cv2.boundingRect(contour)
    #   cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
    #  
    cv2.imshow('Motion detector', img_rgb)
    #  
    if (cv2.waitKey(30) == 27):
      out.release()
      break
    #  
  # Cleanup
  cv2.destroyAllWindows()

if __name__ == "__main__":
  motion_detector_gist()