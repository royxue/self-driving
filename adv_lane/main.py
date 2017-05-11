import cv2
import glob
import numpy as np

from moviepy.editor import VideoFileClip
from calibration import *
from curvature import *
from search import *
from threshold import cal_thresh_img
from vis import vis
from warp import warper
from line import Line

left_line, right_line = Line(5), Line(5)
detected = False
objp, imgp = cal_calib_points('./camera_cal/calibration*.jpg')

src = np.float32([[200, 720], [1100, 720], [595, 450], [685, 450]])
dst = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])
n = 0

def process(image):
    global left_line, right_line, detected

    undist = cal_undistort(image, objp, imgp)

    thresh_s = (170, 255)
    thresh_sx = (50, 255)
    thresh_sy = (50, 255)
    thresh_mag = (50, 255)
    thresh_gdir = (0.7, 1.3)
    com = cal_thresh_img(undist, thresh_s, thresh_sx, thresh_sy, thresh_mag, thresh_gdir)
    warped = warper(com, src, dst)

    if not detected:
        fit = sliding_window_search(warped)
        if fit != None:
            left_line.add_fit(fit[0])
            right_line.add_fit(fit[1])
            detected = True
    else:
        fit = sliding_window_search_with_known(warped, left_line.get_fit(), right_line.get_fit())
        if fit != None:
            left_line.add_fit(fit[0])
            right_line.add_fit(fit[1])
        else:
            detected = False

    left_fit, right_fit = left_line.get_fit(), right_line.get_fit()
    res = vis(undist, warped, left_fit, right_fit, src, dst)

    left_c, right_c = cal_curvature(warped, left_fit, right_fit)
    offset = cal_offset(undist, left_fit, right_fit)

    avg_c = (left_c + right_c)/2
    label_c = 'Radius of curvature: %.2f m' % (avg_c)
    offset_c = 'Offset from the center of lane: %.2f m' % (offset)

    cv2.putText(res, label_c, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(res, offset_c, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    cv2.imwrite('./res.jpg', res)

    return res

def output_each_step():
    images = glob.glob('./test_images/*.jpg')
    objp, imgp = cal_calib_points('./camera_cal/calibration*.jpg')
    output_path = './output_images/'

    for img in images:
        print('Processing' + img)
        img_name = img.split('/')[-1]
        image = cv2.imread(img)

        undist = cal_undistort(image, objp, imgp)
        cv2.imwrite(output_path+'undist_'+img_name, undist)

        thresh_s = (170, 255)
        thresh_sx = (30, 255)
        thresh_sy = (30, 255)
        thresh_mag = (50, 255)
        thresh_gdir = (0.7, 1.3)
        com = cal_thresh_img(undist, thresh_s, thresh_sx, thresh_sy, thresh_mag, thresh_gdir)
        cv2.imwrite(output_path+'thresh_'+img_name, com*255)

        src = np.float32([[200, 720], [1100, 720], [595, 450], [685, 450]])
        dst = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])
        warped = warper(com, src, dst)
        cv2.imwrite(output_path+'warped_'+img_name, warped*255)

        fit_name = output_path+'fit_'+img_name
        left_fit, right_fit = sliding_window_search(warped, vis=True, file_name=fit_name)

        out_img = np.dstack((warped, warped, warped))*255
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        res = vis(undist, warped, left_fit, right_fit, src, dst)
        cv2.imwrite(output_path+'final_'+img_name,res)


def output_video(filepath):
    video = VideoFileClip(filepath).subclip(40, 44)
    output = video.fl_image(process)
    output.write_videofile('./out_test.mp4', audio=False)

def main():
    output_video('./project_video.mp4')
    # output_each_step()

if __name__ == "__main__":
    main()
