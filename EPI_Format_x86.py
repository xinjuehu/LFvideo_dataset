import cv2
import os
import shutil
import numpy as np


def dec_VP(Frames_num, VP_Frames_dir, Video_dir):
    ## Decode LF frames by VP
    VP_num = len(os.listdir(Video_dir))
    for i in range(VP_num):
        VP_name = VP_Frames_dir + str(i+1) + '\\'
        if os.path.exists(VP_name):
            shutil.rmtree(VP_name)
        os.mkdir(VP_name)
        Video_name = Video_dir + str(i+1) + '.MP4'
        cmd_line_1 = 'ffmpeg -i ' + Video_name + ' -vf "select=between(n\,' + str(0) + '\,' + str(Frames_num) + ')" -vsync 0 ' + VP_name + '%04d.jpg'
        os.system(cmd_line_1)

def reshuf(Frames_num_arg, VP_Frames_dir, Matrix_frames, Col_num):
    ## Reshuffle as Matrix
    Frames_num = Frames_num_arg + 1
    VP_num = len(os.listdir(VP_Frames_dir))
    for i in range(Frames_num):
        Matrix_frame_name = Matrix_frames + str(i+1) + '\\'
        if os.path.exists(Matrix_frame_name):
            shutil.rmtree(Matrix_frame_name)
        os.mkdir(Matrix_frame_name)
        for j in range(VP_num):
            source_file_name = VP_Frames_dir + str(j+1) + '\\' + '%04d'%(i+1) + '.jpg'
            matrix_row = j // Col_num
            matrix_col = j %  Col_num
            dest_file_name = Matrix_frame_name + '%02d_%02d'%(matrix_row, matrix_col) + '.jpg'
            shutil.copyfile(source_file_name, dest_file_name)

def genEPI(Frames_num_arg, Matrix_frames, EPI_frames, Col_num):
    #generate EPI by frames
    Frames_num = Frames_num_arg
    for i in range(Frames_num):
        EPI_frame_name = EPI_frames + str(i+1) + '\\'
        if os.path.exists(EPI_frame_name):
            shutil.rmtree(EPI_frame_name)
        os.mkdir(EPI_frame_name)
        Matrix_frame_name = Matrix_frames + str(i+1) + '\\'
        VP_num = len(os.listdir(Matrix_frame_name))
        [height, width, channel] = cv2.imread(Matrix_frame_name + os.listdir(Matrix_frame_name)[0]).shape
        row = VP_num // Col_num
        col = Col_num
        img_matrix_R = np.zeros((row, col, height, width))
        img_matrix_G = np.zeros((row, col, height, width))
        img_matrix_B = np.zeros((row, col, height, width))
        for j in range(VP_num):
            matrix_row = j // Col_num
            matrix_col = j %  Col_num
            raw_img_file = Matrix_frame_name + '%02d_%02d'%(matrix_row, matrix_col) + '.jpg'
            img_matrix_R[matrix_row, matrix_col, :, :] = cv2.imread(raw_img_file)[:,:,0]
            img_matrix_G[matrix_row, matrix_col, :, :] = cv2.imread(raw_img_file)[:,:,1]
            img_matrix_B[matrix_row, matrix_col, :, :] = cv2.imread(raw_img_file)[:,:,2]
        
        h_5 = height // 5
        for h in range(h_5):
            for r in range(row):
                EPI_img_hor = np.zeros((col*5, width, channel))
                EPI_img_hor[:,:,0] = img_matrix_R[r, :, h*5:(h+1)*5, :].reshape((col*5, width))
                EPI_img_hor[:,:,1] = img_matrix_G[r, :, h*5:(h+1)*5, :].reshape((col*5, width))
                EPI_img_hor[:,:,2] = img_matrix_B[r, :, h*5:(h+1)*5, :].reshape((col*5, width))
                EPI_img_file = EPI_frame_name + '%02d_%04d'%(r, h*5) + '.jpg'
                cv2.imwrite(EPI_img_file, EPI_img_hor)






if __name__ == "__main__":
    frames_num = 1
    col_num = 10
    vP_Frames_dir = '.\\VP_frames\\'
    video_dir = '.\\dynamic_50\\'
    matrix_frames = '.\\Matrix_frames\\'
    ePI_frames = '.\\EPI_frames\\'
    dec_VP(frames_num, vP_Frames_dir, video_dir)
    reshuf(frames_num, vP_Frames_dir, matrix_frames, col_num)
    genEPI(frames_num, matrix_frames, ePI_frames, col_num)