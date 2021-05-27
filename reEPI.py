import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage
from skimage import segmentation, color
from skimage.future import graph




def normalization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    data[np.where(data<mu-3*sigma)] = mu-3*sigma
    data[np.where(data>mu+3*sigma)] = mu+3*sigma
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

if __name__ == "__main__":
    mask_matrix = np.zeros([1056,1920])
    a = 0
    b = 0.5
    for i in range(1056):
        col_base = a*np.abs((i - 728)/728) + b
        for j in range(1920):
            # mask_matrix[i,j] = (0.2*np.abs((i - 528)/528) + 0.5) * (0.5-0.2*np.abs((j-960)/960))
            mask_matrix[i,j] = col_base-(a+b-col_base)*np.abs((j-960)/960)
    mask_matrix = mask_matrix[0:-70, :]
    depth_map_hor = np.zeros([1056,1920])
    raw_map = np.zeros([1056,1920,3])
    for i in range(209):
        depth_name = '.\\Depth_frames\\' + '04_' + '%04d'%(i*5) + '.npy'
        block_depth =  np.load(depth_name)
        raw_name = '.\\EPI_frames\\1\\' + '04_' + '%04d'%(i*5) + '.jpg'
        img_raw = cv2.imread(raw_name)
        raw_map[i*5:(i+1)*5,:,:] = img_raw[20:25, :, :]
        depth_map_hor[i*5:(i+1)*5,:] = block_depth
    depth_map_hor = 255*normalization(depth_map_hor[0:-70, :])
    
    depth_map_ver = np.zeros([1056,1920])
    for i in range(383):
        depth_name = '.\\Depth_frames_ver\\' + '04_' + '%04d'%(i*5) + '.npy'
        block_depth =  np.load(depth_name)
        depth_map_ver[:,i*5:(i+1)*5] = np.transpose(block_depth, axes=[1, 0])
    depth_map_ver = 255*normalization(depth_map_ver[0:-70, :])

    depth_map_hor = depth_map_hor.astype('uint8')
    depth_map_ver = depth_map_ver.astype('uint8')
    depth_map_ver = cv2.medianBlur(depth_map_ver, 3)
    depth_map_hor = cv2.medianBlur(depth_map_hor, 3)
    
    depth_map = mask_matrix * depth_map_hor + (1-mask_matrix) * depth_map_ver

    depth_map = 255*normalization(depth_map)
    cv2.imwrite('depth_map.jpg', depth_map)

    # depth_map = cv2.GaussianBlur(depth_map,(15,15),0)
    # depth_map = depth_map.astype('uint8')
    # depth_map = cv2.medianBlur(depth_map, 3)

    img = cv2.imread('./Matrix_frames/1/04_04.jpg')
    img = img[0:-70, :, :]
    labels1 = segmentation.slic(img, compactness=2, n_segments=5000)
    g = graph.rag_mean_color(img, labels1, connectivity=2, mode='similarity',sigma= 50)
    labels2 = graph.cut_normalized(labels1, g)
    out1 = color.label2rgb(labels1, depth_map, kind='avg')
    out1 = out1.astype('uint8')
    out1_norm = 255*cv2.normalize(out1[:,:,0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite('depth_out1.jpg',out1_norm)
    # area_num = np.max(labels2)+1
    # out2 = np.zeros(depth_map.shape)
    # for i in range(area_num):
    #     out2[labels2==i] = np.median(depth_map[labels2==i])
    area_num = np.max(labels2)+1
    out2 = np.zeros(depth_map.shape)
    for i in range(area_num):
        out2[labels2==i] = np.median(depth_map[labels2==i])

    # out2 = 255*normalization(out2)
    out2 = out2.astype('uint8')


    # out2 = color.label2rgb(labels2, depth_map, kind='avg')
    # out2 = out2.astype('uint8')
    out2_norm = 255*cv2.normalize(out2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite('depth_out2.jpg',out2_norm)

    # depth_map = cv2.medianBlur(depth_map, 11)
    # depth_map = cv2.medianBlur(depth_map, 11)
    # depth_map = cv2.blur(depth_map, (50,50))
    # depth_map = depth_map.astype('uint8')
    raw_map = raw_map.astype('uint8')
    # lsc = cv2.ximgproc.createSuperpixelLSC(raw_map, region_size = 10)
    # lsc.iterate(100)
    # lsc.enforceLabelConnectivity(min_element_size = 20)
    # lsc = cv2.ximgproc.createSuperpixelSEEDS(raw_map.shape[1],raw_map.shape[0],raw_map.shape[2],2000,5,3,5,True)
    # lsc.iterate(raw_map,100)
    # lsc = cv2.ximgproc.createSuperpixelSLIC(raw_map,region_size=15,ruler = 20.0) 
    # lsc.iterate(100)


    # depth_map_new = np.zeros(depth_map.shape)
    # mask_lsc = lsc.getLabelContourMask()
    # label_lsc = lsc.getLabels()
    # number_lsc = lsc.getNumberOfSuperpixels()
    # print(number_lsc)
    # img_gray = cv2.cvtColor(raw_map, cv2.COLOR_RGB2GRAY)
    # thresh = 10
    # area_indx = 0
    # new_label_lsc = np.zeros(label_lsc.shape)
    # for sp in range(number_lsc-1):
    #     next_sp = sp + 1
    #     sp_cord = np.where(label_lsc == sp)
    #     next_sp_cord = np.where(label_lsc == next_sp)
    #     if np.abs(np.mean(img_gray[sp_cord]) - np.mean(img_gray[next_sp_cord])) > thresh:
    #         area_indx = area_indx + 1
    #     new_label_lsc[next_sp_cord] = area_indx
    # new_number_lsc = np.max(new_label_lsc)
    # new_label_lsc = new_label_lsc.astype(int)
    # new_number_lsc = new_number_lsc.astype(int)
    # print(new_number_lsc)



    # for sp in range(new_number_lsc):
    #     sp_cord = np.where(new_label_lsc == sp)
    #     depth_map_new[sp_cord] = np.median(depth_map[sp_cord])

    # depth_map_new = depth_map_new.astype('uint8')
    # depth_map_new = cv2.medianBlur(depth_map_new, 11)
    # # depth_map_new = cv2.blur(depth_map_new, (5,5))
    # depth_map_new = depth_map_new.astype('uint8')





    # mask_inv_slic = cv2.bitwise_not(mask_lsc)  
    # img_slic = cv2.bitwise_and(raw_map,raw_map,mask =  mask_inv_slic)

    cv2.imwrite('depth_map_hor.jpg', depth_map_hor)
    cv2.imwrite('depth_map_ver.jpg', depth_map_ver)
    cv2.imwrite('raw_map.jpg', raw_map)
    
    #Block Maching
    imgL = cv2.imread('./Matrix_frames/1/04_03.jpg',0)
    imgR = cv2.imread('./Matrix_frames/1/04_04.jpg',0)
    stereo = cv2.StereoBM_create(numDisparities=160, blockSize=21)
    disparity = stereo.compute(imgL,imgR)

    #Semi-Global Block Matching
    window_size = 3
    min_disp = 0
    num_disp = 320 - min_disp

    stereo2 = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=240,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=3,
        P1=8 * 3 * window_size ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity2 = stereo2.compute(imgL, imgR)

    cv2.imwrite('BM.jpg', disparity)
    cv2.imwrite('SGBM.jpg', disparity2)


    plt.figure()
    plt.subplot(221)
    plt.imshow(raw_map)
    plt.subplot(222)
    plt.imshow(out2_norm)
    plt.subplot(223)
    plt.imshow(disparity2)
    plt.subplot(224)
    plt.imshow(out1_norm)
    plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()