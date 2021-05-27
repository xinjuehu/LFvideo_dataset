import cv2
import numpy as np
from matplotlib import pyplot as plt

#Process single EPI
def Evaluate_EPI(EPI_file,Mask_Roi):
    EPI_dir = '.\\EPI_frames\\1\\'
    Depth_dir = '.\\Depth_frames\\'
    ePI_frames = EPI_dir+EPI_file + '.jpg'
    #Read EPI file    
    img_in = cv2.imread(ePI_frames)
    img_slice = img_in[20:25, :, :]
    img_gray = cv2.cvtColor(img_slice, cv2.COLOR_RGB2GRAY)
    img = cv2.medianBlur(img_gray,5)
    img = cv2.blur(img,(3,3),0)
    img = img_gray
    #Parameter    
    seg_len = 1 #Segment division length
    # thresh = 20 #Edge detection threshold
    #Sobel EPI detection
    # x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    # y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    # Scale_absX = cv2.convertScaleAbs(x)
    # Scale_absY = cv2.convertScaleAbs(y)
    # result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
    # cv2.imwrite('sobel.jpg', result)
    # mask_roi = np.ones((result.shape), dtype="uint8")
    # # # #Aggregate block from segments
    seg_num = img_gray.shape[1] // seg_len
    # for k in range(seg_num):
    #     mean_edge = np.mean(result[:,k*seg_len:(k+1)*seg_len])
    #     if mean_edge>thresh:
    #         mean_edge = 255
    #     else:
    #         mean_edge = 0
    #     mask_roi[:,k*seg_len:(k+1)*seg_len] = mean_edge*mask_roi[:,k*seg_len:(k+1)*seg_len]

    mask_roi = Mask_Roi

    # mask_inv = cv2.bitwise_not(mask_roi)
    # image=cv2.bitwise_and(img_slice,img_slice,mask = mask_roi)
    # image_not=cv2.bitwise_and(img_slice,img_slice,mask = mask_inv)
    # plt.figure()
    # plt.imshow(img_in)
    # plt.figure()
    # plt.imshow(img_slice)
    # plt.figure()
    # plt.imshow(mask_roi)
    # plt.figure()
    # plt.imshow(mask_inv)
    # plt.show()
    
    #Block seperation & Build block table
    block_table = np.array([[0,0]])
    previous_block = mask_roi[0,0]
    edge_trigger = 0
    for k in range(seg_num):
        positive_block = mask_roi[0,k*seg_len]
        if (positive_block != previous_block) and (edge_trigger == 0):
            block_table[-1, 1] = (k-1)*seg_len # - 1
            add_block = np.array([[(k-1)*seg_len, 0]])
            block_table =np.append(block_table, add_block,axis=0)
            edge_trigger = 1
            previous_block = positive_block
            continue
        if (positive_block != previous_block) and (edge_trigger == 1):
            edge_trigger = 0
            previous_block = positive_block
            continue
        previous_block = positive_block
    block_table[-1,1] = img_gray.shape[1]-1
    block_num = block_table.shape[0]

    # showblock = 0*img_slice[:,:,0]
    # showblock[:,block_table[4,0]:block_table[4,1]] = 1 + showblock[:,block_table[4,0]:block_table[4,1]]
    # print(block_table)
    # plt.figure()
    # plt.imshow(img_in)
    # plt.figure()
    # plt.imshow(mask_roi)
    # plt.figure()
    # plt.imshow(img_slice[:,:,:])
    # plt.figure()
    # plt.imshow(showblock)
    # plt.show()

    block_depth = np.zeros((mask_roi.shape))
    for block_id in range(block_num):
        depth_value = Match_block(block_id, block_table, img_slice, img_in)
        block_depth[:, block_table[block_id,0]:block_table[block_id,1]+1] = depth_value
    
    # block_depth_fig = block_depth.astype('uint8')

    Depth_file = Depth_dir + EPI_file + '.npy'
    np.save(Depth_file, block_depth)

#Match block in all lines
def Match_block(block_id, BLOCK_TABLE, IMG_SLICE, IMG_IN):
    block_table = BLOCK_TABLE
    img_slice = IMG_SLICE
    img_in = IMG_IN
    img_in_len = img_in.shape[1]
    search_area_end = block_table[block_id,1] + 300
    search_area_sta = block_table[block_id,0] - 300
 
    
    templ_block = img_slice[:, block_table[block_id,0]:block_table[block_id,1]+1, :] #template block
    w = templ_block.shape[1]
    h = templ_block.shape[0]
    #Search in all lines (use 6 methods)
    line_result = np.zeros([1,2])
    for line_indx in range(10):
        search_sample_len = search_area_end - search_area_sta
        search_sample = np.zeros([5,search_sample_len,3])
        if (search_area_sta <0) and (search_area_end < img_in_len):
            for s in range(-search_area_sta):
                search_sample[:,s,:] = img_in[line_indx*5:(line_indx+1)*5,  0, :]
            search_sample[:,-search_area_sta-1:-1,:] = img_in[line_indx*5:(line_indx+1)*5,  0:search_area_end, :]

        if (search_area_sta >0) and (search_area_end < img_in_len):
            search_sample = img_in[line_indx*5:(line_indx+1)*5,  search_area_sta:search_area_end, :]
            search_sample = search_sample.astype('uint8')

        if (search_area_sta <0) and (search_area_end > img_in_len):
            for s in range(-search_area_sta):
                search_sample[:,s,:] = img_in[line_indx*5:(line_indx+1)*5,  0, :]
            search_sample[:,-search_area_sta:img_in_len-search_area_sta,:] = img_in[line_indx*5:(line_indx+1)*5,  0:search_area_end, :]
            for t in range(search_area_end - img_in_len):
                search_sample[:,t+img_in_len-search_area_sta,:] = img_in[line_indx*5:(line_indx+1)*5,  -1, :]
            
        if (search_area_sta >0) and (search_area_end > img_in_len):
            search_sample[:,0:img_in_len-search_area_sta-1,:] = img_in[line_indx*5:(line_indx+1)*5,  search_area_sta:-1, :]
            for t in range(search_area_end - img_in_len):
                search_sample[:,t+img_in_len-search_area_sta,:] = img_in[line_indx*5:(line_indx+1)*5,  -1, :]
            # search_sample = search_sample.astype('uint8')
            # plt.figure()
            # plt.imshow(search_sample)
            # plt.show()

        search_sample = search_sample.astype('uint8')

        # plt.figure()
        # plt.imshow(search_sample)
        # plt.figure()
        # plt.imshow(templ_block)
        

        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        # methods = ['cv2.TM_SQDIFF_NORMED']
        candiate_block_start = np.zeros([6,1])
        candiate_block_end = np.zeros([6,1])
        ind_meth = 0
        for meth in methods:
            search_img = search_sample.copy()
            method = eval(meth)
            res = cv2.matchTemplate(search_img, templ_block, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(search_img, top_left, bottom_right, 255, 2)
            candiate_block_start[ind_meth] = top_left[0]
            candiate_block_end[ind_meth] = bottom_right[0]
            ind_meth = ind_meth + 1
        #Find the mode in the result set of all methods            
        candiate_block_start = candiate_block_start[:,0]
        candiate_block_start = candiate_block_start.astype(int)
        candiate_block_end = candiate_block_end[:,0]
        candiate_block_end = candiate_block_end.astype(int)
        # counts1 = np.bincount(candiate_block_start)
        # counts2 = np.bincount(candiate_block_end)
        # mode_start = np.argmax(counts1)
        # mode_end = np.argmax(counts2)
        mode_start = np.median(candiate_block_start)
        mode_end = np.median(candiate_block_end)
        # if (line_indx<4 and mode_start>block_table[block_id,0]) or (line_indx>4 and mode_start<block_table[block_id,0]):
        #     continue
        add_line_result = [[line_indx, (mode_start+mode_end)/2]]

        # print(candiate_block_start)
        # print([mode_start,mode_end])
        # plt.figure()
        # plt.imshow(search_sample[:,int(mode_start):int(mode_end),:])
        # plt.show()

        line_result = np.vstack((line_result, add_line_result))
    line_result = np.delete(line_result, 0, axis=0)

    #First-order regression
    x = line_result[:, 0]
    y = line_result[:, 1]
    y.sort()
    curve = np.polyfit(x, y, deg= 1)
    depth_value = curve[0]
    return depth_value




if __name__ == "__main__":
    central_view = cv2.imread('./Matrix_frames/1/04_04.jpg')
    blurred = cv2.GaussianBlur(central_view,(11,11),0) #15 11
    gaussImg = cv2.Canny(blurred, 10,50) #(30,150) (10,50)
    plt.figure()
    plt.imshow(gaussImg)
    plt.show()

    for i in range(210):
        EPI_name = '04_' + '%04d'%(i*5)
        mask_roi = gaussImg[i*5:(i+1)*5,:]
        Evaluate_EPI(EPI_name, mask_roi)
    # i = 105
    # mask_roi = gaussImg[i*5:(i+1)*5,:]
    # EPI_name = '04_' + '%04d'%(i*5)
    # Evaluate_EPI(EPI_name, mask_roi)