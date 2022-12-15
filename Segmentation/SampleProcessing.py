import cv2 as cv
import imutils
import numpy as np
import os
import time
from tqdm import tqdm
import math

classNames = ["arborio", "basmati", "brown", "jasmin", "parboiled"]

for className in tqdm(classNames):
    inputDirectory = "input/"+className+"/"
    outputDirectory = "Dataset/"

    backgroundImage = "background.bin"
    path_rgb_samples =  outputDirectory+"RGB/"+className+"/"
    path_raw_samples =  outputDirectory+"RAW/"+className+"/"
    path_packed_raw_samples = outputDirectory+"PackedRAW/"+className+"/"

    #Specified parameters for the operations in the processing pipeline
    blur_kernel_size = 5
    threshold_value = 200 
    output_image_resolution = 40
    raw_matrix_width = 14308
    raw_matrix_height = 10760
    contour_area_min_threshold = 50
    contour_area_max_threshold = 2000 
    elo_percentage = 0.7 

    #Define functions
    def elongation(m):
        x = m['mu20'] + m['mu02']
        y = 4 * m['mu11']**2 + (m['mu20'] - m['mu02'])**2
        return (x + y**0.5) / (x - y**0.5)

    for filename in os.listdir(inputDirectory+"bin/"):
        fileName = os.path.splitext(filename)[0]
        workingImage = "bin/"+fileName+".bin"
        workingJPG = "jpg/"+fileName+".png"
        background_raw = np.fromfile(backgroundImage, dtype=np.uint16)
        background_raw= np.reshape(background_raw,(raw_matrix_height,raw_matrix_width))
        rice_raw = np.fromfile(inputDirectory + workingImage, dtype=np.uint16)
        rice_raw= np.reshape(rice_raw,(raw_matrix_height,raw_matrix_width))
        rgbP1 = cv.imread(inputDirectory+workingJPG)

        #Calculate offset for JPEG
        x_offset = int(raw_matrix_width - rgbP1.shape[1])-2
        y_offset = int(raw_matrix_height - rgbP1.shape[0])-1
        
        select_points_y_1 = 3000 + x_offset
        select_points_y_2 = 6000 + x_offset
        select_points_x_1 = 5000 + y_offset
        select_points_x_2 = 9200 + y_offset

        #Subtract background from rice
        rice_raw_cropped = rice_raw[select_points_y_1:select_points_y_2, select_points_x_1:select_points_x_2]
        jpeg_cropped = rgbP1[select_points_y_1-y_offset:select_points_y_2 - y_offset, select_points_x_1 - x_offset:select_points_x_2 - x_offset]
        background_raw_cropped = background_raw[select_points_y_1:select_points_y_2, select_points_x_1:select_points_x_2]
        difference = cv.subtract(rice_raw_cropped,background_raw_cropped)

        #Apply median blur
        blur = cv.medianBlur(difference, blur_kernel_size)

        #Apply thresholding
        (T, threshOut) = cv.threshold(blur, threshold_value, 65535,
        cv.THRESH_BINARY)
        
        #Find contours in the thresholded image
        threshOut8bit = threshOut.astype(np.uint8)
        cnts = cv.findContours(threshOut8bit.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)
        print("[INFO] {} unique contours found".format(len(cnts)))
        
        #Find center of objects
        center = [ ]
        
        for i in cnts:
            if cv.contourArea(i) > contour_area_min_threshold and cv.contourArea(i) < contour_area_max_threshold:
                M = cv.moments(i)
                if M['m00'] != 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        center.append((cx, cy))
                        
        sampleCount = 0
        elongationValues = []
        for list in center:

            h = int(output_image_resolution/2)
            w = int(output_image_resolution/2)
            
            #Binary image cropping
            x = int(list[0])
            y = int(list[1])
            
            #Contour check
            crop_img = threshOut8bit[y-h:y+h, x-w:x+w] 
            cropped_cnts = cv.findContours(crop_img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            cropped_cnts = imutils.grab_contours(cropped_cnts) 
            
            #If we only have one countour in the cropped images
            if len(cropped_cnts) == 1:
                
                #Calculate elongation value
                elo = int(elongation(cv.moments(crop_img)))
                elongationValues.append(elo)
                
        #For any images not equal to average, delete
        elo_mean = np.mean(elongationValues)
        threshold_elongation = elo_percentage*elo_mean
        
        #We're running it again.
        for list in center:
            h = int(output_image_resolution/2)
            w = int(output_image_resolution/2)
            x = int(list[0])
            y = int(list[1])
            
            if (y-h) % 2 != 0 :
                y+=1
            if (x-w) % 2 != 0:
                x+=1
            
            #Contour check
            crop_img = threshOut8bit[y-h:y+h, x-w:x+w] #
            crop_img_jpg = jpeg_cropped[y-h:y+h, x-w:x+w]
            crop_img_raw = rice_raw_cropped[y-h:y+h, x-w:x+w]
            cropped_cnts = cv.findContours(crop_img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            cropped_cnts = imutils.grab_contours(cropped_cnts)
            
            crop_img_packed_raw = np.zeros((int(crop_img_raw.shape[0]/2),int(crop_img_raw.shape[1]/2),4), dtype=crop_img_raw.dtype)
            
            
            #Bayer pattern used for samples
            packedSample = np.zeros((crop_img.shape[0],crop_img.shape[1],3), dtype=np.uint8)
            
            if len(cropped_cnts) == 1:
                #Every new sample, reinit the R,G1,B,G2
                #Stores Pixel values from sample
                R,G1,B,G2 = [],[],[],[]
            
                
                
                #Calculate elongation value
                elo = elongation(cv.moments(crop_img))
                if (elo >  elo_mean - threshold_elongation) and (elo < elo_mean + threshold_elongation):
                    
                    if (crop_img.shape[0]==crop_img.shape[1]==output_image_resolution):
                        
                        #For all repeating patterns og RG/BR, store separate values in different arrays
                        #             E           O      (j)
                        # (i)    E   ee(R)       eo(G1)
                        #        O   oe(G2)      oo(B)

                        #Nested for-loop that goes through each pixel in the cropped RAW sample to store the values in the packed RAW sample
                        for i in range (len(crop_img_raw)):
                            for j in range (len(crop_img_raw[0])):
                                px_value = crop_img_raw[i][j]
                                #ee
                                if i % 2 == 0 and j % 2 == 0:
                                    #store red samples for RAW
                                    R.append(px_value)
                                #eo
                                elif i % 2 == 0 and j % 2 != 0:
                                    #store green1 samples for RAW
                                    G1.append(px_value)
                                #oo
                                elif i % 2 != 0 and j % 2 != 0:
                                    #store blue samples for RAW
                                    B.append(px_value)
                                #oe
                                elif i % 2 != 0 and j % 2 == 0: 
                                    #store green2 samples for RAW
                                    G2.append(px_value)

                        #Inserting the gathered color information pixel within the (RGBG) respective color channel 
                        crop_img_packed_raw[:,:,0] = np.array(R).reshape((20,20))
                        crop_img_packed_raw[:,:,1] = np.array(G1).reshape((20,20))
                        crop_img_packed_raw[:,:,2] = np.array(B).reshape((20,20))
                        crop_img_packed_raw[:,:,3] = np.array(G2).reshape((20,20))
                        
                        

                        #Cropping and storing original RAW files and packed RAW files as numpy array files.
                        np.save(os.path.join(path_raw_samples, 'sample_{}.npy'.format(fileName+"_"+str(sampleCount))), crop_img_raw)
                        np.save(os.path.join(path_packed_raw_samples, 'sample_{}.npy'.format(fileName+"_"+str(sampleCount))), crop_img_packed_raw)
                    
                        #Cropping RGB sample from JPEG file and storing as JPG
                        cv.imwrite((os.path.join(path_rgb_samples, 'sample_{}.jpg'.format(fileName+"_"+str(sampleCount)))), crop_img_jpg)
                    
                        sampleCount +=1

        #Print amount of samples per current processed high resolution capture
        print("[INFO] Stored amount of samples after filtering: "+ str(sampleCount))
