import sqlite3
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from datetime import datetime


CAMERA_WIDTH = 768
CAMERA_HEIGHT = 1088
FRAME_RATE = 32
H_FLIP = False
V_FLIP = False
SHARPNESS = 0
RESIZE_WIDTH = 512
RESIZE_HEIGHT = 624
STARTPOINT_X = 128
STARTPOINT_Y = 140
ENDPOINT_X = 660
ENDPOINT_Y = 890
CONTRAST = 30
KPNUMBER = 2000
RECOGRATE = 0.0375
BURSTCOUNT = 20
THRESHOLD = 15
INTERVAL = 1

conn = sqlite3.connect('mainData.db')  # create connection with the database file with variable conn
c = conn.cursor()  # cursor with variable c allows to execute sql commands with execute method

# Create Table -> uncomment the statements below when the database file is deleted to create a new one.
'''c.execute("""
     CREATE TABLE mainTable (
     person_id TEXT,
     first_name TEXT,
     last_name TEXT,
     image_name TEXT,
     image_data BLOB,
     descriptor TEXT)""")'''

# Delete Table -> uncomment below if you want to delete all items in database, and then comment it again after deleting.
#c.execute("DELETE FROM mainTable")

c.execute("SELECT person_id, first_name, last_name, descriptor FROM mainTable")
mainTable_copy = c.fetchall()  # fetches the data to the variable result which returns a list
print("database uploaded to memory. size=", len(mainTable_copy), "type:", type(mainTable_copy))

MATCHVALUE = KPNUMBER*RECOGRATE

# def captureImage(camera, width, height, framerate, hflip, vflip):
#      # Grab a reference to the raw cam capture
#     if width <= 0 or width > 3280 or height <= 0 or height > 2464:
#         print('Invalid resolution values.')
#         return
#     print("Camera is now set. Please place your tongue in front of the camera.")
#     print("Press c to capture. Press q to quit.")
#     camera.resolution = (width, height)
#     camera.framerate = framerate
#     camera.vflip = vflip
#     camera.hflip = hflip
#     rawCapture = PiRGBArray(camera, size=(width, height))
#
#     # Start coordinate, here (5, 5)
#     # represents the top left corner of rectangle
#     start_point = (STARTPOINT_X, STARTPOINT_Y)
#     # Ending coordinate, here (220, 220)
#     # represents the bottom right corner of rectangle
#     end_point = (ENDPOINT_X, ENDPOINT_Y)
#     # Blue color in BGR
#     color = (255, 0, 0)
#     # Line thickness of 2 px
#     thickness = 2
#
#     # allow the camera to warmup
#     time.sleep(0.1)
#
#     result = ""
#
#     # capture frames from the camera
#     for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
#         # grab the raw NumPy array representing the image,
#         # then initialize the timestamp
#         # and occupied/unoccupied text
#         image = frame.array
#
#         image = cv2.rectangle(image, start_point, end_point, color, thickness)
#
#         # show the frame
#         cv2. imshow("Frame", image)
#         key = cv2.waitKey(1) & 0xFF
#
#         # clear the stream in preparation for the next frame
#         rawCapture.truncate(0)
#
#         # if the q key was pressed, break from the loop
#         if key == ord("q"):
#             break
#         elif key == ord("c") or key == ord("C"):
#             key2 = int(input("Enter 1 to accept, 2 to try again, 0 to exit: "))
#             if key2 == 1:
#                 now = datetime.now() # current date and time
#                 date_time = now.strftime("%m%d%Y-%H%M%S")
#                 filename = date_time + ".jpg"
#                 cv2.imwrite(filename, image)
#                 result = filename
#                 break
#
#             elif key2 == 2:
#                 continue
#
#             elif key2 == 0:
#                 break
#
#     #cv2.destroyAllWindows()
#     return result

def get_thirty(camera, width, height, framerate, hflip, vflip):
    X_data = []
    
    # Grab a reference to the raw cam capture
    if width <= 0 or width > 3280 or height <= 0 or height > 2464:
        print('Invalid resolution values.')
        return
    print("Camera is now set. Please place your tongue in front of the camera.")
    print("Press c to capture. Press q to quit.")
    camera.resolution = (width, height)
    camera.framerate = framerate
    camera.vflip = vflip
    camera.hflip = hflip
    rawCapture = PiRGBArray(camera, size=(width, height))
    # Start coordinate, here (5, 5)
    # represents the top left corner of rectangle
    start_point = (STARTPOINT_X, STARTPOINT_Y)
    # Ending coordinate, here (220, 220)
    # represents the bottom right corner of rectangle
    end_point = (ENDPOINT_X, ENDPOINT_Y)
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    # allow the camera to warmup
    time.sleep(0.1)
    
    counter = 1
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image,
        # then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        # show the frame
        cv2.imshow("Frame", image)
        
        
        key = cv2.waitKey(1) & 0xFF
        rawCapture.truncate(0)
        
        if key == ord("c") or key == ord("C"):
            print("Capture: ", counter)
            counter += 1
            X_data.append(image)
        
            if counter > BURSTCOUNT:
                return np.array(X_data)
        
        
        # clear the stream in preparation for the next frame
        '''rawCapture.truncate(0)
        counter += 1
        print("Capture: ", counter)
        X_data.append(image)
        
        if counter > BURSTCOUNT:
            return np.array(X_data)'''
        
        # if the q key was pressed, break from the loop
        '''if key == ord("q"):
            break
        elif key == ord("c") or key == ord("C"):
            key2 = int(input("Enter 1 to accept, 2 to try again: "))
            if key2 == 1:
                #now = datetime.now()  # current date and time
                #date_time = now.strftime("%m%d%Y-%H%M%S")
                #filename = date_time + ".jpg"
                #cv2.imwrite(filename, image)
                
                X_data.append(image)
                counter += 1
                if counter > BURSTCOUNT:
                    return np.array(X_data)
                    #break
                else:
                    continue
                
            else:
                continue
                
                # BURST SHOT:
                for i in range(BURSTCOUNT):
                    print("Capture: ", counter)
                    counter += 1
                    
                    #image = frame.array
                    image = cv2.rectangle(image, start_point, end_point, color, thickness)
                    # show the frame
                    cv2.imshow("Frame", image)
                    key3 = cv2.waitKey(1) & 0xFF
                    X_data.append(image)
                    # clear the stream in preparation for the next frame
                    rawCapture.truncate(0)
                    
                    time.sleep(INTERVAL)
                    #X_data.append(image)
                
                for frame2 in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                    # grab the raw NumPy array representing the image,
                    # then initialize the timestamp
                    # and occupied/unoccupied text
                    image2 = frame2.array
                    image2 = cv2.rectangle(image, start_point, end_point, color, thickness)
                    # show the frame
                    cv2.imshow("Frame", image2)
                    key3 = cv2.waitKey(1) & 0xFF
                    # clear the stream in preparation for the next frame
                    rawCapture.truncate(0)
                
                #return np.array(X_data)
            elif key2 == 2:
                continue
            elif key2 == 0:
                break'''
    return None
    
    

def get_image(camera, width, height, framerate, hflip, vflip):
    # Grab a reference to the raw cam capture
    if width <= 0 or width > 3280 or height <= 0 or height > 2464:
        print('Invalid resolution values.')
        return
    print("Camera is now set. Please place your tongue in front of the camera.")
    print("Press c to capture. Press q to quit.")
    camera.resolution = (width, height)
    camera.framerate = framerate
    camera.vflip = vflip
    camera.hflip = hflip
    rawCapture = PiRGBArray(camera, size=(width, height))
    # Start coordinate, here (5, 5)
    # represents the top left corner of rectangle
    start_point = (STARTPOINT_X, STARTPOINT_Y)
    # Ending coordinate, here (220, 220)
    # represents the bottom right corner of rectangle
    end_point = (ENDPOINT_X, ENDPOINT_Y)
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    # allow the camera to warmup
    time.sleep(0.1)
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image,
        # then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        # show the frame
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        # if the q key was pressed, break from the loop
        if key == ord("q"):
            break
        elif key == ord("c") or key == ord("C"):
            key2 = int(input("Enter 1 to accept, 2 to try again, 0 to exit: "))
            if key2 == 1:
                #now = datetime.now()  # current date and time
                #date_time = now.strftime("%m%d%Y-%H%M%S")
                #filename = date_time + ".jpg"
                #cv2.imwrite(filename, image)
                return image
            elif key2 == 2:
                continue
            elif key2 == 0:
                break
    return None

def find_image_if_exists(image, resizeWidth, resizeHeight):
    # Find keypoints of the user's image image and compute for its descriptors
    kp1, des1 = get_kp_and_des_using_image(image, resizeWidth, resizeHeight)

    #c.execute("SELECT * FROM mainTable")  # selects all data (rows and columns) from the mainTable
    #c.execute("SELECT person_id, first_name, last_name, image_name FROM mainTable")  # selects all data (rows and columns) from the mainTable
    #result = c.fetchall()  # fetches the data to the variable result which returns a list

    # c.execute("SELECT person_id, first_name, last_name, descriptor FROM mainTable")  # selects data (rows and columns) from the mainTable
    # result = c.fetchall()  # fetches the data to the variable result which returns a list
    # print("length = ", len(result))

    # for printing the results
    highest = 0
    highest_name = None
    record = 0
    user_id = 0
    for x in mainTable_copy:
        # this is where your algorithm for scanning should be
        # Reading the image from DB
        #image_from_DB = cv2.imread(x[3])
        # Find keypoints of every image in DB and compute for its descriptors
        #kp2, des2 = get_kp_and_des_using_image(image_from_DB, resizeWidth, resizeHeight)

        filename = x[3]
        des2 = np.loadtxt(filename,dtype=int)
        des2 = np.ubyte(des2)

        # FLANN parameters
        index_params = dict(algorithm=6,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2

        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # ratio test as per Lowe's paper
        ctr = 0
        for i, (m, n) in enumerate(matches):
            # print("m.distance: " + str(m.distance) + " < 0.7*n.distance" + str(0.7*n.distance))
            if m.distance < 0.7 * n.distance:
                ctr += 1
                # print("ctr : " + str(ctr))
        record += 1
        print("ctr : " + str(ctr) + " record: " + str(record))
        
        #MATCHVALUE = 4000*0.1
        #if ctr >= MATCHVALUE:
        #    #print("Found a match.")
        #    print('Welcome ' + x[1] + ' ' + x[2])
        #    return x[0]
        #    #return x[1] + " " + x[2]
            
        # Find the highest and store to highest variable
        if ctr >= highest:
            highest = ctr
            user_id = x[0]
#            print('Welcome ' + x[1] + ' ' + x[2])
            highest_name = x[1] + " " + x[2]
    return user_id, highest_name
        

def find_highest(image, resizeWidth, resizeHeight):
    # Find keypoints of the user's image image and compute for its descriptors
    kp1, des1 = get_kp_and_des_using_image(image, resizeWidth, resizeHeight)

    #c.execute("SELECT * FROM mainTable")  # selects all data (rows and columns) from the mainTable
    # c.execute("SELECT first_name, last_name, image_name, descriptor FROM mainTable")  # selects data (rows and columns) from the mainTable
    # result = c.fetchall()  # fetches the data to the variable result which returns a list
    # print("length = ", len(result))


    # for printing the results
    highest = 0
    highest_name = None
    record = 0
    for x in mainTable_copy:
        # this is where your algorithm for scanning should be
        # Reading the image from DB
        #image_from_DB = cv2.imread(x[2])
        # Find keypoints of every image in DB and compute for its descriptors
        #kp2, des2 = get_kp_and_des_using_image(image_from_DB, resizeWidth, resizeHeight)
        # Get the descriptor

        filename = x[3]
        des2 = np.loadtxt(filename,dtype=int)
        des2 = np.ubyte(des2)
        
        # FLANN parameters
        index_params = dict(algorithm=6,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2

        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        #del des2
        print("x: ", x)
        # ratio test as per Lowe's paper
        ctr = 0
        for i, (m, n) in enumerate(matches):
            # print("m.distance: " + str(m.distance) + " < 0.7*n.distance" + str(0.7*n.distance))                
                if m.distance < 0.7 * n.distance:
                    ctr += 1
                    # print("ctr : " + str(ctr))
        record += 1
        print("ctr : " + str(ctr) + " record: " + str(record))
        #MATCHVALUE = 4000*0.1
        '''if ctr >= MATCHVALUE:
            print("Found a match.")
            print("Person id = ", x[0])
            return x[0]'''
            
            
        # Find the highest and store to highest variable
        if ctr >= highest:
            highest = ctr
            highest_name = x[1] + " " + x[2]

    return highest, highest_name


def get_kp_and_des_using_image(image, resizeWidth, resizeHeight):
    # Resizing the image for compatibility
    # image = cv2.resize(image, (resizeWidth, resizeHeight))
    # The initial processing of the image
    image = cv2.medianBlur(image, 3)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit=5)
    enhanced_image = clahe.apply(image_bw) + CONTRAST

    # y = STARTPOINT_Y - 56
    # x = STARTPOINT_X - 44
    # h = (ENDPOINT_Y - 378) - y
    # w = (ENDPOINT_X - 220) - x
    y = STARTPOINT_Y
    x = STARTPOINT_X
    h = (ENDPOINT_Y) - y
    w = (ENDPOINT_X) - x
    crop = enhanced_image[y:y + h, x:x + w]
    #plt.imshow(crop), plt.show()

    # temp_image_file = "Resources/CLAHE-temp-image.jpg"
    # cv2.imwrite(temp_image_file, crop)
    # enhanced_image = cv2.imread(temp_image_file, cv2.IMREAD_GRAYSCALE)  # queryImage

    # Initiate ORB detector
    orb1 = cv2.ORB_create(nfeatures=KPNUMBER)
    # find the keypoints and descriptors with ORB
    kp, des = orb1.detectAndCompute(crop, None)

    kp_image = cv2.drawKeypoints(crop, kp, None, color=(0, 255, 0), flags=0)
    #plt.imshow(kp_image), plt.show()
    return kp, des

def convertImageToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData


def insert_initial_data(first_name, last_name, image_data, person_id):  # insert data into the database
    # insert algorithm for image processing

    c.execute("""INSERT INTO mainTable (person_id, first_name, last_name, image_data)
                VALUES (:person_id, :first_name, :last_name, :image_data)""",
              {'person_id': person_id, 'first_name': first_name, 'last_name': last_name, 'image_data': image_data})
    conn.commit()  # apply changes


def update_initial_data(first_name, last_name, image_name, image_data, descriptor, person_id):
    c.execute("""UPDATE mainTable SET image_name = :image_name, descriptor = :descriptor 
                WHERE first_name = :first_name AND last_name = :last_name AND image_data = :image_data""",
              {'descriptor': descriptor, 'first_name': first_name, 'last_name': last_name, 'image_name': image_name,
               'image_data': image_data})
    '''c.execute("""UPDATE mainTable SET image_name = :image_name, descriptor = :descriptor 
                WHERE person_id = :person_id""",
              {'descriptor': descriptor, 'first_name': first_name, 'last_name': last_name, 'image_name': image_name,
               'image_data': image_data})'''
    conn.commit()


def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
    else:
        print("The file does not exist")


def get_user_id(firstname, lastname, imagename):
    c.execute("SELECT person_id FROM mainTable WHERE first_name = :firstname AND last_name = :lastname AND image_data = :imagename",
        {'firstname': firstname, 'lastname': lastname, 'imagename': imagename})
    userid = c.fetchall()

    for user_id in userid:
        return user_id[0]


def generate_keypoint_file(keypoint, descriptor, user_id):
    file = open("Resources/kpdesID"+str(user_id)+".txt", "w")
    file.write(listToString(keypoint))
    file.write(listToString(descriptor))
    file.close()


def generate_des_file(descriptor, user_id):
    now = datetime.now()  # current date and time
    filename = "Resources/desID" + str(user_id) + "_" + now.strftime("%m%d%Y-%H%M%S") + ".txt"
    #file = open(filename, "w")
    #file.write(listToString(descriptor))
    #file.write(descriptor.tostring())
    #file.close()
    # save to file in numpy.ndarray format
    np.savetxt(filename, descriptor, fmt='%d')
    #descriptor.tofile(filename)
    return filename

def add_match_img(user, image):
    #Save image to a file
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m%d%Y-%H%M%S")
    filename = "Matched/match_" + user + "-" + date_time + ".jpg"
    cv2.imwrite(filename, image)
    

# function for converting list to string
def listToString(org_list, separator='\n'):
    """ Convert list to string, by joining all item in list with given separator.
        Returns the concatenated string """
    # return separator.join(org_list)
    return separator.join(str(v) for v in org_list)


def enhance_img(filename, firstName, lastName, imageFileName):
    # Reading the image from the present directory
    image = cv2.imread(filename)
    # Resizing the image for compatibility
    image = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))

    # The initial processing of the image
    image = cv2.medianBlur(image, 3)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit=5)
    final_img = clahe.apply(image_bw) + CONTRAST

    # # Ordinary thresholding the same image
    # _, ordinary_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)

    # Showing all the three images
    # cv2.imshow("ordinary threshold", ordinary_img)
    writeClahe = "Resources/CLAHE-image" + str(get_user_id(firstName, lastName, convertImageToBinaryData(imageFileName))) + ".jpg"
    cv2.imwrite(writeClahe, final_img)
    # cv2.imshow("CLAHE image", final_img)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()
    return writeClahe       # return file that contains the image enhanced by CLAHE


def save_key_points(image, user_id, resizeWidth, resizeHeight):

    # compute the descriptors with ORB
    kp1, des1 = get_kp_and_des_using_image(image, resizeWidth, resizeHeight)

    # calls generate_keypoint_file function to write keypoint and descriptor to a file
    # generate_keypoint_file(kp1, des1, user_id)

    # calls generate_des_file function to write descriptor to a file
    des_file = generate_des_file(des1, user_id)

    return des_file     # return file that contains the descriptor

# def get_kp_and_des(filename, resizeWidth, resizeHeight):
#     # Reading the image from the present directory
#     image = cv2.imread(filename)
#     # Resizing the image for compatibility
#     image = cv2.resize(image, (resizeWidth, resizeHeight))
#     # The initial processing of the image
#     image = cv2.medianBlur(image, 3)
#     image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # The declaration of CLAHE
#     # clipLimit -> Threshold for contrast limiting
#     clahe = cv2.createCLAHE(clipLimit=5)
#     enhanced_image = clahe.apply(image_bw) + CONTRAST
#
#     y = STARTPOINT_Y - 56
#     x = STARTPOINT_X - 44
#     h = (ENDPOINT_Y - 378) - y
#     w = (ENDPOINT_X - 220) - x
#     crop = enhanced_image[y:y + h, x:x + w]
#     #plt.imshow(crop), plt.show()
#
#     # temp_image_file = "Resources/CLAHE-temp-image.jpg"
#     # cv2.imwrite(temp_image_file, crop)
#     # enhanced_image = cv2.imread(temp_image_file, cv2.IMREAD_GRAYSCALE)  # queryImage
#
#     # Initiate ORB detector
#     orb1 = cv2.ORB_create(nfeatures=KPNUMBER)
#     # find the keypoints and descriptors with ORB
#     kp, des = orb1.detectAndCompute(crop, None)
#
#     kp_image = cv2.drawKeypoints(crop, kp, None, color=(0, 255, 0), flags=0)
#     plt.imshow(kp_image), plt.show()
#     return kp, des

# def find_tongueImage_match(filename, resizeWidth, resizeHeight):
#     isFound = False
#
#     # Find keypoints of the user's image image and compute for its descriptors
#     kp1, des1 = get_kp_and_des(filename, resizeWidth, resizeHeight)
#
#     c.execute("SELECT * FROM mainTable")  # selects all data (rows and columns) from the mainTable
#     result = c.fetchall()  # fetches the data to the variable result which returns a list
#
#     # for printing the results
#     for x in result:
#         # this is where your algorithm for scanning should be
#         # Reading the image from DB
#         # Find keypoints of every image in DB and compute for its descriptors
#         kp2, des2 = get_kp_and_des(x[3], resizeWidth, resizeHeight)
#
#         # FLANN parameters
#         index_params = dict(algorithm=6,
#                             table_number=6,  # 12
#                             key_size=12,  # 20
#                             multi_probe_level=1)  # 2
#
#         # FLANN_INDEX_KDTREE = 1
#         # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#         search_params = dict(checks=50)  # or pass empty dictionary
#         flann = cv2.FlannBasedMatcher(index_params, search_params)
#         matches = flann.knnMatch(des1, des2, k=2)
#         # ratio test as per Lowe's paper
#         ctr = 0
#         for i, (m, n) in enumerate(matches):
#             # print("m.distance: " + str(m.distance) + " < 0.7*n.distance" + str(0.7*n.distance))
#             if m.distance < 0.7 * n.distance:
#                 ctr += 1
#                 # print("ctr : " + str(ctr))
#         print("ctr : " + str(ctr))
#         #MATCHVALUE = 4000*0.1
#         if ctr >= MATCHVALUE:
#             # print("Found a match.")
#             isFound = True
#             break
#     return isFound

def add_new_user_to_database(firstname, lastname, image, resizeWidth, resizeHeight):
    #Save image to a file
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m%d%Y-%H%M%S")
    filename = date_time + ".jpg"
    cv2.imwrite(filename, image)

    insert_initial_data(firstname, lastname, convertImageToBinaryData(filename), date_time)
    userid = get_user_id(firstname, lastname, convertImageToBinaryData(filename))
    des_file = save_key_points(image, userid, resizeWidth, resizeHeight)
    update_initial_data(firstname, lastname, filename, convertImageToBinaryData(filename), des_file, date_time)
    return

def add_image(person_id, image, resizeWidth, resizeHeight):
    #Save image to a file
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m%d%Y-%H%M%S")
    filename = date_time + ".jpg"
    #cv2.imwrite(filename, image)

    # Get 1st and last name using person_id 
    c.execute("SELECT first_name, last_name FROM mainTable WHERE person_id = :person_id",
        {'person_id': person_id})
    data = c.fetchone()
    first_name = data[0]
    last_name = data[1]

    # compute the descriptors with ORB
    kp1, des1 = get_kp_and_des_using_image(image, resizeWidth, resizeHeight)

    # calls generate_keypoint_file function to write keypoint and descriptor to a file
    # generate_keypoint_file(kp1, des1, user_id)

    # calls generate_des_file function to write descriptor to a file
    des_file = generate_des_file(des1, person_id)


    #insert_initial_data(firstname, lastname, convertImageToBinaryData(filename), date_time)
    #c.execute("""INSERT INTO mainTable (person_id, first_name, last_name, image_name, image_data, descriptor)
    #            VALUES (:person_id, :first_name, :last_name, :image_name, :image_data, :descriptor)""",
    #          {'person_id': person_id, 'first_name': first_name, 'last_name': last_name, 'image_name': filename, 'image_data': convertImageToBinaryData(filename), 'descriptor': des_file})
    #conn.commit()  # apply changes

    c.execute("""INSERT INTO mainTable (person_id, first_name, last_name, descriptor)
                VALUES (:person_id, :first_name, :last_name, :descriptor)""",
              {'person_id': person_id, 'first_name': first_name, 'last_name': last_name, 'descriptor': des_file})
    conn.commit()  # apply changes

    #return des_file     # return file that contains the descriptor
    
    #userid = get_user_id(firstname, lastname, convertImageToBinaryData(filename))
    #des_file = save_key_points(image, userid, resizeWidth, resizeHeight)
    #update_initial_data(firstname, lastname, filename, convertImageToBinaryData(filename), des_file, date_time)
    return first_name, last_name, des_file



# def add_to_database(firstname, lastname, filename, resizeWidth, resizeHeight):
#     input_imageData = filename
#     insert_initial_data(firstname, lastname, convertImageToBinaryData(input_imageData))
#     userid = get_user_id(firstname, lastname, convertImageToBinaryData(input_imageData))
#     des_file = save_key_points(input_imageData, userid, resizeWidth, resizeHeight)
#     update_initial_data(firstname, lastname, filename, convertImageToBinaryData(input_imageData), des_file)
#     return

# def get_fullname(filename, resizeWidth, resizeHeight):
#     username = ""
#     # Find keypoints of the user's image image and compute for its descriptors
#     kp1, des1 = get_kp_and_des(filename, resizeWidth, resizeHeight)
#
#     c.execute("SELECT * FROM mainTable")  # selects all data (rows and columns) from the mainTable
#     result = c.fetchall()  # fetches the data to the variable result which returns a list
#     # for printing the results
#     for x in result:
#         # this is where your algorithm for scanning should be
#         # Reading the image from the present directory
#         # Find keypoints of every image in DB and compute for its descriptors
#         kp2, des2 = get_kp_and_des(x[3], resizeWidth, resizeHeight)
#
#         # FLANN parameters
#         index_params = dict(algorithm=6,
#                             table_number=6,  # 12
#                             key_size=12,  # 20
#                             multi_probe_level=1)  # 2
#
#         # FLANN_INDEX_KDTREE = 1
#         # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#         search_params = dict(checks=50)  # or pass empty dictionary
#         flann = cv2.FlannBasedMatcher(index_params, search_params)
#         matches = flann.knnMatch(des1, des2, k=2)
#         # Need to draw only good matches, so create a mask
#         matchesMask = [[0, 0] for i in range(len(matches))]
#         # ratio test as per Lowe's paper
#         ctr = 0
#         for i, (m, n) in enumerate(matches):
#             # print("m.distance: " + str(m.distance) + " < 0.7*n.distance" + str(0.7*n.distance))
#             if m.distance < 0.7 * n.distance:
#                 matchesMask[i] = [1, 0]
#                 ctr += 1
#                 # print("ctr : " + str(ctr))
#         #print("ctr : " + str(ctr))
#         #MATCHVALUE = 4000 * 0.1
#         if ctr >= MATCHVALUE:
#             # print("Found a match.")
#             username = x[1] + " " + x[2]
#
#     return username


# def signIn_this_img(filename):
#     if filename != "":
#         # check if tongue image exists
#         if os.path.exists(filename):
#             # find for matching image
#             if find_tongueImage_match(filename, RESIZE_WIDTH, RESIZE_HEIGHT) == False:
#                 # no match found
#                 print("Invalid user.")
#             else:
#                 print("Welcome " + get_fullname(filename, RESIZE_WIDTH, RESIZE_HEIGHT) + "! You are now signed in.")
#         else:
#             print("User does not exist.")
#     else:
#         print("Invalid filename.")
#     return

def addMoreImages(firstname, lastname, camera):
    print("Additional images are required for verification purposes.")
    while True:
        key3 = int(input("Enter 1 to capture, 0 to exit: "))
        if key3 == 1:
            user_tongue_image = get_image(camera, CAMERA_WIDTH, CAMERA_HEIGHT, FRAME_RATE, H_FLIP, V_FLIP)
            if user_tongue_image is None:
                print("Invalid image. Try again.")
            else:
                add_new_user_to_database(firstname, lastname, user_tongue_image, RESIZE_WIDTH, RESIZE_HEIGHT)
                print("New image added.")
        elif key3 == 0:
            break
    return
  
  
def update_user(camera):
    user = None
    user_tongue_image = get_image(camera, CAMERA_WIDTH, CAMERA_HEIGHT, FRAME_RATE, H_FLIP, V_FLIP)
    print("Signing image in process. Please wait...")
    if user_tongue_image is None:
        return
    else:
        start_time = time.perf_counter()
        person_id, user = find_image_if_exists(user_tongue_image, RESIZE_WIDTH, RESIZE_HEIGHT)
        if user is None:
            # no match found
            print("Invalid user.")
            end_time = time.perf_counter()
            print("Elapsed time:", round((end_time - start_time),2))
            return
    # User exists on the database
    print('Welcome ' + user)
    print("Additional images are required for verification purposes.")
    while True:
        key3 = int(input("Enter 1 to capture, 0 to exit: "))
        if key3 == 1:
            user_tongue_image = get_image(camera, CAMERA_WIDTH, CAMERA_HEIGHT, FRAME_RATE, H_FLIP, V_FLIP)
            if user_tongue_image is None:
                print("Invalid image. Try again.")
            else:
                firstname, lastname, des_file = add_image(person_id, user_tongue_image, RESIZE_WIDTH, RESIZE_HEIGHT)
                # append to mainTable in memory
                mainTable_copy.append([person_id, firstname, lastname, des_file])
                print("New image added.")
        elif key3 == 0:
            break
    return  
  
  
  
def sign_up(camera):  # function for inserting a user
    user_tongue_image = get_thirty(camera, CAMERA_WIDTH, CAMERA_HEIGHT, FRAME_RATE, H_FLIP, V_FLIP)

    start_time = time.perf_counter()
    if len(user_tongue_image) == BURSTCOUNT:
        '''for index in range(len(user_tongue_image)):
            ctrr, user = find_highest(user_tongue_image[index], RESIZE_WIDTH, RESIZE_HEIGHT)
            if ctrr > THRESHOLD:
                print("User already exists.")
                end_time = time.perf_counter()
                print("Elapsed time:", round((end_time - start_time),2))
                return'''

        # No match found
        first_name = input("Enter First Name: ")
        last_name = input("Enter Last Name: ")
        
        #Save image to a file
        image = user_tongue_image[0]
        now = datetime.now()  # current date and time
        person_id = now.strftime("%m%d%Y-%H%M%S")
        filename = person_id + ".jpg"
        #cv2.imwrite(filename, image)
        
        # compute the descriptors with ORB
        kp1, des1 = get_kp_and_des_using_image(image, RESIZE_WIDTH, RESIZE_HEIGHT)

        # calls generate_keypoint_file function to write keypoint and descriptor to a file
        # generate_keypoint_file(kp1, des1, user_id)

        # calls generate_des_file function to write descriptor to a file
        des_file = generate_des_file(des1, person_id)


        #insert_initial_data(firstname, lastname, convertImageToBinaryData(filename), date_time)
#        c.execute("""INSERT INTO mainTable (person_id, first_name, last_name, image_name, image_data, descriptor)
#                    VALUES (:person_id, :first_name, :last_name, :image_name, :image_data, :descriptor)""",
#                  {'person_id': person_id, 'first_name': first_name, 'last_name': last_name, 'image_name': filename, 'image_data': convertImageToBinaryData(filename), 'descriptor': des_file})
#        conn.commit()  # apply changes
 
        c.execute("""INSERT INTO mainTable (person_id, first_name, last_name, descriptor)
                    VALUES (:person_id, :first_name, :last_name, :descriptor)""",
                  {'person_id': person_id, 'first_name': first_name, 'last_name': last_name, 'descriptor': des_file})
        conn.commit()  # apply changes

        # append to mainTable in memory
        mainTable_copy.append([person_id, first_name, last_name, des_file])
 
        counter = 1
        
        for img2 in range(len(user_tongue_image)):
            if img2 > 0: 
                #add_new_user_to_database(input_firstName, input_lastName, user_tongue_image[img], RESIZE_WIDTH, RESIZE_HEIGHT)
                #Save image to a file
                image = user_tongue_image[img2]
                now = datetime.now()  # current date and time
                date_time = now.strftime("%m%d%Y-%H%M%S")
                filename = date_time + "-" + str(counter) + ".jpg"
                #cv2.imwrite(filename, image)
                counter += 1
        
                # compute the descriptors with ORB
                kp1, des1 = get_kp_and_des_using_image(image, RESIZE_WIDTH, RESIZE_HEIGHT)

                # calls generate_keypoint_file function to write keypoint and descriptor to a file
                # generate_keypoint_file(kp1, des1, user_id)

                # calls generate_des_file function to write descriptor to a file
                des_file = generate_des_file(des1, person_id)


                #insert_initial_data(firstname, lastname, convertImageToBinaryData(filename), date_time)
#                c.execute("""INSERT INTO mainTable (person_id, first_name, last_name, image_name, image_data, descriptor)
#                            VALUES (:person_id, :first_name, :last_name, :image_name, :image_data, :descriptor)""",
#                          {'person_id': person_id, 'first_name': first_name, 'last_name': last_name, 'image_name': filename, 'image_data': convertImageToBinaryData(filename), 'descriptor': des_file})
#                conn.commit()  # apply changes

                c.execute("""INSERT INTO mainTable (person_id, first_name, last_name, descriptor)
                            VALUES (:person_id, :first_name, :last_name, :descriptor)""",
                           {'person_id': person_id, 'first_name': first_name, 'last_name': last_name, 'descriptor': des_file})
                conn.commit()  # apply changes

                # append to mainTable in memory
                mainTable_copy.append([person_id, first_name, last_name, des_file])

                #addMoreImages(input_firstName, input_lastName, camera)
        print("User registered.")
    end_time = time.perf_counter()
    print("Elapsed time:", round((end_time - start_time),2))    
    
    '''if user_tongue_image is None:
        return
    else:
        user = find_image_if_exists(user_tongue_image, RESIZE_WIDTH, RESIZE_HEIGHT)
        if user is None:
            # no match found, insert user
            input_firstName = input("Enter First Name: ")
            input_lastName = input("Enter Last Name: ")
            add_new_user_to_database(input_firstName, input_lastName, user_tongue_image, RESIZE_WIDTH, RESIZE_HEIGHT)
            addMoreImages(input_firstName, input_lastName, camera)
            print("User registered.")
        else:
            print("User already exists.")
    return'''

def sign_in(camera):
    user_tongue_image = get_image(camera, CAMERA_WIDTH, CAMERA_HEIGHT, FRAME_RATE, H_FLIP, V_FLIP)
    print("Signing image in process. Please wait...")
    if user_tongue_image is None:
        return
    else:
        start_time = time.perf_counter()
        print("start find")
        ctrr, user = find_highest(user_tongue_image, RESIZE_WIDTH, RESIZE_HEIGHT)
        if ctrr < THRESHOLD:
            user = None
        print("end find")
        if user is None:
            # no match found
            print("Invalid user.")
        else:
            print("Welcome " + user + "! You are now signed in.")
            add_match_img(user, user_tongue_image)
        end_time = time.perf_counter()
        print("Elapsed time:", round((end_time - start_time),2))
    return

def scan_user():
    c.execute("SELECT * FROM mainTable")  # selects all data (rows and columns) from the mainTable
    result = c.fetchall()  # fetches the data to the variable result which returns a list

    # for formatting purposes
    print("%-15s %-15s %-15s %-35s %10s" % ('ID', 'First Name', 'Last Name', 'File Name', 'Descriptor'))
    print("%-15s %-15s %-15s %-35s %-20s" % ('----', '---------', '---------', "----------", '----------'))

    # for printing the results
    for x in result:
        print("%-15s %-15s %-15s %-35s %-20s" % (x[0], x[1], x[2], x[3], x[5]))  # prints the content per row
        # this is where your algorithm for scanning should be
        # return true if similar image is found

    # Show mainTable size in memory
    print("Database size in memory: ", len(mainTable_copy))

'''def update_user(camera):  # function for updating the content of a user
    user_tongue_image = get_image(camera, CAMERA_WIDTH, CAMERA_HEIGHT, FRAME_RATE, H_FLIP, V_FLIP)
    print("Signing image in process. Please wait...")
    if user_tongue_image is None:
        return
    else:
        start_time = time.perf_counter()
        user = find_image_if_exists(user_tongue_image, RESIZE_WIDTH, RESIZE_HEIGHT)
        if user is None:
            # no match found
            print("Invalid user.")
        else:
            print("Welcome " + user + "! You are now signed in.")
            addMoreImages(input_firstName, input_lastName, camera)
            # add_match_img(user, user_tongue_image)
        end_time = time.perf_counter()
        print("Elapsed time:", round((end_time - start_time),2))
    return'''
    

def mainProgram():
    # initialize camera
    cam = PiCamera()
    while True:
        #try:

            key = int(input("Enter 1 to sign up, 2 to sign in, 3 to view DB, 4 to update, 0 to exit: "))
            if key == 1:
                sign_up(cam)  # proceeds to sign_up function

            elif key == 2:
                sign_in(cam)  # proceeds to sign_in function

            elif key == 3:
                scan_user()  # proceeds to scan_user function
                
            elif key == 4:
                update_user(cam)  # proceeds to update_user function
   
            
            elif key == 0:
                break

        #except:
        #    print("Invalid input.")
            
    cv2.destroyAllWindows()

# Main Program
mainProgram()
conn.commit()
