import cv2
import math
import numpy as np
import dlib
from imutils import face_utils,rotate_bound
from imutils.face_utils.helpers import FACIAL_LANDMARKS_68_IDXS, FACIAL_LANDMARKS_5_IDXS
import json
import base64



class FaceCheck():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.desiredLeftEye = (0.35, 0.35)
        self.desiredFaceWidth = 256
        self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #simple hack ;)
        if (len(shape)==68):
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]
            
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1] * 2.0
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, int(h*1.5)),
            flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output

    def image_resize(self,image, width = None, height = None, inter = cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)

        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv2.resize(image, dim, interpolation = inter)
        
        return resized

    def face_orientation(self,frame, landmarks):
        size = frame.shape #(height, width, color_channel)
        image_points = np.array([
            (landmarks[33, 0], landmarks[33, 1]),     # Nose tip
            (landmarks[8, 0], landmarks[8, 1]),       # Chin
            (landmarks[36, 0], landmarks[36, 1]),     # Left eye left corner
            (landmarks[45, 0], landmarks[45, 1]),     # Right eye right corner
            (landmarks[48, 0], landmarks[48, 1]),     # Left Mouth corner
            (landmarks[54, 0], landmarks[54, 1])      # Right mouth corner
            ], dtype="double")
  
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
            ])

        # Camera internals
        center = (size[1]/2, size[0]/2)
        focal_length = center[0] / np.tan(60/2 * np.pi / 180)
        camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )

        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)

        
        axis = np.float32([[500,0,0], 
                            [0,500,0], 
                            [0,0,500]])
                            
        imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

        
        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]


        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), (landmarks[33, 0], landmarks[33, 1]) 

    def process(self,img_pil):
        image = np.asarray(img_pil)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)

        if(len(rects)==1):
            for (i, rect) in enumerate(rects):
                landmarks = self.predictor(gray, rect)
                landmarks = face_utils.shape_to_np(landmarks) 
                imgpts, modelpts, rotate_degree, nose = self.face_orientation(image, landmarks)
                frame = image
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                
                h = int(2.5*h)
                w = int(2*w)
                y = y-h//2
                x = x-w//4
                print((h,w))
                print((y,x))
                cropimg = frame[max(y,0):min(y+h,frame.shape[0]), max(x,0):min(x+w,frame.shape[1])]
                cropimg = rotate_bound(cropimg, int(rotate_degree[0]))
                if(w > h):
                    cropimg = self.image_resize(cropimg,width=200)
                else:
                    cropimg = self.image_resize(cropimg,height=200)

                success, cropimg = cv2.imencode('.jpg', cropimg)
                cropimg_str = cropimg.tostring()
                cropimg_encoded = base64.b64encode(cropimg_str)
                cropimg_base64 = cropimg_encoded.decode('utf-8')

            output = ({
                "message" : "Face is valid.",
                "isFaceValid":True,
                "cropImage": cropimg_base64,
                })

        elif(len(rects)>1):
            output = ({
                "message" : "Found multiple faces.",
                "isFaceValid":False,
                "cropImage": None,
            })
        
        elif(len(rects)<1):
            output = ({
                "message" : "No face found.",
                "isFaceValid":False,
                "cropImage": None,
            })
        
        return output

    def process_noResize(self,img_pil):
        image = np.asarray(img_pil)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)

        if(len(rects)==1):
            for (i, rect) in enumerate(rects):
                landmarks = self.predictor(gray, rect)
                landmarks = face_utils.shape_to_np(landmarks) 
                imgpts, modelpts, rotate_degree, nose = self.face_orientation(image, landmarks)

                imagecv = np.asarray(image)
                faceAligned = self.align(imagecv, gray, rect)

                success, cropimg = cv2.imencode('.jpg', faceAligned)
                cropimg_str = cropimg.tostring()
                cropimg_encoded = base64.b64encode(cropimg_str)
                cropimg_base64 = cropimg_encoded.decode('utf-8')

            output = ({
                "message" : "Face is valid.",
                "isFaceValid":True,
                "cropImage": cropimg_base64,
                })

        elif(len(rects)>1):
            output = ({
                "message" : "Found multiple faces.",
                "isFaceValid":False,
                "cropImage": None,
            })
        
        elif(len(rects)<1):
            output = ({
                "message" : "No face found.",
                "isFaceValid":False,
                "cropImage": None,
            })
        
        return output
        
            
        
            
