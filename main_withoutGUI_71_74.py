import cv2
import numpy as np
from mediapipe.python.solutions.face_detection import FaceDetection
from typing import List, Iterable
from mediapipe.python.solutions.face_mesh import FaceMesh

upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 408, 415, 272, 271, 268, 12, 38, 41, 42, 191, 78, 76]
lower_lip = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

#:::::::::::::::: Functionns :::::::::::::::::::::::::::;
# rescaling saved video or image
def rescaleFrame(frame,scale=0.75):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width,height)
    return cv2.resize(frame,dimensions)
def detect_landmarks(src: np.ndarray, is_stream: bool = False):
    """
    Given an image `src` retrieves the facial landmarks associated with it
    """
    with FaceMesh(static_image_mode=not is_stream, max_num_faces=2) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    return None #none type error when landmarks are not detected
def landmarks(landmarks, height: int, width: int, mask: Iterable=None):
    """
    The landmarks returned by mediapipe have coordinates between [0, 1].
    This function normalizes them in the range of the image dimensions so they can be played with.
    """
    normalized_landmarks = np.array([(int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks])
    #landmark is not iterable in landmarks if landmarks are  nonetype.
    if mask:
        normalized_landmarks = normalized_landmarks[mask]
    return normalized_landmarks

# applying lipstick on the image
def apply_lipstick(src: np.ndarray, is_stream: bool, feature: str,clr='pink', show_landmarks: bool = False):
    """
    Takes in a source image and applies effects onto it.
    """
    ret_landmarks = detect_landmarks(src, is_stream) #if returns none then in normalize_landmark func..
    height, width, _ = src.shape
    feature_landmarks = None
    if feature == 'lips':
        feature_landmarks = landmarks(ret_landmarks, height, width, upper_lip + lower_lip)
        #lip color selection
        if clr=='orange':
            mask = lip_mask(src, feature_landmarks, [0, 0, 255])
        elif clr=='purple':
            mask = lip_mask(src, feature_landmarks, [255, 0, 0])
        elif clr=='pink':
            mask = lip_mask(src, feature_landmarks, [153, 0, 157])
        elif clr=='green':
            mask = lip_mask(src, feature_landmarks, [0, 255, 0])
        elif clr=='berry':
            mask = lip_mask(src, feature_landmarks, [40, 0, 100])
        elif clr=='caramel':
            mask = lip_mask(src, feature_landmarks, [50, 70, 70])
        elif clr=='yellow':
            mask = lip_mask(src, feature_landmarks, [0,255,255])
        elif clr=='aqua':
            mask = lip_mask(src, feature_landmarks, [255,255,0])
        elif clr=='peach':
            mask = lip_mask(src, feature_landmarks, [35,35,139])
        elif clr=='red':
            mask = lip_mask(src, feature_landmarks, [40, 37, 255])
        output = cv2.addWeighted(src, 1.0, mask, 0.4, 0.0)
    return output
# mask for lips
def lip_mask(src: np.ndarray, points: np.ndarray, color: list):
    """
    Given a src image, points of lips and a desired color
    Returns a colored mask that can be added to the src
    """
    mask = np.zeros_like(src)  # Create a mask
    mask = cv2.fillPoly(mask, [points], color)  # Mask for the required facial feature
    # Blurring the region, so it looks natural
    # TODO: Get glossy finishes for lip colors, instead of blending in replace the region
    mask = cv2.GaussianBlur(mask, (7, 7), 5)
    return mask
#---------------------------------------------------------------------
#------------------------(Main Code)----------------------------------
# Video Input from Webcam
video_cap = cv2.VideoCapture("Videos/v5.mp4")
while True:
    ret, frame = video_cap.read()
    if ret is False: #Resolving_not getting frames thus giving error(NoneType)
        continue
    frame=rescaleFrame(frame,scale=.15)
    frame = cv2.flip(frame, 1)# flip code = 1 _for horizontal
    if ret:
        output = apply_lipstick(frame, False, 'lips','pink', False)
        cv2.imshow("Original", frame)
        cv2.imshow("Feature", output)

        cv2.waitKey(1) # 1ms wait for next frame/1ms delay__ 0 means forever

# # Static Images
# image = cv2.imread("model.jpg", cv2.IMREAD_UNCHANGED)
# output = apply_lipstick(image, False, 'foundation', False)
#
# cv2.imshow("Original", image)
# cv2.imshow("Feature", output)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----------------------------------------------------------------------