import cv2.cv2 as cv2
import numpy as np

from utils.image_classifier import ImageClassifier, NO_FACE_LABEL

# Color RGB Codes & Font
WHITE_COLOR = (255, 255, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 255, 104)
FONT = cv2.QT_FONT_NORMAL

# Frame Width & Height
FRAME_WIDTH = 640
FRAME_HEIGHT = 490


class BoundingBox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @property
    def origin(self) -> tuple:
        return self.x, self.y

    @property
    def top_right(self) -> int:
        return self.x + self.w

    @property
    def bottom_left(self) -> int:
        return self.y + self.h


def draw_face_rectangle(bb: BoundingBox, img, color=BLUE_COLOR):
    cv2.rectangle(img, bb.origin, (bb.top_right, bb.bottom_left), color, 2)


def draw_landmark_points(points: np.ndarray, img, color=WHITE_COLOR):
    if points is None:
        return None
    for (x, y) in points:
        cv2.circle(img, (x, y), 1, color, -1)


def write_label(x: int, y: int, label: str, img, color=BLUE_COLOR):
    if label == NO_FACE_LABEL:
        cv2.putText(img, label.upper(), (int(FRAME_WIDTH / 2), int(FRAME_HEIGHT / 2)), FONT, 1, color, 2, cv2.LINE_AA)
    cv2.putText(img, label, (x + 10, y - 10), FONT, 1, color, 2, cv2.LINE_AA)


class RealTimeEmotionDetector:
    CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    vidCapture = None

    def __init__(self, classifier_model: ImageClassifier):
        self.__init_video_capture(camera_idx=0, frame_w=FRAME_WIDTH, frame_h=FRAME_HEIGHT)
        self.classifier = classifier_model

    def __init_video_capture(self, camera_idx: int, frame_w: int, frame_h: int):
        self.vidCapture = cv2.VideoCapture(camera_idx)
        self.vidCapture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
        self.vidCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

    def read_frame(self) -> np.ndarray:
        rect, frame = self.vidCapture.read()
        return frame

    def transform_img(self, img: np.ndarray) -> np.ndarray:
        # load the input image, resize it, and convert it to gray-scale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray-scale
        resized_img = self.CLAHE.apply(gray_img)  # resize
        return resized_img

    def execute(self, wait_key_delay=33, quit_key='q', frame_period_s=0.75):
        frame_cnt = 0
        predicted_labels = ''
        old_txt = None
        rectangles = [(0, 0, 0, 0)]
        landmark_points_list = [[(0, 0)]]
        while cv2.waitKey(delay=wait_key_delay) != ord(quit_key):
            frame_cnt += 1

            frame = self.read_frame()
            if frame_cnt % (frame_period_s * 100) == 0:
                frame_cnt = 0
                predicted_labels = self.classifier.classify(img=self.transform_img(img=frame))
                rectangles = self.classifier.extract_face_rectangle(img=frame)
                landmark_points_list = self.classifier.extract_landmark_points(img=frame)
            for lbl, rectangle, lm_points in zip(predicted_labels, rectangles, landmark_points_list):
                draw_face_rectangle(BoundingBox(*rectangle), frame)
                draw_landmark_points(points=lm_points, img=frame)
                write_label(rectangle[0], rectangle[1], label=lbl, img=frame)

                if old_txt != predicted_labels:
                    print('[INFO] Predicted Labels:', predicted_labels)
                    old_txt = predicted_labels

            cv2.imshow('Emotion Detection - Mimics', frame)

        cv2.destroyAllWindows()
        self.vidCapture.release()


def run_real_time_emotion_detector(
        classifier_algorithm: str,
        predictor_path: str,
        dataset_csv: str,
        dataset_images_dir: str = None):
    from utils.data_land_marker import LandMarker
    from utils.image_classifier import ImageClassifier
    from os.path import isfile

    land_marker = LandMarker(landmark_predictor_path=predictor_path)

    if not isfile(dataset_csv):  # If data-set not built before.
        print('[INFO]', f'Dataset file: "{dataset_csv}" could not found.')
        from data_preparer import run_data_preparer
        run_data_preparer(land_marker, dataset_images_dir, dataset_csv)
    else:
        print('[INFO]', f'Dataset file: "{dataset_csv}" found.')

    classifier = ImageClassifier(csv_path=dataset_csv, algorithm=classifier_algorithm, land_marker=land_marker)
    print('[INFO] Opening camera, press "q" to exit..')
    RealTimeEmotionDetector(classifier_model=classifier).execute()


if __name__ == "__main__":
    """The value of the parameters can change depending on the case."""
    run_real_time_emotion_detector(
        classifier_algorithm='RandomForest',  # Alternatively 'SVM'.
        predictor_path='utils/shape_predictor_68_face_landmarks.dat',
        dataset_csv='data/csv/dataset.csv',
        dataset_images_dir='data/raw'
    )
    print('Successfully terminated.')
    import cv2
import imutils
import numpy as np
import argparse
\

2. Create a model which will detect Humans:

As discussed earlier, We will use HOGDescriptor with SVM already implemented in OpenCV.  Below code will do this work:

HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.HOGDescriptor_getDefaultPeopleDetector() calls the pre-trained model for Human detection of OpenCV and then we will feed our support vector machine with it.

3. Detect() method:

Here, the actual magic will happen.

Video: A video combines a sequence of images to form a moving picture. We call these images as Frame. So in general we will detect the person in the frame. And show it one after another that it looks like a video.

That is exactly what our Detect() method will do.  It will take a frame to detect a person in it. Make a box around a person and show the frame..and return the frame with person bounded by a green box.

def detect(frame):
    bounding_box_cordinates, weights =  HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
    
    person = 1
    for x,y,w,h in bounding_box_cordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        person += 1
    
    cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.putText(frame, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.imshow('output', frame)
    return frame
Everything will be done by detectMultiScale(). It returns 2-tuple.

List containing Coordinates of bounding Box of person.
Coordinates are in form X, Y, W, H.
Where x,y are starting coordinates of box and w, h are width and height of box respectively.
Confidence Value that it is a person.
Now, We have our detect method. Let’s Create a Detector.

4. HumanDetector() method

There are two ways of getting Video.

Web Camera
Path of file stored
In this deep learning project, we can take images also. So our method will check if a path is given then search for the video or image in the given path and operate. Otherwise, it will open the webCam.

def humanDetector(args):
    image_path = args["image"]
    video_path = args['video']
    if str(args["camera"]) == 'true' : camera = True 
    else : camera = False
    writer = None
    if args['output'] is not None and image_path is None:
        writer = cv2.VideoWriter(args['output'],cv2.VideoWriter_fourcc(*'MJPG'), 10, (600,600))
    if camera:
        print('[INFO] Opening Web Cam.')
        detectByCamera(ouput_path,writer)
    elif video_path is not None:
        print('[INFO] Opening Video from path.')
        detectByPathVideo(video_path, writer)
    elif image_path is not None:
        print('[INFO] Opening Image from path.')
        detectByPathImage(image_path, args['output'])
5. DetectByCamera() method

def detectByCamera(writer):   
    video = cv2.VideoCapture(0)
    print('Detecting people...')
    while True:
        check, frame = video.read()
        frame = detect(frame)
        if writer is not None:
            writer.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
cv2.VideoCapture(0) passing 0 in this function means we want to record from a webcam. video.read() read frame by frame. It returns a check which is True if this was able to read a frame otherwise False.

Now, For each Frame, we will call detect() method. Then we write the frame in our output file.

6. DetectByPathVideo() method

This method is very similar to the previous method except we will give a path to the Video. First, we check if the video on the provided path is found or not.

Note – A full path must be given.

def detectByPathVideo(path, writer):
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if check == False:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return
    print('Detecting people...')
    while video.isOpened():
        #check is True if reading was successful 
        check, frame =  video.read()
        if check:
            frame = imutils.resize(frame , width=min(800,frame.shape[1]))
            frame = detect(frame)
            
            if writer is not None:
                writer.write(frame)
            
            key = cv2.waitKey(1)
            if key== ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()
def detectByCamera(writer):   
    video = cv2.VideoCapture(0)
    print('Detecting people...')
    while True:
        check, frame = video.read()
        frame = detect(frame)
        if writer is not None:
            writer.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
                break
    video.release()
    cv2.destroyAllWindows()
The implementation is similar to the previous function except for each frame we will check that it successfully reads the frame or not. At the end when the frame is not read we will end the loop.

7. DetectByPathimage() method

This method is used if a person needs to be detected from an image.

def detectByPathImage(path, output_path):
    image = cv2.imread(path)
    image = imutils.resize(image, width = min(800, image.shape[1])) 
    result_image = detect(image)
    if output_path is not None:
        cv2.imwrite(output_path, result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
8. Argparse() method

The function argparse() simply parses and returns as a dictionary the arguments passed through your terminal to our script. There will be Three arguments within the Parser:

Image: The path to the image file inside your system
Video: The path to the Video file inside your system
Camera: A variable that if set to ‘true’ will call the cameraDetect() method.
def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
    arg_parse.add_argument("-i", "--image", default=None, help="path to Image File ")
    arg_parse.add_argument("-c", "--camera", default=False, help="Set true if you want to use the camera.")
    arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file")
    args = vars(arg_parse.parse_args())
    return args
9. Main function

We have reached the end of our project.

if __name__ == "__main__":
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    args = argsParser()
    humanDetector(args)
