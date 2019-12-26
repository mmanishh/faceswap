import dlib
from imutils import face_utils


class FacialLandMark:

    def __init__(self, predictor_name="shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_name)

    def get_landmarks(self, frame):
        landmarks = []
        # detect faces in the grayscale frame
        rects = self.detector(frame, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)

            faces = []
            # loop over the (x, y)-coordinates for the facial landmarks
            for (x, y) in shape:
                faces.append((x, y))
            landmarks.append(faces)

        return landmarks
