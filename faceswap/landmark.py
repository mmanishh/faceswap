import os
import dlib
from imutils import face_utils


class FacialLandMark:
    def __init__(self, predictor_name="assets/shape_predictor_68_face_landmarks.dat"):
        file_path = os.path.join(os.path.dirname(__file__), predictor_name)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(file_path)

    def get_landmarks(self, frame):
        """
        Get facial landmark from dlib
        :param frame: image matix
        :return: list of facial landmarks
        """
        if frame is None:
            raise TypeError("Argument passed is None Type")

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
