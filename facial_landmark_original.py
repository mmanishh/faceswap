# import the necessary packages
import cv2
import imutils
from landmark import FacialLandMark
from imutils.video import VideoStream


def load():
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(usePiCamera=-1 > 0).start()

    face_landmark = FacialLandMark()

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faciallandmark in the grayscale frame

        facial_landmarks = face_landmark.get_landmarks(gray)

        for faces in facial_landmarks:
            for each in faces:
                print(each)
                cv2.circle(frame, each, 1, (0, 0, 255), -1)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    load()
