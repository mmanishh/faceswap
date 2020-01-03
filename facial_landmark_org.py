# import the necessary packages
import cv2
import imutils
import numpy as np
from landmark import FacialLandMark
from imutils.video import VideoStream
from scipy.spatial import distance


def blend_images(src, img, position, alpha=0):
    """
    blend two images
    :param src: src image
    :param img: img to blend
    :param position: (x,y) to place img in src
    :param alpha: degree of transparency value close to 1 indicates high transparency
    :return:
    """
    x_start, y_start = position
    x_end, y_end = x_start + img.shape[1], y_start + img.shape[0]
    added = cv2.addWeighted(src[y_start:y_end, x_start:x_end], alpha, img,
                            1 - alpha, 0)
    src[y_start:y_end, x_start:x_end] = added

    return src


def resize_image(img, scale_percent=40, scale=False, dim=(100, 20)):
    """
    resize the give image
    :param img: src image
    :param scale_percent: scale percent
    :param scale: flag to indicate if resize by scale percent
    :param dim: resize by dimension (x,y)
    :return:
    """
    if scale:
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    else:
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def mid_point(start, end):
    """
    return mid point between two coordinate
    :param start: start coordinate
    :param end: end coordinate
    :return: tuple of midpoint
    """
    return int((start[0] + end[0]) / 2), int((start[1] + end[1]) / 2)


class FacialCamera:

    def __init__(self):
        self.face_landmark = FacialLandMark()
        self.video_stream = VideoStream(usePiCamera=-1 > 0).start()

    def cal_size_clipart(self, img, start, end, nose):
        """
        calculate clipart size dynamically
        :param img: image to resize
        :param start: start point coordinate
        :param end: end point coordinate
        :param nose: coordinate of nose
        :return:
        """
        w = int(distance.euclidean(start, end))
        m = mid_point(start, end)
        h = int(distance.euclidean(m, nose))
        return resize_image(img, dim=(w, h))

    def load_cam(self, frame_name="Facial Landmark"):

        clip_art = cv2.imread("data/sunglasses_1.png")  # ,cv2.IMREAD_UNCHANGED)
        clip_art = resize_image(clip_art)

        i = 0
        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream, resize it to
            # have a maximum width of 400 pixels, and convert it to
            # grayscale
            i += 1
            frame = self.video_stream.read()
            frame = imutils.resize(frame, width=800)
            gray = cv2.flip(frame, 1)

            # detect facial landmark in the grayscale frame

            facial_landmarks = self.face_landmark.get_landmarks(gray)

            for faces in facial_landmarks:
                cv2.circle(gray, faces[17], 2, (0, 255, 255), 2)
                cv2.circle(gray, faces[30], 2, (0, 255, 255), 2)

                clip_art = self.cal_size_clipart(clip_art,
                                                 faces[17],  # left eyebrow starting coordinate
                                                 faces[26],  # right eyebrow end coordinate
                                                 faces[30])  # lower nose starting coordinate

                try:
                    gray = blend_images(gray, clip_art, faces[17])
                except ValueError as e:
                    print(e)

            # show the frame
            cv2.imshow(frame_name, gray)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        self.video_stream.stop()

        return True


"""
      (0, 16);     // Jaw line
      (17, 21);    // Left eyebrow
      (22, 26);    // Right eyebrow
      (27, 30);    // Nose bridge
      (30, 35);    // Lower nose
      (36, 41);    // Left eye
      (42, 47);    // Right Eye
      (48, 59);    // Outer lip
      (60, 67);    // Inner lip
"""

if __name__ == "__main__":
    FacialCamera().load_cam()
