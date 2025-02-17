import sys
import os
import imutils
import numpy as np
import cv2
from imutils.video import VideoStream
from .landmark import FacialLandMark

file_path_haarcascade = os.path.join(
    os.path.dirname(__file__), "assets/haarcascade_frontalface_default.xml"
)


def get_hull8U(hull2):
    hull8U = []
    for i, _ in enumerate(hull2):
        hull8U.append((hull2[i][0], hull2[i][1]))

    return hull8U


def apply_affine_transform(src, src_tri, dst_tri, size):
    """
    Given a pair of triangles, find the affine transform.
    :param src:
    :param src_tri:
    :param dst_tri:
    :param size:
    :return:
    """
    warpMat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(
        src,
        warpMat,
        (size[0], size[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return dst


def rect_contains(rect, point):
    """
    Check if a point is inside a rectangle
    :param rect:
    :param point:
    :return:
    """
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


def warp_triangle(img1, img2, t1, t2):
    """
    Warps and alpha blends triangular regions from img1 and img2 to img
    :param img1:
    :param img2:
    :param t1:
    :param t2:
    :return:
    """

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
    # img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

    size = (r2[2], r2[3])

    img2Rect = apply_affine_transform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = img2[
        r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]
    ] * ((1.0, 1.0, 1.0) - mask)

    img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = (
        img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] + img2Rect
    )

    return True


def read_points(path):
    """
    read (x,y) points from file
    :param path: file path
    :return: list of points
    """
    # Create an array of points.
    points = []

    # Read points
    with open(path) as file:
        for line in file:
            x, y = line.split()
            points.append((int(x), int(y)))

    return points


def get_convex_hull(points1, points2):
    """
    Find convex hull
    :param points1: facial key point
    :param points2: facial key point
    :return: hull1,hull2
    """

    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

    for i, _ in enumerate(hullIndex):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])

    return hull1, hull2


def calculate_delaunay_triangles(rect, points):
    """
    calculate delanauy triangle
    :param rect:
    :param points:
    :return:
    """
    # create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList()

    delaunayTri = []

    pt = []

    for t in triangleList:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if (
            rect_contains(rect, pt1)
            and rect_contains(rect, pt2)
            and rect_contains(rect, pt3)
        ):
            ind = []
            # Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k, _ in enumerate(points):
                    if (
                        abs(pt[j][0] - points[k][0]) < 1.0
                        and abs(pt[j][1] - points[k][1]) < 1.0
                    ):
                        ind.append(k)
                        # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []

    return delaunayTri


class FaceSwap:
    def __init__(self):
        self.face_landmark = FacialLandMark()
        self.face_cascade = cv2.CascadeClassifier(file_path_haarcascade)
        self.video_stream = VideoStream(usePiCamera=-1 > 0).start()

    def detect_faces(self, img):
        """
        Detect faces using face haarcascade
        :param img:
        :return: (x,y,w,h) of first face
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img, 1.3, 5)
        if len(faces) >= 1:
            return faces[0]

    def find_delaunay(self, hull1, hull2, img1, img2, img1Warped):
        """
        Find delanauy traingulation for convex hull points
        :param hull1:
        :param hull2:
        :param img1:
        :param img2:
        :param img1Warped:
        :return:
        """
        sizeImg2 = img2.shape
        rect = (0, 0, sizeImg2[1], sizeImg2[0])

        dt = calculate_delaunay_triangles(rect, hull2)

        if len(dt) == 0:
            quit()

        # Apply affine transformation to Delaunay triangles
        for i, _ in enumerate(dt):
            t1 = []
            t2 = []

            # get points for img1, img2 corresponding to the triangles
            for j in range(0, 3):
                t1.append(hull1[dt[i][j]])
                t2.append(hull2[dt[i][j]])

            warp_triangle(img1, img1Warped, t1, t2)

    def wrap_face(self, src_img, dest_img):
        """
        Wrap Face
        :return: None
        """
        # Make sure OpenCV is version 3.0 or above
        (major_ver, minor_ver, subminor_ver) = cv2.__version__.split(".")

        if int(major_ver) < 3:
            print("ERROR: Script needs OpenCV 3.0 or higher")
            sys.exit(1)

        # Read images

        img1 = cv2.imread(src_img)
        img2 = cv2.imread(dest_img)
        img1Warped = np.copy(img2)

        # Raise TypeError if path is not found

        if img1 is None or img2 is None:
            raise TypeError(
                "Either {0} or {1} image is not found".format(src_img, dest_img)
            )

        # Read array of corresponding points
        # points1 = self.read_points(filename1 + '.txt')
        # points2 = self.read_points(filename2 + '.txt')

        points1 = self.face_landmark.get_landmarks(img1)[0]
        points2 = self.face_landmark.get_landmarks(img2)[0]

        hull1, hull2 = get_convex_hull(points1, points2)

        # cal delaunay
        self.find_delaunay(hull1, hull2, img1, img2, img1Warped)
        # get hull8U
        hull8U = get_hull8U(hull2)
        # Calculate Mask
        mask = np.zeros(img2.shape, dtype=img2.dtype)

        cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

        r = cv2.boundingRect(np.float32([hull2]))

        center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))

        # Clone seamlessly.
        output = cv2.seamlessClone(
            np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE
        )

        return output

    def load_cam(self, frame_name="Facial Landmark"):

        i = 0
        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream, resize it to
            # have a maximum width of 400 pixels, and convert it to
            # grayscale
            i += 1
            frame = self.video_stream.read()
            frame = imutils.resize(frame, width=800)
            frame = cv2.flip(frame, 1)

            if self.detect_faces(frame) is not None:
                (x, y, w, h) = self.detect_faces(frame)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # show the frame
            cv2.imshow(frame_name, frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        self.video_stream.stop()

        return True


if __name__ == "__main__":
    FaceSwap().wrap_face()
