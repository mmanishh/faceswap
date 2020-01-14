import cv2
from faceswap import FaceSwap


def wrap_face():
    output = FaceSwap().wrap_face(
        src_img="data/manish2.jpg", dest_img="data/clinton.jpg"
    )
    output_path = "data/output.jpg"
    cv2.imshow("Face Swapped", output)
    cv2.waitKey(0)
    cv2.imwrite(output_path, output)
    print("Output file saved at path:", output_path)


if __name__ == "__main__":
    wrap_face()
