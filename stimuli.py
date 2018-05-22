import numpy as np
import cv2
import scipy.signal
import matplotlib.pyplot as plt
from math import pi

def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

class Grating():
    def __init__(self):
        self.length = 1000  # lower resolution if images taking too long to load
        self.clear_plaid()

    def __repr__(self):
        return self.image

    def add_sinusoid(self, frequency, amp = 1, orientation=0):
        """frequency is in cycles/cm. Orientation is in degrees ccw"""
        x = np.linspace(0, 2*pi*5, self.length*2)
        y = amp * (np.sin(x*(frequency*2)) + 1)
        img = np.array([y]*self.length*2)
        (rows, cols) = np.shape(img)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), orientation, 1)
        img_rotated = cv2.warpAffine(img, M, (cols, rows))
        img_rotated_cropped = crop_around_center(img_rotated, self.length, self.length)
        self.image += img_rotated_cropped

    def add_gauss_window(self, size=0.4):
        window_matrix_vert = np.array([scipy.signal.general_gaussian(1000, 1, size*(self.length-1)/2)] * self.length)
        (rows, cols) = np.shape(window_matrix_vert)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
        window_matrix_horiz = cv2.warpAffine(window_matrix_vert, M, (cols, rows))
        self.image = self.image * window_matrix_vert * window_matrix_horiz
        
    def clear_plaid(self):
        self.image = np.zeros([self.length, self.length])

    def show_plaid(self):
        plt.axis("off")
        plt.imshow(self.image, cmap='gray')

    def crop_center(img, cropx, cropy):
        y, x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)   
        return img[starty:starty+cropy, startx:startx+cropx]

    def shape(self):
        return self.image.shape


    
p = Grating()
p.add_sinusoid(2, 100,  0) ## 2 works for 20cm projection
p.show_plaid()

