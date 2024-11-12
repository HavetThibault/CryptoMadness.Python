import cv2
import numpy as np
import scipy

VALID_IMG_EXT = ('jpg', 'png')

def rgb_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def grayscale_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def bicubic_resize(img, shape):
    return cv2.resize(img, dsize=shape, interpolation=cv2.INTER_CUBIC)


def binary_thresh(img, thresh):
    return cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]


def otsu_thresh(img):
    return cv2.threshold(img, 128, 255, cv2.THRESH_OTSU)[1]


# Return the same value as scipy lib method, but is much slower
def calculate_mass_center(img) -> tuple[float, float]:
    x_weighted_sum = 0.0
    y_weighted_sum = 0.0
    pixel_sum = 0.0
    width = len(img[0])
    height = len(img)
    for y in range(height):
        for x in range(width):
            pixel_value = img[y][x]/255.0
            x_weighted_sum += pixel_value * (x + 1)
            y_weighted_sum += pixel_value * (y + 1)
            pixel_sum += pixel_value
    return (x_weighted_sum / pixel_sum - 1,
            y_weighted_sum / pixel_sum - 1)


def center_of_mass_of_threshold(binary_img, threshold) -> tuple[float, float]:
    mass_y, mass_x = np.where(binary_img >= threshold)
    return np.average(mass_x), np.average(mass_y)


def center_of_mass(img):
    img_y, img_x = scipy.ndimage.center_of_mass(img)
    return int(round(np.average(img_x)) + 0.1), int(round(np.average(img_y)) + 0.1)


def square_crop(img):
    img_width = len(img[0])
    img_height = len(img)
    if img_width > img_height:
        square_crop_width = img_height
        square_crop_height = img_height
    else:
        square_crop_width = img_width
        square_crop_height = img_width

    return crop_by_center(img, img_width // 2, img_height // 2, square_crop_width // 2, square_crop_height // 2)


def crop_by_center(img, center_point_x, center_point_y, half_width, half_height):
    img_width = len(img[0])
    img_height = len(img)
    top_left_point = [center_point_x - half_width, center_point_y - half_height]
    bot_right_point = [center_point_x + half_width, center_point_y + half_height]

    if top_left_point[0] < 0:
        top_left_point[0] = 0
        bot_right_point[0] = 2 * half_width
    elif bot_right_point[0] >= img_width:
        bot_right_point[0] = img_width - 1
        top_left_point[0] = max(0, bot_right_point[0] - 2 * half_width)

    if top_left_point[1] < 0:
        top_left_point[1] = 0
        bot_right_point[1] = 2 * half_height
    elif bot_right_point[1] >= img_height:
        bot_right_point[1] = img_height - 1
        top_left_point[1] = max(0,bot_right_point[1] - 2 * half_height)

    return img[top_left_point[1]:bot_right_point[1], top_left_point[0]:bot_right_point[0]]


def remove_sensor_noise(img, calibration_img):
    subtracted_img = cv2.subtract(img, calibration_img)
    subtracted_img[np.where(subtracted_img < 0)] = 0
    white_img = np.full(calibration_img.shape, 255)
    pixel_range_length = np.subtract(white_img, calibration_img)
    rescaled_img = np.divide(white_img, pixel_range_length)
    return np.multiply(subtracted_img, rescaled_img).astype('uint8')


def set_pixels(frame: np.ndarray, min_value, max_value, new_value):
    low_value_filter = np.zeros_like(frame)
    low_value_filter[:, :] = min_value <= frame[:, :] <= max_value
    frame[low_value_filter.astype(bool)] = new_value
    return frame


def get_laser_center(img) -> tuple[int, int]:
    height = len(img)
    width = len(img[0])
    # Measured 969*1041 as the laser spot, so rounding a bit to get a smaller square 800x800
    laser_width = int(round(width / 3840 * 800) + 0.1)
    laser_height = int(round(height / 2748 * 800) + 0.1)

    mean_subtracted_img = np.subtract(img, np.mean(img))
    one_dim_shape = (mean_subtracted_img.shape[0] * mean_subtracted_img.shape[1],)
    one_dim_img = np.reshape(mean_subtracted_img, one_dim_shape)
    sorted_img = np.sort(one_dim_img)
    min_laser_value = sorted_img[height*width - laser_width*laser_height]

    mean_subtracted_img[np.where(mean_subtracted_img < min_laser_value)] = 0
    x_center_of_mass, y_center_of_mass = center_of_mass(mean_subtracted_img)
    return int(round(x_center_of_mass) + 0.1), int(round(y_center_of_mass) + 0.1)


def get_magnitude_spectrum(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    return 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
