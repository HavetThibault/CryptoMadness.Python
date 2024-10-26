import cv2


def get_frame_nbr(video_path):
    video = None
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise Exception("Error opening video stream or file.")
        frame_nbr = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        if video is not None:
            video.release()
    return frame_nbr


def get_video_seconds(video_path):
    video = None
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise Exception("Error opening video stream or file.")
        frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video.get(cv2.CAP_PROP_FPS)
        seconds = round(frames / fps)
    finally:
        if video is not None:
            video.release()
    return seconds


def apply(video_path, first_frame_skip, frame_skip, frame_processor, verbose=True, frame_nbr=None):
    video = None
    result = []
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print("Error opening video stream or file")
        k = 0
        while video.isOpened():
            read_success = False
            frame = None
            if k == 0:
                for i in range(first_frame_skip):
                    read_success, frame = video.read()
                    if not read_success:
                        break
            else:
                for i in range(frame_skip):
                    read_success, frame = video.read()
                    if not read_success:
                        break
            if read_success:
                result.append(frame_processor(frame))
                if verbose:
                    print('Processed', k+1, 'image(s).')
            else:
                if verbose:
                    print('Processed', k+1, 'image(s).')
                break
            k += 1
            if frame_nbr is not None and k == frame_nbr:
                break
    finally:
        if video is not None:
            video.release()
    return result
