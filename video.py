import cv2


def opencamera(frame_modifier=None):
    video_web = cv2.VideoCapture(0)

    while True:
        ret, frame = video_web.read()

        if ret:
            if frame_modifier:
                frame = frame_modifier(frame)
            cv2.imshow("screen", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_web.release()
    cv2.destroyAllWindows()


def detect_and_draw_bounding_box(frame):


    return frame
