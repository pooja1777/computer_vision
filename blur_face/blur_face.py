
import mediapipe as mp
import cv2
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# normalize bounding boxes to pixel form
def normal(width, height):
  detection_results= []
  for detection in results.detections:
      bbox = detection.location_data.relative_bounding_box
      bbox_points = {
          "xmin" : int(bbox.xmin * width),
          "ymin" : int(bbox.ymin * height),
          "xmax" : int(bbox.width * width + bbox.xmin * width),
          "ymax" : int(bbox.height * height + bbox.ymin * height)
      }

      detection_results.append(bbox_points)
  return detection_results


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('face_blur.avi', fourcc, 20.0, (640,480) )

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    hi, wd,_ = image.shape
    if results.detections:
      for detection in results.detections:
        # detection.location_data.relative_bounding_box,
        valu = normal( wd, hi)
        xmi, ymn, xma, ymx = valu[0]['xmin'], valu[0]['ymin'], valu[0]['xmax'], valu[0]['ymax']
        img_crop = image[ymn:ymx, xmi:xma]
        # cv2.imshow("cropped_img", img_crop)
        blur_img = cv2.blur(img_crop,(33,33))
        image[ymn:ymx, xmi:xma] = blur_img
        # cv2.imshow("cropped_img", blur_img)
        image = cv2.rectangle(image, (xmi, ymn), (xma, ymx), (0, 255,255), 2)
    # Flip the image horizontally for a selfie-view display.
    out.write(image)
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
out.release()
cap.release()
cv2.destroyAllWindows()