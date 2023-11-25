import os

import cv2
import numpy as np
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def overlayPNG(imgBack, imgFront, pos=[0, 0]):
    hf, wf, cf = imgFront.shape
    hb, wb, cb = imgBack.shape

    x1, y1 = max(pos[0], 0), max(pos[1], 0)
    x2, y2 = min(pos[0] + wf, wb), min(pos[1] + hf, hb)

    # For negative positions, change the starting position in the overlay image
    x1_overlay = 0 if pos[0] >= 0 else -pos[0]
    y1_overlay = 0 if pos[1] >= 0 else -pos[1]

    # Calculate the dimensions of the slice to overlay
    wf, hf = x2 - x1, y2 - y1

    # If overlay is completely outside background, return original background
    if wf <= 0 or hf <= 0:
        return imgBack

    # Extract the alpha channel from the foreground and create the inverse mask
    alpha = imgFront[y1_overlay:y1_overlay + hf, x1_overlay:x1_overlay + wf, 3] / 255.0
    inv_alpha = 1.0 - alpha

    # Extract the RGB channels from the foreground
    imgRGB = imgFront[y1_overlay:y1_overlay + hf, x1_overlay:x1_overlay + wf, 0:3]

    # Alpha blend the foreground and background
    for c in range(0, 3):
        imgBack[y1:y2, x1:x2, c] = imgBack[y1:y2, x1:x2, c] * inv_alpha + imgRGB[:, :, c] * alpha

    return imgBack

def scale_image(image, scale_factor):
    # Get the image dimensions
    height, width = image.shape[:2]

    # Calculate the center
    center = (width // 2, height // 2)

    # Perform the scaling transformation matrix
    scaling_matrix = cv2.getRotationMatrix2D(center, 0, scale_factor)

    # Apply the scaling transformation
    scaled_image = cv2.warpAffine(image, scaling_matrix, (width, height))

    return scaled_image

def try_on(frame,item_img):
    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # # Render detections
        # mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                        mp_draw.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=1),
        #                        mp_draw.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=1)
        #                        )
        
        try:
            # take the landmark of the left shoulder and right shoulder
            lm11 = results.pose_landmarks.landmark[11]
            lm12 = results.pose_landmarks.landmark[12]

            # take the landmark of the left hip and right hip
            lm23 = results.pose_landmarks.landmark[23]
            lm24 = results.pose_landmarks.landmark[24]

            # take img size
            h, w, c = image.shape

            # fine center of the body
            cx = int((lm11.x + lm12.x) * w // 2)
            cy = int((lm11.y + lm23.y) * h // 2)

            # draw circle on the center of the body
            cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # read the item image
            img_shirt = cv2.imread(item_img, cv2.IMREAD_UNCHANGED)

            # find center of the shirt
            img_shirt_h, img_shirt_w, _ = img_shirt.shape
            img_shirt_cx = int(img_shirt_w // 2)
            img_shirt_cy = int(img_shirt_h // 2)

            # scale the shirt image to fit the body
            diagonal_body = np.sqrt((lm11.x*w - lm24.x*w) ** 2 + (lm11.y*h - lm24.y*h) ** 2)
            diagonal_shirt = np.sqrt(img_shirt_h ** 2 + img_shirt_w ** 2)
            scale_factor = (diagonal_body / diagonal_shirt) * 1.5
            # print(scale_factor)
            img_shirt = scale_image(img_shirt, scale_factor)

            # overlay the center of the shirt on the center of the body
            try:
                coordinates = [cx - img_shirt_cx, cy - img_shirt_cy]
                image = overlayPNG(image, img_shirt, coordinates)
            except:
                pass
        except:
            pass

    return image

cap = cv2.VideoCapture(0)

item_img = "Shirts/5.png"

while cap.isOpened():
    ret, frame = cap.read()
    
    image = try_on(frame,item_img)
    
    cv2.imshow("Image", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
