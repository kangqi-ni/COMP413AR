import cv2
import mediapipe as mp
import math

def dist(lm1, lm2):
    return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)

def is_visible(landmark):
    return landmark.visibility > 0.9

# Sample Dist:
# 0.008447473081335715
# 0.01145010870364718
# 0.01150040239045948
# 0.011575607650855255
# 0.008996876389810183
# 0.009246202084822115
# 0.00936043135808242
# 0.005691222851731195
# 0.0008668815470799742
# 0.0036565868695992254
# 0.0016404391255509332
# 0.01324360790111218
# 0.011316379097523817
def match(ref_feat, feat):
    ref_encoding = [lm != None for lm in ref_feat]
    encoding = [lm != None for lm in feat]

    comp_encoding = [lm1 != lm2 for lm1, lm2 in zip(ref_encoding, encoding)]
    bi_distance = sum(comp_encoding)

    if bi_distance >= 2:
        return False

    distances = list()
    for lm1, lm2 in zip(ref_feat, feat):
        if lm1 is not None and lm2 is not None:
            distance = dist(lm1, lm2)
            distances.append(distance)

    if sum(distances) < 5.0:
        return True
    return False

def get_center(feat):
    x = 0.0
    y = 0.0
    z = 0.0
    count = 0
    for f in feat:
        if f is None:
            continue
        x += f.x
        y += f.y
        z += f.z

        count += 1

    return x/count, y/count, z/count

def guide(ref_feat, feat):
    ref_center = get_center(ref_feat)
    center = get_center(feat)

    diff = ref_center[0] - center[0], ref_center[1] - center[1], ref_center[2] - center[2]
    
    return diff

def get_messages(diff):
    # x > 0 -> camera need to move left
    # y > 0 -> camera need to move up
    # z > 0 -> camera need to move back
    messages = []

    if abs(diff[0]) > 0.1:
        messages.append("The camera needs to move left." if diff[0] > 0 else "The camera needs to move right.")
        
    if abs(diff[1]) > 0.1:
        messages.append("The camera needs to move up." if diff[1] > 0 else "The camera needs to move down.")
        
    if abs(diff[2]) > 0.2:
        messages.append("The camera needs to move back." if diff[2] > 0 else "The camera needs to move forward.")

    valid_photo = len(messages) == 0
    return valid_photo, messages

if __name__ == "__main__":
    # Initialize
    blur = True
    ref_feat = None
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    features = [mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_EYE_INNER,
        mp_pose.PoseLandmark.LEFT_EYE,
        mp_pose.PoseLandmark.LEFT_EYE_OUTER,
        mp_pose.PoseLandmark.RIGHT_EYE_INNER,
        mp_pose.PoseLandmark.RIGHT_EYE,
        mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.MOUTH_LEFT,
        mp_pose.PoseLandmark.MOUTH_RIGHT,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_PINKY,
        mp_pose.PoseLandmark.RIGHT_PINKY,
        mp_pose.PoseLandmark.LEFT_INDEX,
        mp_pose.PoseLandmark.RIGHT_INDEX,
        mp_pose.PoseLandmark.LEFT_THUMB,
        mp_pose.PoseLandmark.RIGHT_THUMB,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_HEEL,
        mp_pose.PoseLandmark.RIGHT_HEEL,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
    ]

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        # Read the frame
        _, frame = cap.read()

        try:
            # Process the frame for pose detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            pose_results = pose.process(frame_rgb)
            if blur:
                face_results = face_detection.process(frame_rgb)
        except:
            continue

        # Get the feature
        feat = list()
        for f in features:
            try:
                landmark = pose_results.pose_landmarks.landmark[f]

                if is_visible(landmark):
                    feat.append(landmark)
                else:
                    feat.append(None)
            except:
                feat.append(None)

        # Skip invalid features
        if all(landmark is None for landmark in feat):
            continue

        # Record the ref feature
        if ref_feat is None:
            ref_feat = feat
        # Match the ref feature
        else:
            direction = match(ref_feat, feat)
            diff = guide(ref_feat, feat)
            valid, messages = get_messages(diff)
            print(valid)
            print(messages)

        if blur and face_results.detections:
            for detection in face_results.detections:
                # Each detection has a location_data field that includes the bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                            int(bboxC.width * iw), int(bboxC.height * ih)

                # Blur the detected face
                face_region = frame[y:y+h, x:x+w]
                if len(face_region) == 0:
                    continue

                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                frame[y:y+h, x:x+w] = blurred_face

        # Draw the pose skeleton on the frame
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Output', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # Close the windows   
    cap.release()
    cv2.destroyAllWindows()