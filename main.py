from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
results = model("images.jpg")


def is_person_on_ground(keypoints):
    left_foot_index = 15
    right_foot_index = 16
    ground_threshold = 0.3

    if keypoints.shape[0] < 17:
        return False

    left_foot = keypoints[left_foot_index]
    right_foot = keypoints[right_foot_index]

    # Check if confidence is high enough and y-coordinate indicates on ground
    left_on_ground = left_foot[2] > 0.5 and left_foot[1] > ground_threshold
    right_on_ground = right_foot[2] > 0.5 and right_foot[1] > ground_threshold

    return left_on_ground or right_on_ground


for result in results:
    kpts = result.keypoints.data

    if kpts is not None and len(kpts) > 0:
        for i, person_keypoints in enumerate(kpts):
            if is_person_on_ground(person_keypoints):
                print(f"Person {i} is on the ground.")
            else:
                print(f"Person {i} is not on the ground.")
