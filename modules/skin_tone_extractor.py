# skin_tone_extractor.py
import cv2
import numpy as np
import face_recognition
import matplotlib.pyplot as plt

class SkinToneExtractor:
    def __init__(self, debug=False):
        self.debug = debug

    def _rgb_to_hex(self, rgb_color):
        return '#{:02x}{:02x}{:02x}'.format(int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))

    def _get_largest_face(self, face_locations, face_landmarks):
        if not face_locations:
            return None, None
        largest_face_index = 0
        max_area = 0
        for i, face_loc in enumerate(face_locations):
            top, right, bottom, left = face_loc
            area = (bottom - top) * (right - left)
            if area > max_area:
                max_area = area
                largest_face_index = i
        return face_landmarks[largest_face_index], face_locations[largest_face_index]

    def _get_forehead_box(self, landmarks, face_scale):
        if "left_eyebrow" not in landmarks or "right_eyebrow" not in landmarks:
            return None
        left_eyebrow = np.array(landmarks["left_eyebrow"])
        right_eyebrow = np.array(landmarks["right_eyebrow"])
        x1 = int(left_eyebrow[:, 0].min())
        x2 = int(right_eyebrow[:, 0].max())
        y2 = int(min(left_eyebrow[:, 1].min(), right_eyebrow[:, 1].min()))
        box_height = int(face_scale * 0.3)
        y1 = max(0, y2 - box_height)
        return (x1, y1, x2, y2)

    def _get_cheek_box(self, landmarks, face_scale):
        if "nose_bridge" not in landmarks or "left_eye" not in landmarks:
            return None
        nose_bridge = np.array(landmarks["nose_bridge"])
        left_eye = np.array(landmarks["left_eye"])
        cheek_center_x = int((nose_bridge[-1][0] + left_eye[0][0]) / 2)
        cheek_center_y = int((nose_bridge[-1][1] + left_eye[0][1]) / 2)
        box_size = int(face_scale * 0.2)
        half_box = box_size // 2
        return (
            cheek_center_x - half_box,
            cheek_center_y - half_box,
            cheek_center_x + half_box,
            cheek_center_y + half_box
        )

    def _get_neck_box(self, image, landmarks, face_scale):
        if "chin" not in landmarks:
            return None
        chin = np.array(landmarks["chin"])
        chin_center_x = int(np.mean(chin[:, 0]))
        chin_bottom_y = int(np.max(chin[:, 1]))
        gap_from_chin = int(face_scale * 0.1)
        box_height = int(face_scale * 0.25)
        box_width = int(face_scale * 0.4)
        half_width = box_width // 2
        y1 = chin_bottom_y + gap_from_chin
        y2 = y1 + box_height
        x1 = chin_center_x - half_width
        x2 = chin_center_x + half_width
        h, w, _ = image.shape
        return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))

    def _visualize_regions(self, image, regions, face_location):
        img_copy = image.copy()
        top, right, bottom, left = face_location
        cv2.rectangle(img_copy, (left, top), (right, bottom), (255, 0, 0), 2)
        for name, box in regions.items():
            if box:
                x1, y1, x2, y2 = box
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_copy, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

    def extract_skin_tone(self, image_path, face_locations=None):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not loaded: {image_path}")
        if not face_locations:
            return None, {"error": "No face locations provided."}

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_landmarks_list = face_recognition.face_landmarks(rgb_image, face_locations)
        if not face_landmarks_list:
            return None, {"error": "No facial landmarks found."}

        landmarks, face_location = self._get_largest_face(face_locations, face_landmarks_list)
        if landmarks is None or face_location is None:
            return None, {"error": "No valid face detected."}

        face_scale = landmarks["right_eyebrow"][-1][0] - landmarks["left_eyebrow"][0][0]
        regions = {
            "forehead": self._get_forehead_box(landmarks, face_scale),
            "cheek": self._get_cheek_box(landmarks, face_scale),
            "neck": self._get_neck_box(image, landmarks, face_scale)
        }

        rgb_values = {}
        for name, box in regions.items():
            if box is None or box[2] <= box[0] or box[3] <= box[1]:
                continue
            x1, y1, x2, y2 = box
            region = rgb_image[y1:y2, x1:x2]
            if region.size == 0:
                continue
            avg_color_rgb = np.mean(region.reshape(-1, 3), axis=0)
            rgb_values[name] = avg_color_rgb

        if not rgb_values:
            return None, {"error": "Could not sample valid skin regions."}

        overall_avg_rgb = np.mean(list(rgb_values.values()), axis=0)
        overall_avg_hex = self._rgb_to_hex(overall_avg_rgb)
        regional_hex_values = {k: self._rgb_to_hex(v) for k, v in rgb_values.items()}

        debug_img = self._visualize_regions(image, regions, face_location) if self.debug else None
        return overall_avg_hex, regional_hex_values, debug_img
