import cv2
import numpy as np
import face_recognition

class SelfieValidation:

    """
    Validates a user's selfie based on visual quality and content.
    Criteria include front-facing pose, face and neck visibility,
    lighting, absence of makeup, and absence of filters.
    """

    def __init__(self):
    # Define the scoring weights for each criterion 
        self.criteria = {
            "front_facing": 20,
            "face_neck_visible": 20,
            "good_lighting": 20,
            "no_makeup": 20,
            "no_filters": 20
        }

    def validate_image(self, image_path):
        """
        Validates a selfie against multiple visual quality criteria.

        Args:
            image_path (str): Path to the input image.

        Returns:
            Tuple[bool, int, list, list[str]]: 
                - is_valid: Whether the selfie passed all critical checks
                - final score out of 100
                - face_locations (for reuse in skin tone module)
                - list of feedback messages
        """
        score = 100
        feedback = []

        image = cv2.imread(image_path)
        if image is None:
            return False, 0, [], ["Image not loaded. Please upload a valid image file."]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_height = image.shape[0]

        # 1. Face detection
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) != 1:
            score -= self.criteria["front_facing"]
            feedback.append("Make sure your face is clearly visible and facing the camera.")
        else:
            top, right, bottom, left = face_locations[0]
            margin_below_face = image_height - bottom
            if margin_below_face < (bottom - top) * 0.5:
                score -= self.criteria["face_neck_visible"]
                feedback.append("Ensure your neck is visible in the selfie.")

        # 2. Lighting check
        brightness = np.mean(gray.astype(float))
        if brightness < 80 or brightness > 200:
            score -= self.criteria["good_lighting"]
            feedback.append("Use even lighting without harsh shadows or overexposure.")

        # 3. Makeup detection (heuristic)
        makeup_detected = self.detect_makeup_heuristic(image, face_locations)
        if makeup_detected:
            score -= self.criteria["no_makeup"]
            feedback.append("Avoid heavy makeup that alters natural features.")

        # 4. Filter check
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            score -= self.criteria["no_filters"]
            feedback.append("Avoid heavy filters or overly smooth skin effects.")

        is_valid = len(face_locations) == 1 and score >= 75
        return is_valid, max(score, 0), face_locations, feedback


    def detect_makeup_heuristic(self, image, face_locations):
        """
        Enhanced heuristic to detect makeup using saturation comparison between face regions.
    
        Args:
        image: Original image (BGR)
        face_locations: List of face bounding boxes

        Returns:
        bool: True if makeup is likely present, False otherwise
        """
        if not face_locations:
            return False

        top, right, bottom, left = face_locations[0]
        face = image[top:bottom, left:right]

        h, w, _ = face.shape

        # Define regions as proportions of face dimensions
        forehead = face[int(h * 0.15):int(h * 0.30), int(w * 0.30):int(w * 0.70)]
        cheek = face[int(h * 0.55):int(h * 0.75), int(w * 0.20):int(w * 0.80)]
        jawline = face[int(h * 0.80):h, int(w * 0.30):int(w * 0.70)]

        # Skip if any region is empty
        if forehead.size == 0 or cheek.size == 0 or jawline.size == 0:
            return False

        # Convert to HSV
        forehead_hsv = cv2.cvtColor(forehead, cv2.COLOR_BGR2HSV)
        cheek_hsv = cv2.cvtColor(cheek, cv2.COLOR_BGR2HSV)
        jawline_hsv = cv2.cvtColor(jawline, cv2.COLOR_BGR2HSV)

        # Get average saturation
        avg_sat_forehead = np.mean(forehead_hsv[:, :, 1])
        avg_sat_cheek = np.mean(cheek_hsv[:, :, 1])
        avg_sat_jawline = np.mean(jawline_hsv[:, :, 1])

        # Makeup is likely if cheek saturation is significantly higher than both
        sat_diff_forehead = avg_sat_cheek - avg_sat_forehead
        sat_diff_jawline = avg_sat_cheek - avg_sat_jawline

        if sat_diff_forehead > 25 and sat_diff_jawline > 25:
            return True

        return False

