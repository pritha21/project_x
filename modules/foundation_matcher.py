import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


class FoundationMatcher:
    """
    Finds the closest foundation match using CIEDE2000 (Lab space) for perceptual accuracy,
    and KDTree for fast nearest-neighbor search when scaling to 1000+ shades.
    """

    def __init__(self, csv_path="foundation_shades.csv"):
        """
        Loads foundation shades and precomputes LAB values and KDTree for fast matching.

        Args:
            csv_path (str): Path to CSV with columns: brand, shade_name, hex
        """
        self.shades = pd.read_csv(csv_path)

        if "hex" not in self.shades.columns:
            raise ValueError("CSV must contain a 'hex' column.")

        # Convert all hex values to RGB and then to Lab
        self.shades["rgb"] = self.shades["hex"].apply(self._hex_to_rgb)
        self.shades["lab"] = self.shades["rgb"].apply(self._rgb_to_lab)

        # Build KDTree on Lab values for efficient nearest-neighbor search
        lab_array = np.vstack(self.shades["lab"].tolist())
        self.kdtree = KDTree(lab_array)

    def _hex_to_rgb(self, hex_code):
        """
        Converts hex string (e.g., "#ddbba5") to (R, G, B) in 0-255.
        """
        hex_code = hex_code.lstrip("#")
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

    def _rgb_to_lab(self, rgb):
        """
        Converts RGB (0-255) to Lab using colormath.

        Args:
            rgb (tuple): (R, G, B) in 0–255

        Returns:
            tuple: (L, a, b)
        """
        srgb = sRGBColor(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
        lab = convert_color(srgb, LabColor)
        return (lab.lab_l, lab.lab_a, lab.lab_b)

    def _delta_e(self, lab1, lab2):
        """
        Computes perceptual difference using CIEDE2000.

        Args:
            lab1, lab2: Tuples (L, a, b)

        Returns:
            float: Delta E (lower = more similar)
        """
        c1 = LabColor(*lab1)
        c2 = LabColor(*lab2)
        return delta_e_cie2000(c1, c2)

    def match(self, user_hex, top_n=3):
        """
        Finds the top N best-matching foundation shades to user's skin tone.

        Args:
            user_hex (str): User's skin tone as hex code
            top_n (int): Number of best matches to return

        Returns:
            pd.DataFrame: Top N matching shades with delta E scores
        """
        # Convert user skin tone to Lab
        user_rgb = self._hex_to_rgb(user_hex)
        user_lab = self._rgb_to_lab(user_rgb)

        # Use KDTree to find top N closest Lab points
        distances, indices = self.kdtree.query(np.array(user_lab).reshape(1, -1), k=top_n)
        indices = indices[0]

        # Fetch the matched rows
        matches = self.shades.iloc[indices].copy()

        # Compute exact perceptual distances (CIEDE2000)
        matches["delta_e"] = matches["lab"].apply(lambda lab: self._delta_e(user_lab, lab))

        # Sort by perceptual similarity (in case KDTree used basic Euclidean)
        matches = matches.sort_values(by="delta_e").reset_index(drop=True)

        return matches[["brand", "shade_name", "hex", "delta_e"]]




if __name__ == "__main__":
    matcher = FoundationMatcher("foundation_shades.csv")
    results = matcher.match("#ddbba5", top_n=3)
    df = pd.DataFrame(results)
    df["delta_e"] = df["delta_e"].round(2)
    print(df)
