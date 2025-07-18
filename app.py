# app.py
import streamlit as st
import tempfile
from PIL import Image
import cv2
from modules.selfie_validation import SelfieValidation
from modules.skin_tone_extractor import SkinToneExtractor
from modules.foundation_matcher import FoundationMatcher

validator = SelfieValidation()
extractor = SkinToneExtractor(debug=True)
matcher = FoundationMatcher("foundation_shades.csv")

st.set_page_config(page_title="Foundation Matcher", layout="centered")
st.title("💄 AI Foundation Shade Matcher")

uploaded_file = st.file_uploader("Upload your selfie (face + neck visible, no makeup):", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        image_path = tmp_file.name

    st.image(Image.open(image_path), caption="Uploaded Selfie", use_column_width=True)

    st.subheader("📋 Selfie Quality Check")
    is_valid, score, face_locations, feedback = validator.validate_image(image_path)

    st.write(f"**Validation Score:** {score}/100")
    if feedback:
        st.warning("⚠️ " + "\n\n⚠️ ".join(feedback))
    else:
        st.success("✅ Selfie passed all validation checks!")

    if is_valid and len(face_locations) == 1:
        st.subheader("🎨 Extracting Skin Tone")
        overall_hex, hex_values, debug_image = extractor.extract_skin_tone(image_path, [face_locations[0]])

        if hex_values and overall_hex:
            st.image(debug_image, caption="Detected Regions", use_column_width=True)
            st.write("**Extracted Skin Tones:**")
            for region, hex_val in hex_values.items():
                st.markdown(f"- **{region.capitalize()}**: `{hex_val}`")

            user_hex = overall_hex
            st.write(f"\n🎯 **Matching Foundation for:** `{user_hex}`")

            st.subheader("💡 Recommended Foundation Shades")
            matches = matcher.match(user_hex, top_n=3)
            for idx, row in matches.iterrows():
                st.markdown(f"""
                **{idx + 1}. {row['brand']} – {row['shade_name']}**  
                HEX: `{row['hex']}`  
                🎯 Delta E: `{row['delta_e']:.2f}`
                """)
        else:
            st.error("❌ Skin tone extraction failed. Please try another image.")
    else:
        st.info("Please upload a better quality selfie to proceed with matching.")
