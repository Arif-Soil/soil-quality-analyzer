import streamlit as st
import numpy as np
import cv2
import joblib
import os
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
import io

# -------------------- CONFIG -------------------------------
# Load environment variables (e.g., OpenAI API key)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client safely
client = None
if openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key)
    except Exception as e:
        # Use st.error which is visible even if sidebar is collapsed
        st.error(f"Failed to initialize OpenAI client: {e}", icon="🚨")
else:
    # Use st.warning which is visible
    st.warning("OpenAI API Key not found. AI Agronomist features will be disabled.", icon="⚠️")


# --- Page Configuration (Set First) ---
st.set_page_config(
    page_title="Mobile Soil Analyzer",
    layout="centered", # Centered layout is better for mobile feel
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Enhanced Styling ---
# (Using the mobile-optimized CSS from the previous response)
st.markdown("""
<style>
    /* OSU Header Styles - Slightly reduced size for mobile */
    .osu-header {
        text-align: center;
        margin-bottom: 30px; /* Reduced margin */
        padding: 15px; /* Reduced padding */
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 8px; /* Slightly smaller radius */
        border: 1px solid #dee2e6;
    }
    .osu-header h1 {
        color: #BB0000;
        font-size: 36px; /* Adjusted size */
        font-weight: 900; margin: 0; letter-spacing: 0.5px;
    }
    .osu-header h2 {
        color: #464646;
        font-size: 24px; /* Adjusted size */
        font-weight: 700; margin: 4px 0;
    }
    .osu-header h3 {
        color: #6c757d;
        font-size: 18px; /* Adjusted size */
        font-weight: 500; margin: 4px 0;
    }

    /* Main App Title - Made BIGGER */
    .stApp > header { background-color: transparent; }
    .main-title {
        font-size: 48px; /* << INCREASED FONT SIZE >> */
        font-weight: 800; color: #343a40; text-align: center;
        margin-bottom: 25px; line-height: 1.2;
    }

    /* General Styling */
    .stApp { background-color: #ffffff; }
    .block-container {
        padding: 1rem 1rem 2rem 1rem; /* Reduced padding */
        max-width: 700px;
    }
    .stButton button {
        background-color: #007bff; color: white; font-weight: 600;
        border-radius: 5px; padding: 12px 24px; border: none;
        transition: background-color 0.3s ease; width: 100%;
        margin-top: 10px;
    }
    .stButton button:hover { background-color: #0056b3; }
    .stFileUploader, .stCameraInput {
        border: 1px dashed #ced4da; padding: 15px; border-radius: 8px;
        background-color: #f8f9fa; margin-bottom: 10px;
    }
     /* Add specific spacing for camera permission note */
    .stCameraInput + .stCaption {
        margin-top: -5px; /* Pull caption closer */
        margin-bottom: 10px; /* Add space below caption */
        text-align: center;
    }


    /* Image Display - Constrained Size */
    .soil-image-container {
        display: flex; justify-content: center; margin-bottom: 15px;
    }
    .soil-image-container img {
        max-width: 100%; max-height: 300px; /* << LIMIT IMAGE HEIGHT >> */
        width: auto; height: auto; border-radius: 8px; border: 1px solid #dee2e6;
    }

    /* Results Display */
    .results-container {
        background: #f8f9fa; padding: 20px; border-radius: 8px;
        border: 1px solid #dee2e6; box-shadow: none; margin-top: 15px;
    }
    .results-header {
        font-size: 20px; font-weight: 600; color: #495057;
        margin-bottom: 15px; border-bottom: 1px solid #eee;
        padding-bottom: 8px; text-align: center;
    }
    .quality-category-label {
        font-size: 16px; color: #6c757d; display: block;
        text-align: center; margin-bottom: 5px;
    }
    .quality-category-value {
        font-size: 20px; font-weight: 700; padding: 5px 15px;
        border-radius: 5px; display: block; text-align: center;
        width: fit-content; margin: 0 auto 15px auto;
    }
    .score-progress-bar {
        height: 12px; background-color: #e9ecef; border-radius: 6px;
        overflow: hidden; margin: 5px 0 5px 0;
    }
    .score-progress-bar div {
        height: 100%; border-radius: 6px; transition: width 0.5s ease-in-out;
    }
    .score-display {
        text-align: center; font-size: 15px; font-weight: 600;
        color: #495057; margin-top: 8px;
    }

    /* AI Advice Box */
    .ai-advice-box {
        background: #eef7ff; padding: 15px; border-radius: 8px;
        border: 1px solid #cce0ff; margin-top: 15px; font-size: 15px;
        line-height: 1.5;
    }
    .ai-advice-header {
         font-size: 20px; font-weight: 600; color: #0056b3;
         margin-bottom: 10px; text-align: center;
    }
     h3:has(+ .stButton) { margin-bottom: 5px !important; } /* Spacing above AI button */

    /* Footer */
    .footer {
        text-align: center; color: #6c757d; margin-top: 40px;
        padding: 15px; border-top: 1px solid #dee2e6; font-size: 12px;
    }

</style>
""", unsafe_allow_html=True)

# -------------------- INSTITUTIONAL HEADER -----------------
st.markdown("""
<div class="osu-header">
    <h1>THE OHIO STATE UNIVERSITY</h1>
    <h2>OSU SOUTH CENTERS</h2>
    <h3>SOIL, WATER & BIOENERGY RESEARCH TEAM</h3>
</div>
""", unsafe_allow_html=True)

# -------------------- MAIN APP TITLE -----------------------
st.markdown('<p class="main-title">Soil Quality Analyzer<br>& AI Agronomist</p>', unsafe_allow_html=True)

# -------------------- LOAD MODEL ---------------------------
@st.cache_resource
def load_model(path="RandomForest_HSV_SoilScore_Model.pkl"):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Model file '{path}' not found. Please check the file path.", icon="🚨")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}", icon="🚨")
        return None

model = load_model()

# Proceed only if the model loaded successfully
if model:
    # -------------------- IMAGE INPUT --------------------------
    st.subheader("1. Provide Soil Image")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_image = st.file_uploader("Upload File", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    with col2:
        captured_image_bytes = st.camera_input("Use Camera", label_visibility="collapsed")
        # ********* Add Note about Permissions *********
        st.caption("Browser will ask for camera permission. Please allow it.")
        # **********************************************

    # Determine which image source to use
    image_source = None
    pil_image = None # Store the PIL image for later use

    if uploaded_image is not None:
        image_source = uploaded_image
        try:
            pil_image = Image.open(image_source).convert('RGB')
        except Exception as e:
            st.error(f"Error opening uploaded image: {e}", icon="️🚨")
            image_source = None # Reset if error

    elif captured_image_bytes is not None:
        image_source = captured_image_bytes # Keep track that camera was used
        try:
            # The camera_input returns bytes, need to open them with PIL
            pil_image = Image.open(io.BytesIO(captured_image_bytes.getvalue())).convert('RGB')
        except Exception as e:
            st.error(f"Error processing camera image: {e}", icon="🚨")
            image_source = None # Reset if error

    # -------------------- PROCESSING & DISPLAY ---------------------
    # Proceed only if we have a valid pil_image
    if pil_image:
        st.markdown("---") # Add a separator
        st.subheader("2. Analysis & Results")
        try:
            # Convert PIL image to NumPy array for OpenCV processing
            img_np = np.array(pil_image)
            hsv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

            # Calculate mean HSV values safely
            h_mean = np.mean(hsv_img[:, :, 0]) if hsv_img.size > 0 else 0
            s_mean = np.mean(hsv_img[:, :, 1]) if hsv_img.size > 0 else 0
            v_mean = np.mean(hsv_img[:, :, 2]) if hsv_img.size > 0 else 0

            # --- Display Image ---
            st.markdown('<div class="soil-image-container">', unsafe_allow_html=True)
            # Display the PIL image directly
            st.image(pil_image, caption="Soil Sample", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


            # --- Analysis Details (Expander) ---
            with st.expander("Show Color Analysis Details (HSV)"):
                col_h, col_s, col_v = st.columns(3)
                with col_h: st.metric(label="Hue (H)", value=f"{h_mean:.1f}")
                with col_s: st.metric(label="Saturation (S)", value=f"{s_mean:.1f}")
                with col_v: st.metric(label="Value (V)", value=f"{v_mean:.1f}")

            # --- Prediction & Results ---
            score = model.predict([[h_mean, s_mean, v_mean]])[0]
            score = max(0, min(100, score)) # Clamp score

            categories = {
                85: ("Excellent", "#2ecc71"), 70: ("Very Good", "#27ae60"),
                60: ("Good", "#f1c40f"), 50: ("Fair", "#e67e22"),
                40: ("Poor", "#e74c3c"), 0:  ("Very Poor", "#c0392b")
            }
            category = "Very Poor"; progress_color = "#c0392b"
            for cutoff, (cat, color) in sorted(categories.items(), reverse=True):
                if score >= cutoff:
                    category = cat; progress_color = color; break

            # --- Results Display (Mobile Optimized) ---
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            st.markdown('<p class="results-header">Soil Health Assessment</p>', unsafe_allow_html=True)
            category_html = f'<span class="quality-category-value" style="background-color: {progress_color}20; color: {progress_color}; border: 1px solid {progress_color}50;">{category}</span>'
            st.markdown('<span class="quality-category-label">Quality Category:</span>', unsafe_allow_html=True)
            st.markdown(category_html, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="score-progress-bar">
                    <div style="width: {score}%; background-color: {progress_color};"></div>
                </div>
                <div class="score-display">{score:.1f} / 100</div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True) # Close results-container

            # -------------------- AI ADVICE --------------------------
            if client:
                st.markdown("---") # Separator
                st.subheader("3. AI Agronomist Advice")
                if st.button("Generate Customized Advice"): # Button is full-width via CSS
                    with st.spinner("Connecting to AI Agronomist... Please wait."):
                        try:
                            system_prompt = """You are an expert AI agronomist specializing in precision agriculture for Ohio conditions.
                            Your recommendations should be:
                            1.  **Specific & Actionable**: Clear steps, rates, methods.
                            2.  **Contextualized**: Based on score category and HSV values (Low V ~ high OM; H ~ mineralogy/drainage; S ~ color purity).
                            3.  **Ohio-Focused**: Common Ohio soils, climate, crops (corn, soy, wheat, forage), practices (no-till, cover crops).
                            4.  **Sustainable**: Long-term health, erosion control, nutrient management.
                            5.  **Reference OSU Extension**: Suggest consulting OSU Extension.
                            6.  **Structured & Concise**: Use lists/headings. Be mobile-friendly (shorter paragraphs). Start with score interpretation.
                            """
                            user_prompt = f"""Analyze the following soil data and provide tailored, concise agronomic recommendations suitable for viewing on a mobile device:
                             - Soil Health Score: {score:.1f}/100
                             - Soil Quality Category: {category}
                             - Average Color Values (HSV): H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}

                             Focus on practical advice for an Ohio context."""

                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                temperature=0.5
                            )
                            ai_reply = response.choices[0].message.content

                            st.success("Analysis Complete!")
                            st.markdown(f'<p class="ai-advice-header">Customized Action Plan</p>', unsafe_allow_html=True)
                            st.markdown(f'<div class="ai-advice-box">', unsafe_allow_html=True)
                            st.markdown(ai_reply) # Render AI reply using st.markdown
                            st.markdown(f'</div>', unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"Could not retrieve AI advice: {str(e)}", icon="❌")
            else:
                 # Add margin if AI client isn't available
                 st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
                 st.info("AI Agronomist feature requires an OpenAI API Key.", icon="ℹ️")

        except Exception as e:
            st.error(f"An error occurred during image processing or prediction: {e}", icon="🚨")
            # st.exception(e) # Uncomment for detailed traceback during debugging

# -------------------- FOOTER --------------------------------
st.markdown("""
<div class="footer">
    Copyright © Dr. Arif Rahman & Rafiq Islam, OSU South Centers
</div>
""", unsafe_allow_html=True)

# ---- Placeholder if no image has been successfully processed yet ----
if 'pil_image' not in locals() or not pil_image:
     # Center the placeholder text
     st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
     st.info("Upload or capture a soil image using the options above to begin analysis.", icon="☝️")
     st.markdown("</div>", unsafe_allow_html=True)