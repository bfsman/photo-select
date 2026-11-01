# Wedding Smart Select — AI photo selection MVP

Python · Streamlit · OpenCV · pretrained YuNet · NumPy · Pandas

This project ranks photo candidates using manually designed sharpness, exposure, contrast and face-presence/relative-size scores. It does **not** infer beauty, eyes closed, emotion, relationships or artistic quality. The **atmosphere** scoring mode is a quality-weighting preset, not an emotion recognition model.

## Deploy on Streamlit Community Cloud
- Go to https://share.streamlit.io → Create app → Yup, I have an app.
- Repository: `bfsman/photo-select`
- Branch: `main`
- Main file path: `app.py`
- **Advanced settings: Python 3.11** (important for pinned dependencies).
- Click Deploy, then test the generated .streamlit.app URL with 2–5 non-sensitive, personally owned images.

No model binary is committed. On first analysis, the app downloads the official MIT-licensed OpenCV Zoo YuNet 2023mar model to a temporary file and verifies its SHA-256 digest. Internet connectivity is needed for the initial download.

## Safety and limitations
- This is a proof-of-concept, not a production photo processing service. Max 50 photos or 100MB per upload session.
- Uploading photos sends them to the Streamlit backend for analysis. Don't upload other people's photos without permission.
- The app itself doesn't write uploaded images to a database or disk. Cloud hosting privacy, uptime, authentication and data handling must be evaluated before collecting real customer data.
- Face-count matching previously measured on a limited synthetic dataset (Haar 7/19, YuNet 17/19) does not imply performance on arbitrary real wedding photos.
- AI-assisted software development was used; the developer performed design decisions, debugging, model integration and local checks.
