# Deepfake Detection: A Systematic Literature Review

An enterprise-grade, modernized digital media forensics platform based on the IEEE Access research paper:  
**"Deepfake Detection: A Systematic Literature Review" (IEEE Access, Volume 10)**.

The system analyzes digital video and media metadata, forensic token distribution, and transmission characteristics, coupled with a state-of-the-art **PyTorch MesoInception-4 Deep Learning Computer Vision Scanner** to classify media authenticity into **REAL** or **FAKE**.

---

## 🚀 Key Modernizations & Enhancements

1. **Real Deep Learning Media Scanner (`video_scanner`)**:
   - **MesoInception-4 CNN**: Implements multi-scale dilated convolutions (*Darius Afchar et al.*) to inspect facial micro-textures.
   - **2D-FFT Spectral Noise Forensics**: Analyzes high-frequency Fourier power spectra to identify GAN and Diffusion upsampling grid artifacts.
   - **Facial Boundary Disparity**: Detects mask seam anomalies and blending inconsistencies between face and neck.
   - **Frame-by-Frame Timeline**: Interactive Chart.js timeline showing manipulation probability over time.
   - **Keyframe Face Gallery**: Extracts facial crops with per-frame anomaly ratings.
   - **1-Click Quick Demo**: Built-in test buttons for instant evaluation of sample video and image clips.

2. **Django 5.x Full Upgrade**:
   - Upgraded from legacy Django 3.x patterns to Django 5.x.
   - Converted deprecated `url()` regex patterns to modern `django.urls.path()` routes.
   - Modernized `settings.py` with `pathlib.Path` and `BigAutoField`.

3. **Smart Database Architecture**:
   - Zero-configuration portable setup with automatic fallback to **SQLite** (`db.sqlite3`).
   - Seamless **MySQL** support via `pymysql` when a MySQL server is configured.

4. **High-Performance ML Engine (`deepfake_detection/ml_engine.py`)**:
   - **Persistent Model Caching**: Models are trained and serialized with `joblib`.
   - **Sub-Millisecond Inference**: Predictions now take **< 5 ms** (previously 20–30s per request).
   - **Benchmarked Classifiers**:
     - Multinomial Naive Bayes
     - Linear Support Vector Machine (LinearSVC)
     - Logistic Regression
     - Decision Tree Classifier
     - Gradient Boosting Classifier
     - Voting Ensemble Classifier (Hard Voting)
   - Evaluates Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

5. **Modernized UI / UX (2026 Dark Cyber-Forensic Theme)**:
   - Built on **Bootstrap 5.3**, **FontAwesome 6**, and modern Google typography (**Plus Jakarta Sans**).
   - Responsive layouts for desktops, tablets, and mobile devices.
   - Dynamic real-time charts using **Chart.js 4.4** (Pie, Doughnut, Bar, Line).
   - Convenient **⚡ Auto-Fill Sample Data** button for instant one-click demonstration.

6. **Modern Excel Export**:
   - Upgraded legacy binary `xlwt` (`.xls`) to modern styled OpenXML spreadsheets (`.xlsx`) powered by `openpyxl`.

---

## 📂 Project Structure

```
Deepfake_Detection/
│
├── requirements.txt                      # Pinned Python package dependencies
├── Database/
│   └── deepfake_detection.sql            # Original MySQL database dump
│
├── deepfake_detection/                   # Django Application Root
│   ├── manage.py                         # Django administrative utility
│   ├── seed_initial_data.py              # Database seeding script for sample records
│   ├── Datasets.csv                      # Primary forensic dataset (22,910 rows)
│   ├── Labeled_Data.csv                  # Exported labeled dataset
│   │
│   ├── deepfake_detection/               # Project Configuration & Core Services
│   │   ├── __init__.py                   # PyMySQL configuration
│   │   ├── settings.py                   # App settings & smart DB fallback
│   │   ├── urls.py                       # Modern Django 5.x URL routing
│   │   ├── ml_engine.py                  # Production-grade ML training & inference engine
│   │   ├── deep_learning/                # Computer Vision Forensics
│   │   │   ├── mesonet.py                # PyTorch MesoInception-4 CNN architecture
│   │   │   └── forensic_analyzer.py      # Video frame extraction, FFT & boundary analyzer
│   │   ├── wsgi.py & asgi.py
│   │
│   ├── Remote_User/                      # End-User Application
│   │   ├── models.py                     # Models for user accounts & predictions
│   │   ├── views.py                      # User auth, video scanner & deepfake prediction view
│   │   └── migrations/
│   │
│   ├── Service_Provider/                 # Admin / Service Provider Application
│   │   ├── views.py                      # Model retraining, analytics, and Excel export
│   │   └── models.py
│   │
│   └── Template/                         # Modern HTML Templates & Assets
│       ├── htmls/
│       │   ├── RUser/                    # User interface (Video Scanner, Login, Register, Predict, Profile)
│       │   └── SProvider/                # Admin interface (Dashboard, Charts, Logs, Users)
│       └── images/                       # UI assets
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- Python 3.10 to 3.13 installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Initialize Database & Seed Sample Records
Navigate into `deepfake_detection`:
```bash
python manage.py migrate
python seed_initial_data.py
```

### 4. Run the Development Server
```bash
python manage.py runserver
```
Visit **http://127.0.0.1:8000/** in your browser.

---

## 🔑 Default Credentials

| Portal | Role | URL | Username | Password |
|---|---|---|---|---|
| **Live Media Scanner** | Real-Time Video/Image Scanner | `http://127.0.0.1:8000/video_scanner/` | — | — |
| **Remote User** | Client / Analyst | `http://127.0.0.1:8000/` | `Rajesh` | `Rajesh` |
| **Service Provider** | Administrator | `http://127.0.0.1:8000/serviceproviderlogin/` | `Admin` | `Admin` |

*(Additional users can be registered at `/Register1/`)*

---

## 📜 License
This project is licensed under the MIT License.
