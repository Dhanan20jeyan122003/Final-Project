# heart-disease-prediction-Final-Year-Project
# 🫀 Final Heart Disease Prediction Project

This repository contains the frontend and backend code for a multimodal heart disease prediction system using:

- Clinical form data
- ECG images
- Chest X-ray images
- Echocardiogram videos

---

## 📦 Backend Model Files & Dependencies

The large model files and Python virtual environment packages **are not included in this repository** due to GitHub’s file size restrictions.

You can download them from this Google Drive link:

🔗 [Download Models & Packages from Google Drive](https://drive.google.com/drive/folders/1yikz8xOE52OobvCUsU3Om7DC9Tf57fye?usp=drive_link)


---


## ⚙️ Setup Instructions

### 🔧 Backend Setup

1. Navigate to the backend folder:
    ```bash
    cd backend
    ```

2. Create and activate a virtual environment (optional):
    ```bash
    python -m myenv
    venv\Scripts\activate  # On Windows
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the FastAPI server:
    ```bash
    uvicorn main:app --reload
    ```

---

### 🌐 Frontend Setup

1. Navigate to the frontend:
    ```bash
    cd my-heartcare-app
    ```

2. Install packages:
    ```bash
    npm install
    ```

3. Run the frontend:
    ```bash
    npm start
    ```

---

## 📝 Notes

- Ensure that the `models` folder contains:
  - `ecg_model.keras`
  - `xray_model.keras`
  - `echo_model.pth`
  - `clinical_model.pkl`

- Do **not** upload model files or virtual environments directly to GitHub — keep them in Google Drive or other storage.

---

## 📧 Contact

For any issues, please reach out at: [kmayan1967@gmail.com, mohamedhisam1100@gmail.com, jayasurya9247@gmail.com, murugan888311@gmail.com]

