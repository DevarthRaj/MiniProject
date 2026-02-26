🩺 VITAL-AI: Intraoperative Hypotension Predictor
Introduction
VITAL-AI is a deep learning-based clinical tool designed to predict hypotensive events (dangerously low blood pressure) in ICU patients. By analyzing a 30-minute window of vital signs (Heart Rate, Mean BP, and Systolic BP), the system uses a Convolutional Neural Network (CNN) to identify risk patterns before a crash occurs.

The project handles real-world "messy" data using Linear Interpolation to fill gaps in patient recording times, ensuring the AI always sees a continuous trend.

🚀 How to Run the Backend
1. Prerequisites
Ensure you have Python 3.9+ installed. It is highly recommended to use a virtual environment.
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
2. Install Dependencies
pip install -r requirements.txt
3. Start the Server
Run the FastAPI server using Uvicorn:
uvicorn server:app --reload
The API will be live at: http://localhost:8000

Interactive Docs: Visit http://localhost:8000/docs to test the API by uploading a CSV file.
🖥️ Sample Frontend
Inside this repository, you will find a folder named sample-frontend (formerly icu-monitor).
Purpose: This folder contains a "ready-to-use" React/Vite dashboard that connects to the backend.
Note: This is provided as a sample implementation to show how the data should be visualized.
To run it:
cd sample-frontend
npm install
npm run dev