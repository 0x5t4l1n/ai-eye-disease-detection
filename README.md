![image](https://github.com/user-attachments/assets/c26c02c6-614c-4672-974b-020df633032f)


# AI Eye Disease Detection



A web-based application for detecting glaucoma severity from retinal images using a ResNet101-based deep learning model with Squeeze-and-Excitation (SE) blocks. The app features a Flask backend for image processing and a frontend for user interaction. The pre-trained model is hosted on Hugging Face.

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Testing via API](#testing-via-api)
  - [Using the Frontend](#using-the-frontend)
- [Training the Model](#training-the-model)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)


## Project Overview
This application predicts glaucoma severity (Normal, Mild, Moderate, Severe) from retinal images by estimating the Cup-to-Disc Ratio (CDR) and providing confidence scores. The backend, served via Flask on port 5000, uses a pre-trained ResNet101 model. The frontend, served on port 8000, allows users to upload images and view results. The model is available at [Hugging Face](https://huggingface.co/5t4l1n/ai-eye-disease-detection/blob/main/model/best_glaucoma_model.pth), and the dataset includes images in `dataset/G1020/Images_Square/` (e.g., `237.jpg`).

## Prerequisites
- Python 3.10.16 (recommended for PyTorch compatibility)
- Git
- Dependencies listed in `backend/requirements.txt`
- GPU recommended for faster inference/training (CPU supported)
- Retinal images in `dataset/G1020/Images_Square/` (e.g., `237.jpg`)
- Pre-trained model from [Hugging Face](https://huggingface.co/5t4l1n/ai-eye-disease-detection/blob/main/model/best_glaucoma_model.pth)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Stalin-143/ai-eye-disease-detection.git
   cd ai-eye-disease-detection
   ```

2. **Create Virtual Environment**:
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   pip install --upgrade pip
   ```

3. **Install Dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
   If `torch==2.0.1` fails, try:
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2
   ```

4. **Download the Model**:
   ```bash
   mkdir -p backend/model
   wget https://huggingface.co/5t4l1n/ai-eye-disease-detection/resolve/main/model/best_glaucoma_model.pth -O backend/model/best_glaucoma_model.pth
   ```

5. **Configure Environment Variables**:
   Create `backend/.env`:
   ```plaintext
   FLASK_ENV=development
   FLASK_DEBUG=True
   FLASK_HOST=0.0.0.0
   FLASK_PORT=5000
   MODEL_PATH=./backend/model/best_glaucoma_model.pth
   MAX_CONTENT_LENGTH=16777216
   ALLOWED_ORIGINS=http://localhost:8000
   ```
   Update `MODEL_PATH` with the absolute path if needed (e.g., `/home/stalin/Projects/ai-eye-disease-detection/backend/model/best_glaucoma_model.pth`).

6. **Clear Cache**:
   ```bash
   rm -rf __pycache__ backend/__pycache__ backend/app/__pycache__
   ```

## Project Structure
```
ai-eye-disease-detection/
├── backend/
│   ├── app/
│   │   ├── main.py              # Flask app entry point
│   │   ├── routes/prediction.py # API endpoints for predictions
│   │   ├── utils/glaucoma_predictor.py # Model inference logic
│   ├── model/
│   │   ├── best_glaucoma_model.pth # Pre-trained model
│   ├── requirements.txt         # Backend dependencies
│   ├── .env                    # Environment variables
├── frontend/
│   ├── index.html              # Frontend UI
│   ├── script.js               # Frontend JavaScript
│   ├── styles.css              # Frontend styles
├── dataset/
│   ├── G1020/
│   │   ├── Images_Square/      # Retinal images (e.g., 237.jpg)
│   ├── training.ipynb          # Jupyter notebook for training
├── server.py                   # Runs backend and frontend
├── venv/                       # Virtual environment
├── backup.ipynb                # Backup notebook
├── models/                     # Additional models (optional)
├── README.md                   # Project documentation
├── requirements.txt            # Root-level dependencies (optional)
├── templates/                  # Flask templates (if used)
```

## Usage

### Running the Application
1. **Start Servers**:
   ```bash
   cd ai-eye-disease-detection
   source venv/bin/activate
   python server.py
   ```
   - Backend: `http://localhost:5000`
   - Frontend: `http://localhost:8000`
   Expected logs:
   ```
   INFO:__main__:Starting Flask backend on http://localhost:5000...
   Backend: DEBUG:__main__:✅ Glaucoma predictor initialized successfully!
   INFO:__main__:Starting frontend HTTP server on http://localhost:8000...
   INFO:__main__:Both servers started successfully!
   ```

   ![image](https://github.com/user-attachments/assets/e003b000-1eaa-4279-9894-f7e8d477f88a)


2. **Stop Servers**:
   Press `Ctrl+C`.

   

### Build Docker Image

#### Linux
```bash
docker build -t ai-eye-disease-detection .
```

#### Windows (Command Prompt or PowerShell)
```bash
docker build -t ai-eye-disease-detection .
```

#### Run Docker Container
Mount the `dataset/` directory to access images.

#### Linux
```bash
docker run -d \
  -p 5000:5000 \
  -p 8000:8000 \
  -v $(pwd)/dataset:/app/dataset \
  --name ai-eye-disease-container \
  ai-eye-disease-detection
```

#### Windows (PowerShell)
```powershell
docker run -d `
  -p 5000:5000 `
  -p 8000:8000 `
  -v ${PWD}/dataset:/app/dataset `
  --name ai-eye-disease-container `
  ai-eye-disease-detection
```

#### Windows (Command Prompt)
```cmd
docker run -d ^
  -p 5000:5000 ^
  -p 8000:8000 ^
  -v %CD%\dataset:/app/dataset ^
  --name ai-eye-disease-container ^
  ai-eye-disease-detection
```

### Notes
- `-v` mounts the local `dataset/` directory to `/app/dataset` in the container.
- Use backticks (`) in PowerShell or carets (^) in Command Prompt for line continuation.
- Ensure the `dataset/` directory exists locally.

### Verify Container

#### Linux/Windows
```bash
docker ps
docker logs ai-eye-disease-container
```

#### Expected Logs
```
INFO:__main__:Starting Flask backend on http://localhost:5000...
Backend: DEBUG:__main__:✅ Glaucoma predictor initialized successfully!
INFO:__main__:Starting frontend HTTP server on http://localhost:8000...
```
![image](https://github.com/user-attachments/assets/ae26563e-969f-4d69-a8d4-613b54f2ea1f)


### Testing via API
1. **Check Health**:
   ```bash
   curl http://localhost:5000/health
   ```
   Expected:
   ```json
   {"status":"healthy","model_loaded":true}
   ```
   ![image](https://github.com/user-attachments/assets/ffeef47a-95e3-4ef8-9dc8-1bffe53b7fcb)


2. **Get Model Info**:
   ```bash
   curl http://localhost:5000/api/model-info
   ```

   
   Expected:
   ```json
   {"success":true,"data":{"model_type":"GlaucomaSeverityModel","architecture":"ResNet101 with SE Blocks",...}}
   ```
    ![image](https://github.com/user-attachments/assets/c4385ff4-cbaf-4ba3-8eca-2384984f931d)

3. **Predict Glaucoma**:
   ```bash
   curl -F "image=@dataset/G1020/Images_Square/237.jpg" -F "return_probabilities=true" http://localhost:5000/api/predict
   ```
  

   Expected:
   ```json
   {"success":true,"data":{"cdr":0.45,"severity":"Normal","severity_description":"No significant glaucoma signs","confidence":0.95,"risk_level":"Low","probabilities":{"Normal":0.95,"Mild":0.03,"Moderate":0.01,"Severe":0.01}},"filename":"237.jpg"}
   ```
   ![image](https://github.com/user-attachments/assets/87c86959-0202-4843-aa80-752445bed7fe)

### Using the Frontend
1. Open `http://localhost:8000` in a browser.
2. **Get Model Info**: Click “Get Model Info” to view model details.
3. **Predict Glaucoma**:
   - Upload an image (e.g., `016.jpg`).
   - Check “Show Probabilities” (optional).
   - Click “Predict” to view results.
  
    ![image](https://github.com/user-attachments/assets/b6ebdf21-54ca-4224-b890-2683d6ffc653)



## Training the Model
1. **Prepare Dataset**:
   - Ensure `dataset/G1020/Images_Square/` has labeled images.
   - Check labels in a CSV or directory structure (e.g., `Normal/`, `Mild/`).
   - Run:
     ```bash
     ls dataset/G1020
     ```

2. **Install Jupyter**:
   ```bash
   source venv/bin/activate
   pip install jupyter
   ```

3. **Run training.ipynb**:
   ```bash
   jupyter notebook dataset/training.ipynb
   ```
   - Update dataset path (e.g., `dataset/G1020/Images_Square/`).
   - Configure model (ResNet101 with SE blocks).
   - Adjust hyperparameters (e.g., learning rate, batch size).
   - Example:
     ```python
     import torch
     import torchvision
     from torch.utils.data import DataLoader
     from torchvision import transforms
     transform = transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
     dataset = torchvision.datasets.ImageFolder('dataset/G1020/Images_Square/', transform=transform)
     loader = DataLoader(dataset, batch_size=32, shuffle=True)
     model = torchvision.models.resnet101(pretrained=True)
     model.fc = torch.nn.Linear(model.fc.in_features, 4)
     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
     criterion = torch.nn.CrossEntropyLoss()
     ```

4. **Save Model**:
   ```python
   torch.save(model.state_dict(), 'backend/model/new_glaucoma_model.pth')
   ```

5. **Update .env**:
   Set `MODEL_PATH` to the new model in `backend/.env`.

## Troubleshooting
- **Model Not Loaded**:
  - Check:
    ```bash
    ls backend/model/
    ```
  - Verify `MODEL_PATH` in `backend/.env`.
  - Share logs.

- **Prediction Fails**:
  - Run:
    ```bash
    curl -F "image=@dataset/G1020/Images_Square/237.jpg" -F "return_probabilities=true" http://localhost:5000/api/predict
    ```
  - Share output and logs.
  - Check browser console (F12).

- **Pip Install Fails**:
  - Share error. Try:
    ```bash
    pip install torch==2.0.1 torchvision==0.15.2
    ```

## Contributing
1. Fork: `https://github.com/Stalin-143/ai-eye-disease-detection`
2. Branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a pull request.

## Authors

- [@Stalin-143](https://www.github.com/Stalin-143)




