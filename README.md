# Plant Disease Detection API

A FastAPI-based REST API for plant disease detection using EfficientNetB3 deep learning model.

## Features

- 🔍 Single image disease prediction
- 📦 Batch processing (up to 10 images)
- 🚀 High-performance asynchronous API
- 📊 Detailed prediction confidence scores
- 🛡️ Input validation and error handling
- 📝 Comprehensive logging
- 🔧 Modular and maintainable code structure

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd plant_disease_api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Place your model file (`best_model_32epochs.keras`) in the project root
2. Optionally create `.env` file for custom configuration:

```env
MODEL_PATH=your_custom_model.keras
HOST=0.0.0.0
PORT=8000
MAX_BATCH_SIZE=10
LOG_LEVEL=info
```

### Running the API

```bash
# Development mode
python main.py

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

- **GET /** - API information
- **POST /predict** - Single image prediction
- **POST /predict/batch** - Batch image prediction
- **GET /health** - Health check
- **GET /health/model** - Model information
- **GET /docs** - Interactive API documentation

### Usage Examples

#### cURL

```bash
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@plant_image.jpg"
```

#### Python

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("plant_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Project Structure

```
plant_disease_api/
├── main.py                 # FastAPI app entry point
├── config/
│   └── settings.py         # Configuration settings
├── models/
│   ├── schemas.py          # Pydantic models
│   └── ml_model.py         # ML model wrapper
├── services/
│   ├── image_service.py    # Image processing
│   └── prediction_service.py # Prediction logic
├── api/
│   ├── endpoints/
│   │   ├── health.py       # Health check endpoints
│   │   └── prediction.py   # Prediction endpoints
│   └── dependencies.py     # FastAPI dependencies
├── utils/
│   ├── logger.py           # Logging configuration
│   └── constants.py        # Constants and class names
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

## Supported Plant Diseases

The API can detect 38 different plant diseases across multiple crops including:

- Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

## License

MIT License
'''

print("File structure breakdown completed!")
print("\\nTo implement this structure:")
print("1. Create the directory structure as shown")
print("2. Create each file with its respective content")
print("3. Add **init**.py files to make directories Python packages")
print("4. Install dependencies: pip install -r requirements.txt")
print("5. Run: python main.py")
"""
