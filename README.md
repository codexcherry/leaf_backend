# Tomato Leaf Disease Detection - Backend

This is the backend service for the Tomato Leaf Disease Detection application. It provides a FastAPI-based REST API that can analyze tomato leaf images and detect diseases.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `model` folder and place your model file:
```bash
mkdir model
# Place tomato_model_state.pth in the model folder
```

3. Run the development server:
```bash
uvicorn main:app --reload
```

## API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Upload an image for disease prediction

## Deployment

This backend is configured for deployment on Vercel:

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Deploy:
```bash
vercel
```

## Environment Variables

Update the CORS settings in `main.py` with your frontend URL after deployment. 