# Chatbot Router Demo - Getting Started Guide

This guide will walk you through setting up and running the complete chatbot router application, including both the backend API and frontend chat interface.

## What This Application Does

The Chatbot Router Demo is an intelligent chatbot that can:
- Route user messages to different applications/intents (like attendance, leave management, etc.)
- Use machine learning to understand user intent even with typos or informal language
- Provide responses based on the detected intent
- Handle out-of-scope (OOS) queries gracefully

## Prerequisites

Before starting, make sure you have:
- Python 3.9+ installed
- Node.js 16+ and npm installed
- Terminal/command line access

## Step 1: Set Up the Backend (Python API)

### 1.1 Create Python Environment
```bash
# Navigate to the project directory
cd chatbot-router-demo

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows
```

### 1.2 Install Dependencies
```bash
pip install -r requirements.txt
```

### 1.3 Generate Training Data
```bash
python scripts/generate_synth_data.py --out data/synth_dataset.csv
```

### 1.4 Train the Model
```bash
# Set the embedding model (you can choose different models)
export EMBED_MODEL=sentence-transformers/all-mpnet-base-v2

# Train the router with both multiclass and OOS detection
PYTHONPATH=. python scripts/train_combo.py --data data/synth_dataset.csv \
  --min-oos-recall 0.85 --target-oos-precision 0.90
```

This creates:
- `models/router.joblib` - Main classification model
- `models/oos_head.joblib` - Out-of-scope detection model
- `config/router_thresholds.json` - Decision thresholds

### 1.5 Start the Backend Server
```bash
# Disable Gemini API for local testing (optional)
unset GEMINI_API_KEY

# Set the embedding model
export EMBED_MODEL=sentence-transformers/all-mpnet-base-v2

# Start the server
PYTHONPATH=. uvicorn app.main:app --reload
```

The backend will start at `http://127.0.0.1:8000`

## Step 2: Set Up the Frontend (React Chat Interface)

### 2.1 Navigate to Frontend Directory
```bash
# Open a new terminal window/tab and navigate to the frontend
cd chatbot-router-demo/my-chat
```

### 2.2 Install Frontend Dependencies
```bash
npm install
```

### 2.3 Start the Frontend Development Server
```bash
npm run dev
```

The frontend will start at `http://localhost:5175` (or another port if 5175 is busy)

## Step 3: Test the Application

### 3.1 Using the Web Interface
1. Open your browser and go to `http://localhost:5175`
2. You'll see a chat interface with the title "Chatbot Router Demo"
3. Try these example messages:

**Attendance queries:**
- "who was absent last week"
- "show me attendance for today"
- "mark Manju absent today"

**Leave queries:**
- "I want to apply for leave"
- "check my leave balance"
- "approve John's leave request"

**Out-of-scope queries:**
- "what's the weather today"
- "leave-one-out cross validation"
- "how to cook pasta"

### 3.2 Using Command Line (Alternative)
You can also test the backend directly:
```bash
# Test attendance query
curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"who was absent last week"}'

# Test leave query
curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"I want to apply for leave"}'

# Test out-of-scope query
curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"what is machine learning"}'
```

## How It Works

### Backend Architecture
1. **Message Processing**: When you send a message, it goes through multiple layers:
   - Rule-based matching (keywords and negative keywords)
   - Embedding-based similarity matching
   - Machine learning classification
   - Out-of-scope detection

2. **Intent Detection**: The system tries to match your message to predefined intents like:
   - `attendance_query` - Questions about attendance
   - `attendance_mark` - Marking someone absent/present
   - `leave_apply` - Applying for leave
   - `leave_query` - Checking leave status

3. **Response Generation**: Based on the detected intent, it either:
   - Routes to a specific application handler
   - Returns a helpful response
   - Indicates the query is out-of-scope

### Frontend Features
- **Real-time Chat**: Type messages and get instant responses
- **Intent Display**: Shows detected intent and confidence level
- **Clean Interface**: Simple, responsive design that works on all devices

## Configuration Files

- `apps.yaml` - Defines available applications and their intents
- `config/router_thresholds.json` - Controls decision thresholds
- `my-chat/vite.config.ts` - Frontend proxy configuration

## Troubleshooting

### Backend Issues
- **Port already in use**: Change the port with `uvicorn app.main:app --reload --port 8001`
- **Model not found**: Make sure you ran the training step
- **Import errors**: Ensure `PYTHONPATH=.` is set

### Frontend Issues
- **Proxy errors**: Make sure the backend is running on port 8000
- **Styling issues**: Ensure Tailwind CSS is properly installed
- **Port conflicts**: The frontend will automatically find an available port

### Common Solutions
- **Connection refused**: Make sure both backend (port 8000) and frontend are running
- **Slow first response**: The first request downloads ML models, subsequent requests are faster
- **Memory issues**: The ML models require some RAM; close other applications if needed

## Next Steps

Once everything is working:
1. Explore the `apps.yaml` file to understand available intents
2. Try modifying the training data in `scripts/generate_synth_data.py`
3. Experiment with different embedding models
4. Add new intents and retrain the model
5. Customize the frontend styling and layout

## Getting Help

- Check the detailed `docs/RUNBOOK.md` for advanced configuration
- Look at the training scripts in the `scripts/` directory
- Examine the router logic in `app/router.py`
- Review the chat component in `web/src/Chat.tsx`

Happy chatting! 🤖