# 🎄 Christmas Light Estimator

An intelligent web application that automates the tedious part of estimating Christmas light installations. Upload a photo of a house, let AI detect where lights and extension cords go, and export ready-to-use estimates.

[![React](https://img.shields.io/badge/React-18.3-blue?logo=react)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ (for frontend)
- Python 3.9+ (for backend/ML)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ammar8065/xmas_estimator.git
   cd xmas_estimator
   ```

2. **Setup Frontend**
   ```bash
   cd frontend
   npm install
   npm run build
   ```

3. **Setup Backend (Python)**
   ```bash
   cd ../backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Start the Backend Server**
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

5. **Start the Frontend (in another terminal)**
   ```bash
   cd frontend
   npm run dev
   ```

Visit `http://localhost:5173` in your browser.

---

## ✨ Features

### 🎯 AI-Powered Detection
- **Smart Fascia Detection**: Automatically detects where lights should go on rooflines and fascia
- **Cord Run Planning**: Suggests optimal extension cord routing paths
- **Confidence Scoring**: Low-confidence segments are visually flagged for review

### 🎨 Intelligent Review Canvas
- Interactive vector canvas built with React + Konva
- Layered editing: Toggle between lights and cord layers
- Fine-tune AI suggestions with intuitive drawing tools
- Real-time preview of changes

### 📤 Multiple Export Formats
- **PNG**: Quick preview images for proposals
- **PDF**: Editable vector PDFs that can be reopened and modified
- **SVG**: For further design work

### 🔄 Feedback Loop
- Capture human edits and corrections
- Training data flywheel for continuous model improvement
- Learn from your team's preferences and patterns

---

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────┐
│            Frontend (React + TypeScript)         │
│  Upload → Review Canvas (Konva) → Export Tools  │
│         Toggle Layers • Edit Lines               │
└──────────────────┬──────────────────────────────┘
                   │ HTTP/JSON
┌──────────────────▼──────────────────────────────┐
│         Backend API (FastAPI + PyTorch)         │
│  /projects  /upload  /infer  /export/pdf/png    │
│  Background inference jobs • ML model inference │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
   ┌────▼────┐          ┌────▼──────────┐
   │  Object │          │  ML Inference  │
   │ Storage │          │  (PyTorch +    │
   │ (S3 /   │          │   OpenCV)      │
   │ MinIO)  │          │  Segmentation  │
   └─────────┘          └────────────────┘
```

### Tech Stack

**Frontend**
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool & dev server
- **Konva** - Canvas library for vector drawing
- **ONNX Runtime** - In-browser ML inference (Phase 2)
- **pdf-lib** - PDF manipulation

**Backend**
- **FastAPI** - High-performance API framework
- **PyTorch** - Deep learning inference
- **Segmentation Models PyTorch** - Pre-trained segmentation models
- **OpenCV** - Image processing
- **Shapely** - Geometric operations

**ML**
- **UNet with ResNet34 backbone** - Semantic segmentation
- **scikit-image** - Image analysis
- **NumPy & Pillow** - Array/image handling

---

## 📋 Project Structure

```
xmas_estimator/
├── README.md                 # This file
├── BUILD_SPEC.md            # Detailed technical specification
├── CLAUDE.md                # Development context
├── .gitignore               # Git ignore rules
│
├── frontend/                # React + TypeScript application
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── App.tsx          # Main app component
│   │   └── styles/          # Styling
│   ├── public/              # Static assets
│   ├── package.json         # Frontend dependencies
│   ├── tsconfig.json        # TypeScript config
│   ├── vite.config.ts       # Vite configuration
│   └── index.html           # HTML entry point
│
├── backend/                 # FastAPI application
│   ├── app/
│   │   ├── main.py          # Main FastAPI app
│   │   ├── models.py        # Data models
│   │   ├── api/             # API endpoints
│   │   └── ml/              # ML integration
│   └── requirements.txt      # Python dependencies
│
├── ml/                      # Machine Learning pipeline
│   ├── train/               # Training scripts (Colab)
│   ├── inference/           # Inference pipeline
│   ├── labeling/            # Data labeling tools
│   ├── vectorize/           # PDF vector extraction
│   └── checkpoints/         # Model checkpoints
│
├── Data/                    # Training datasets
└── New Pictures/            # Qualitative test set

```

---

## 🎯 How It Works

### 1. **Upload Phase**
   - User uploads a photo of a house
   - Image is stored and project metadata is created

### 2. **AI Inference**
   - FastAPI backend loads the trained segmentation model
   - Model processes the image to detect:
     - **Fascia/Rooflines** (where lights go)
     - **Cord Runs** (where power extends)
   - Output is vector polylines with confidence scores

### 3. **Interactive Review**
   - Frontend renders the original photo + two editable layers
   - User can:
     - Accept AI suggestions
     - Edit lines and curves
     - Add missing segments
     - Remove incorrect segments
   - Low-confidence areas are visually highlighted

### 4. **Export**
   - Generate PNG for quick previews
   - Create editable PDF for team collaboration
   - All exports are rendered from the canonical vector data

---

## 🏋️ Training & Data

The ML model is trained on Google Colab with hundreds of manually-marked house photos:

- **~644 labeled PDF images** with vector markup
- **~347 unique houses** in the dataset
- Labels are vector overlays showing exact fascia and cord placement
- Model: **UNet with ResNet34 backbone**
- Training: GPU-accelerated on Colab (inference runs on CPU)

**To retrain the model:**
```bash
cd ml/train
# Upload dataset to Google Drive or mount Colab volume
# Run training notebook on Colab (training_pipeline.ipynb)
# Download best checkpoint to ml/checkpoints/
```

---

## 🚀 Deployment

### Local Development
```bash
# Terminal 1: Backend
cd backend && uvicorn app.main:app --reload

# Terminal 2: Frontend
cd frontend && npm run dev
```

### Production Build
```bash
# Frontend
cd frontend && npm run build  # Creates optimized dist/ folder

# Backend
cd backend && pip install -r requirements.txt
# Run with gunicorn or similar in production
```

### Docker (Optional)
```bash
docker-compose up
```

---

## 📊 Key Metrics

Track these to measure success:

- **Time Savings**: Markup time per house (target: 70-80% reduction)
- **Edit Rate**: % of AI suggestions kept by human reviewers (target: >70%)
- **Model Accuracy**: Fascia detection IOU, cord routing success rate
- **User Satisfaction**: Team feedback on usability

---

## 🔄 Continuous Improvement

1. **Capture Corrections**: Every human edit is logged
2. **Feedback Flywheel**: Use corrections as retraining data
3. **Iterative Training**: Periodically retrain on captured feedback
4. **Model Versioning**: Track model performance over time

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows the existing style
- Tests pass (when test suite is added)
- PRs include a clear description of changes

---

## 📝 Development Guidelines

### Frontend
- Use TypeScript for type safety
- Follow React hooks best practices
- Keep components small and focused
- Use Konva for canvas operations

### Backend
- Document API endpoints with docstrings
- Validate input thoroughly
- Handle errors gracefully
- Log inference performance

### ML
- Version all trained models
- Track training hyperparameters
- Document dataset composition
- Maintain train/val/test splits

---

## 🐛 Troubleshooting

### Model not loading?
```bash
# Check model file exists
ls -la backend/app/models/

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"
```

### Frontend not connecting to backend?
- Ensure backend is running on http://localhost:8000
- Check CORS settings in `backend/app/main.py`
- Verify API endpoint URLs in frontend config

### Out of memory during inference?
- Check image size (large images need more memory)
- Consider running inference on GPU if available
- Batch smaller images together

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the BUILD_SPEC.md for detailed technical context

---

## 🎄 Have a Holly Jolly Time!

This tool is designed to take the tedium out of Christmas light estimation so your team can focus on what they do best: installing beautiful, festive displays.

**Happy estimating!** 🎅✨
