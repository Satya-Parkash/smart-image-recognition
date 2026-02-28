# Smart Image Recognition System (SIRS)

A production-ready image classification web app powered by **MobileNetV2**, **TensorFlow/Keras**, **OpenCV**, and **Flask**.  
Upload any image and receive the top-3 ImageNet predictions with confidence scores — in under a second.

---

## Screenshots

```
┌─────────────────────────────────────────────────────────┐
│  ◈ SIRS   Smart Image Recognition System · MobileNetV2  │
├───────────────────────┬─────────────────────────────────┤
│  01 / UPLOAD          │  02 / RESULTS                   │
│  ┌─────────────────┐  │  ┌─────────────────────────────┐│
│  │ Drop image here │  │  │ TOP PREDICTION              ││
│  │  or click       │  │  │ Golden Retriever            ││
│  └─────────────────┘  │  │ 96.42% confidence           ││
│  [ANALYSE IMAGE →]    │  │ ─────────────────────────── ││
│                       │  │ #1 Golden Retriever  96.42% ││
│                       │  │ ████████████████████████    ││
│                       │  │ #2 Labrador Retriever  2.3% ││
│                       │  │ ██                          ││
│                       │  └─────────────────────────────┘│
└───────────────────────┴─────────────────────────────────┘
```

---

## Project Structure

```
smart-image-recognition/
│
├── app.py                  ← Flask app, routes, upload handling
├── gunicorn.conf.py        ← Production WSGI server config
├── Procfile                ← For Render / Heroku
├── render.yaml             ← Render.com Blueprint deployment
├── requirements.txt        ← Python dependencies
├── .env.example            ← Environment variable template
├── .gitignore
│
├── model/
│   ├── __init__.py
│   └── classifier.py       ← MobileNetV2 + OpenCV preprocessing pipeline
│
├── templates/
│   └── index.html          ← Single-page upload + results UI
│
├── static/
│   ├── style.css           ← Industrial dark-mode design system
│   └── script.js           ← Drag-drop, AJAX fetch, results rendering
│
└── uploads/                ← Temporary image storage (auto-created)
```

---

## Tech Stack

| Layer        | Technology                          |
|:-------------|:------------------------------------|
| Model        | MobileNetV2 · ImageNet weights      |
| ML Framework | TensorFlow 2.17 / Keras             |
| Vision       | OpenCV (headless)                   |
| Backend      | Flask 3.0                           |
| Frontend     | Vanilla HTML/CSS/JS · Fetch API     |
| Server       | Gunicorn                            |
| Deploy       | Render · AWS EC2 / Elastic Beanstalk |

---

## Local Setup & Run

### Prerequisites

- Python 3.10 or 3.11 (recommended)
- `pip` and `venv`

### Step 1 — Clone or download

```bash
git clone https://github.com/yourname/smart-image-recognition.git
cd smart-image-recognition
```

### Step 2 — Create & activate a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** The first run downloads MobileNetV2 weights (~14 MB) from the Keras CDN automatically.  
> Subsequent runs use the cached copy (`~/.keras/models/`).

### Step 4 — Configure environment

```bash
cp .env.example .env
# Edit .env and set a strong SECRET_KEY if desired
```

### Step 5 — Run the development server

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### Step 6 — Run with Gunicorn (production-like locally)

```bash
gunicorn -c gunicorn.conf.py app:app
```

---

## Deploy on Render (Free Tier)

1. Push your project to a **GitHub** repository.
2. Go to [https://render.com](https://render.com) → **New → Web Service**.
3. Connect your GitHub repo.
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn -c gunicorn.conf.py app:app`
   - **Environment Variables:** `SECRET_KEY` = (generate a random string)
5. Click **Deploy**.

> ⚠️ Render's free tier has 512 MB RAM. TensorFlow needs ~800 MB–1 GB.  
> **Upgrade to the Starter plan ($7/mo)** for reliable inference, or use `tensorflow-cpu` and the Starter plan.

Alternatively, Render auto-detects `render.yaml` if you use **Blueprints**.

---

## Deploy on AWS EC2

### Option A — Manual EC2

```bash
# 1. Launch an EC2 instance (Ubuntu 22.04 LTS, t3.small or larger)
# 2. SSH into the instance
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# 3. Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git libgl1

# 4. Clone your repo
git clone https://github.com/yourname/smart-image-recognition.git
cd smart-image-recognition

# 5. Set up venv and install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 6. Set env variables
export SECRET_KEY="your-super-secret-key"
export FLASK_ENV=production

# 7. Run with Gunicorn (keep alive with screen or systemd)
screen -S sirs
gunicorn -c gunicorn.conf.py app:app
# Ctrl+A, D to detach

# 8. (Optional) Set up Nginx as a reverse proxy on port 80 → 5000
```

### Option B — AWS Elastic Beanstalk

```bash
pip install awsebcli
eb init -p python-3.11 smart-image-recognition
eb create sirs-env
eb deploy
```

EB auto-detects `Procfile` and uses Gunicorn.

---

## API Reference

### `GET /`
Returns the main HTML page.

### `POST /predict`

**Request:** `multipart/form-data` with field `image` (image file).

**Response (success):**
```json
{
  "success": true,
  "filename": "a3f1b2c4.jpg",
  "predictions": [
    { "label": "Golden Retriever", "confidence": 96.42 },
    { "label": "Labrador Retriever", "confidence": 2.31 },
    { "label": "Cocker Spaniel", "confidence": 0.71 }
  ]
}
```

**Response (error):**
```json
{
  "success": false,
  "error": "No image field in request."
}
```

---

## Model Details

| Property       | Value                              |
|:---------------|:-----------------------------------|
| Architecture   | MobileNetV2                        |
| Weights        | ImageNet (1000 classes)            |
| Input size     | 224 × 224 × 3                      |
| Preprocessing  | BGR→RGB (OpenCV), scale to [−1, 1] |
| Output         | Top-3 softmax predictions          |
| Inference time | ~100–500 ms on CPU                 |

---

## Customisation

**Swap to ResNet50:**

In `model/classifier.py`:
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

self.model = ResNet50(weights="imagenet", include_top=True)
INPUT_SIZE = (224, 224)  # Same for ResNet50
```

**Fine-tune on your own dataset:**  
Replace `include_top=False`, add your own `Dense` head, and train with `model.fit()`.

---

## License

MIT — free to use, modify, and deploy.