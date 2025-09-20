# Auto MPG MLOps Pipeline

A complete MLOps pipeline for Auto MPG prediction using Flask, Docker, GitHub Actions, and Jenkins.

## Features

- **Machine Learning Model**: Random Forest Regressor for MPG prediction
- **REST API**: Flask-based web service
- **CI/CD Pipeline**: GitHub Actions + Jenkins
- **Containerization**: Docker support
- **Code Quality**: flake8 integration
- **Unit Testing**: pytest with coverage
- **Admin Notifications**: Email alerts

## Project Structure

```
auto-mpg-mlops/
├── .github/workflows/     # GitHub Actions workflows
├── app/                   # Main application code
├── tests/                 # Unit tests
├── data/                  # Dataset
├── jenkins/               # Jenkins pipeline
├── Dockerfile             # Docker configuration
└── requirements.txt       # Python dependencies
```

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Predict MPG
- `GET /model/info` - Model information
- `POST /retrain` - Retrain model

## Usage

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run the application
python -m app.app
```

### Docker

```bash
# Build image
docker build -t auto-mpg-mlops .

# Run container
docker run -p 5000:5000 auto-mpg-mlops
```

## CI/CD Workflow

1. **Dev Branch**: Code quality checks (flake8)
2. **Test Branch**: Unit testing (pytest)
3. **Master Branch**: Jenkins builds Docker image and pushes to Docker Hub
4. **Admin Notification**: Email sent on 