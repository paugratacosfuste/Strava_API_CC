# 🏃 Strava Analytics Dashboard

[![AWS](https://img.shields.io/badge/AWS-Serverless-FF9900?logo=amazon-aws)](https://aws.amazon.com/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://python.org/)
[![Strava](https://img.shields.io/badge/Strava-API-FC4C02?logo=strava)](https://developers.strava.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An ML-powered running analytics platform that provides personalized pace predictions, training insights, and performance tracking using your Strava data.

![Dashboard Preview](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Deployment](#-deployment)
- [API Endpoints](#-api-endpoints)
- [Environment Variables](#-environment-variables)
- [Usage](#-usage)
- [Team](#-team)
- [License](#-license)

---

## ✨ Features

### 🔮 ML-Powered Pace Predictions
- **Random Forest model** trained on your personal running data
- Predictions based on distance, terrain, effort level, and heart rate zones
- Model metrics displayed (MAE, RMSE, R²) for transparency

### 📊 Comprehensive Dashboard
- **Statistics Overview**: Total runs, distance, time, elevation, average pace
- **Best Efforts**: Personal records for 400m, 1K, 5K, 10K, Half Marathon, Marathon
- **Training Trends**: Weekly distance, pace, heart rate, and elevation charts with forecasts
- **Heart Rate Zones**: Personalized zones based on your max HR

### 📅 Smart Training Plans
- **Auto-generated weekly plans** based on your training history
- **Acute/Chronic load ratio** analysis for injury prevention
- **Progressive overload** recommendations
- Session types: Intervals, Tempo, Long Run, Recovery

### 🏆 Leaderboard
- Compare best efforts across all users
- Rankings for all standard distances
- Combined OAuth and CSV user data

### 📤 Flexible Data Input
- **Strava OAuth**: Direct sync with your Strava account
- **CSV Upload**: Manual upload of Strava export data (for API rate limits)

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           FRONTEND                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │ CloudFront  │───▶│  S3 Bucket  │───▶│  Static HTML/CSS/JS     │ │
│  │    (CDN)    │    │             │    │  - index.html           │ │
│  └─────────────┘    └─────────────┘    │  - dashboard.html       │ │
│                                         │  - training_plan.html   │ │
│                                         │  - leaderboard.html     │ │
│                                         └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          API LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      API Gateway                             │   │
│  │  /strava_callback  /predict  /stats  /trends  /leaderboard  │   │
│  │  /csv-upload       /dashboard        /plan                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                    │                                 │
│                                    ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Lambda Functions (10)                     │   │
│  │  ┌──────────────────┐  ┌──────────────────┐                 │   │
│  │  │ AuthCallback     │  │ DataFetcher      │                 │   │
│  │  │ DataProcessor    │  │ CSVProcessor     │                 │   │
│  │  │ Predictor        │  │ StatsCalculator  │                 │   │
│  │  │ TrendsCalculator │  │ Leaderboard      │                 │   │
│  │  │ DashboardAPI     │  │ TrainingPlan     │                 │   │
│  │  └──────────────────┘  └──────────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                    │                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Lambda Layer (pandas, numpy, sklearn)           │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    DynamoDB Tables (9)                       │   │
│  │                                                               │   │
│  │  OAuth Tables:              CSV Tables:                      │   │
│  │  • strava_tokens_pau        • strava_activities_csv          │   │
│  │  • strava_activities_pau    • strava_processed_csv           │   │
│  │  • strava_processed_pau     • strava_best_efforts_csv        │   │
│  │  • strava_best_efforts_pau  • strava_predictions_csv         │   │
│  │  • strava_predictions_pau                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                    │                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      Strava API                              │   │
│  │              (OAuth 2.0 / Activity Data)                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **AWS Lambda** | Serverless compute for all backend logic |
| **AWS DynamoDB** | NoSQL database for activities and predictions |
| **AWS API Gateway** | RESTful API endpoints |
| **Python 3.12** | Runtime for Lambda functions |
| **pandas** | Data manipulation and analysis |
| **scikit-learn** | Random Forest ML model for predictions |
| **numpy** | Numerical computations |

### Frontend
| Technology | Purpose |
|------------|---------|
| **AWS S3** | Static website hosting |
| **AWS CloudFront** | CDN for global distribution and HTTPS |
| **HTML5/CSS3** | Responsive UI with dark theme |
| **JavaScript (Vanilla)** | Client-side interactivity |
| **Chart.js** | Data visualization |

### External APIs
| API | Purpose |
|-----|---------|
| **Strava API v3** | OAuth authentication and activity data |

---

## 📁 Project Structure

```
strava-analytics-dashboard/
│
├── 📂 lambda-functions/
│   ├── StravaAuthCallBackHandler.py   # OAuth callback handling
│   ├── StravaDataFetcher.py           # Fetch activities from Strava API
│   ├── StravaDataProcessor.py         # ML feature engineering (OAuth)
│   ├── StravaCSVProcessor.py          # Process CSV uploads
│   ├── StravaPredictor.py             # Random Forest pace predictions
│   ├── StravaStatsCalculator.py       # Calculate user statistics
│   ├── TrainingTrendsCalculator.py    # Weekly trends & forecasts
│   ├── StravaLeaderboard.py           # Cross-user leaderboard
│   ├── StravaDashboardAPI.py          # Dashboard data aggregation
│   └── weekly_training_plan.py        # Auto-generate training plans
│
├── 📂 frontend/
│   ├── index.html                     # Landing page with login options
│   ├── csv_upload.html                # CSV upload interface
│   ├── input_page.html                # Prediction parameters input
│   ├── dashboard.html                 # Main analytics dashboard
│   ├── training_plan.html             # Weekly training plan view
│   ├── leaderboard.html               # Global leaderboard
│   └── documentation.html             # User documentation
│
├── 📂 docs/
│   └── deployment_guide.html          # Step-by-step AWS deployment guide
│
├── .gitignore                         # Git ignore rules
├── README.md                          # This file
└── LICENSE                            # MIT License
```

---

## 🚀 Getting Started

### Prerequisites

- **AWS Account** with billing enabled
- **Python 3.12** installed locally
- **Strava Account** for API access
- **Docker** (recommended for building Lambda layer)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/strava-analytics-dashboard.git
   cd strava-analytics-dashboard
   ```

2. **Create Strava API Application**
   - Go to [Strava API Settings](https://www.strava.com/settings/api)
   - Create a new application
   - Note your Client ID and Client Secret

3. **Follow the Deployment Guide**
   - Open `docs/deployment_guide.html` in your browser
   - Follow the step-by-step instructions

---

## 📦 Deployment

For complete deployment instructions, see the **[Deployment Guide](docs/deployment_guide.html)**.

### Summary of Steps

1. **Create DynamoDB Tables** (9 tables)
2. **Build Lambda Layer** (pandas, numpy, scikit-learn)
3. **Deploy Lambda Functions** (10 functions)
4. **Configure API Gateway** (8 endpoints)
5. **Setup S3 Bucket** (static hosting)
6. **Create CloudFront Distribution** (CDN + HTTPS)
7. **Configure Strava OAuth** (callback URL)
8. **Update Frontend URLs** (API Gateway, S3, CloudFront)

### Estimated Deployment Time
- First-time setup: **2-3 hours**
- Subsequent deployments: **30 minutes**

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/strava_callback` | GET | OAuth callback from Strava |
| `/csv-upload` | POST | Upload CSV activity data |
| `/predict` | POST | Get ML pace prediction |
| `/stats` | GET, POST | User statistics & best efforts |
| `/trends` | GET | Weekly training trends |
| `/leaderboard` | GET | Global leaderboard data |
| `/dashboard` | GET | Dashboard aggregated data |
| `/plan` | GET | Generated training plan |

### Example: Get Prediction

```bash
curl -X POST https://your-api.execute-api.eu-central-1.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -d '{
    "athlete_id": "156086407",
    "distance_km": 10,
    "effort": "race",
    "terrain": "flat",
    "max_hr": 195
  }'
```

### Response
```json
{
  "recommended_pace": "4:32",
  "recommended_pace_decimal": 4.53,
  "estimated_time": "00:45:18",
  "expected_avg_hr": 187,
  "expected_max_hr": 195,
  "model_metrics": {
    "mae": 0.18,
    "rmse": 0.24,
    "r2": 0.847
  }
}
```

---

## ⚙️ Environment Variables

### Required for All Functions
```bash
REGION_NAME=eu-central-1
```

### OAuth Tables
```bash
TOKENS_TABLE_NAME=strava_tokens_pau
ACTIVITIES_TABLE_NAME=strava_activities_pau
PROCESSED_TABLE_NAME=strava_processed_pau
BEST_EFFORTS_TABLE_NAME=strava_best_efforts_pau
PREDICTIONS_TABLE_NAME=strava_predictions_pau
```

### CSV Tables
```bash
ACTIVITIES_TABLE_CSV=strava_activities_csv
PROCESSED_TABLE_CSV=strava_processed_csv
BEST_EFFORTS_TABLE_CSV=strava_best_efforts_csv
PREDICTIONS_TABLE_CSV=strava_predictions_csv
```

### Strava API (Auth & Fetcher)
```bash
STRAVA_CLIENT_ID=your_client_id
STRAVA_CLIENT_SECRET=your_client_secret
REDIRECT_URL=https://your-cloudfront-domain.cloudfront.net
```

---

## 📖 Usage

### Option 1: Strava OAuth Login
1. Visit the dashboard URL
2. Click "Login with Strava"
3. Authorize the application
4. Set prediction parameters
5. View your personalized dashboard

### Option 2: CSV Upload
1. Export data from Strava (Settings → My Account → Download Your Data)
2. Visit the dashboard and click "Upload CSV"
3. Upload the `activities.csv` file
4. Enter your name and prediction parameters
5. View your personalized dashboard

### Features Available
- **Dashboard**: View statistics, predictions, and best efforts
- **Training Plan**: See your auto-generated weekly plan
- **Leaderboard**: Compare with other users
- **Trends**: Analyze your training patterns over time

---

## 🔒 Security Notes

- OAuth tokens are stored securely in DynamoDB
- No passwords are stored - authentication is via Strava OAuth
- API endpoints include CORS protection
- For production: Use restrictive IAM policies and consider API keys

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 👥 Team

**Team 10**

- Project developed as part of an academic/professional project
- ML-powered running analytics using real Strava data

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Strava API](https://developers.strava.com/) for activity data access
- [AWS](https://aws.amazon.com/) for serverless infrastructure
- [scikit-learn](https://scikit-learn.org/) for ML capabilities
- [Chart.js](https://www.chartjs.org/) for data visualization

---

<p align="center">
  Made with 🧡 and lots of ☕
</p>

<p align="center">
  <a href="#-strava-analytics-dashboard">Back to Top ↑</a>
</p>
