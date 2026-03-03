📊 **ForecastPro Enterprise**

![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)  
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)  
![MIT License](https://img.shields.io/badge/license-MIT-green)  
![Version 5.0](https://img.shields.io/badge/version-5.0-orange)  
![Database Firebase](https://img.shields.io/badge/database-Firebase-FFCA28)  
![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

Advanced Time Series Forecasting Application with AutoML, Anomaly Detection, Interactive Visualizations, and Firebase Cloud Database. Built with Streamlit and scikit-learn.

🚀 **Live Demo**  
👉 https://forecastpro-app.streamlit.app

---

✨ **Features**
- 🔐 **User Authentication with Firebase**
  - Secure login/signup system with password hashing
  - User profiles with personalized forecast history
  - All user data stored permanently in Firebase cloud
  - Never lose data on redeploy

- 📈 **Multiple Forecasting Models**
  - Linear Regression - Simple and interpretable
  - Polynomial Regression - Captures non-linear patterns
  - Random Forest - Ensemble learning for robust predictions
  - Gradient Boosting - Advanced boosting algorithm
  - Model Comparison - Compare all models side-by-side

- 🤖 **AutoML**
  - Automatically tests 7 different model configurations
  - Finds the best performing model for your data
  - Displays R², MAE, and RMSE metrics

- 🔍 **Anomaly Detection**
  - Isolation Forest - Machine learning based detection
  - Z-Score - Statistical method using standard deviations
  - IQR - Interquartile range based detection

- 📊 **Data Visualization**
  - Interactive Charts: Line, Bar, Scatter, Histogram, Box plots
  - Legends on every graph at top left for easy understanding
  - Correlation Heatmaps
  - Export charts as CSV/Excel/JSON

- 🧹 **Auto Data Cleaning**
  - Removes duplicate rows
  - Handles missing values (fills with median)
  - Removes outliers (3 standard deviations)
  - Converts date columns to datetime
  - Strips whitespace from string columns

- 🎨 **Theme Support**
  - 🌙 Dark Mode toggle
  - ☀️ Light Mode
  - Theme-aware charts and UI

- 💾 **Forecast History with Firebase**
  - Save all forecasts with timestamps
  - View mini charts of past forecasts
  - Delete individual forecasts
  - Clear all history
  - Download forecasts in multiple formats
  - All data permanently stored in Firebase

- 📥 **Export Options**
  - CSV
  - Excel
  - JSON

---

🛠️ **Tech Stack**

| Technology     | Version | Purpose                          |
|----------------|---------|----------------------------------|
| Python         | 3.12    | Core programming language        |
| Streamlit      | 1.40.2  | Web application framework        |
| Firebase       | -       | Cloud database (Firestore)       |
| Pandas         | 2.2.3   | Data manipulation                |
| NumPy          | 1.26.4  | Numerical computing              |
| Scikit-learn   | 1.5.2   | Machine learning models          |
| Statsmodels    | 0.14.4  | Statistical modeling             |
| Plotly         | 5.24.1  | Interactive visualizations       |
| OpenPyXL       | 3.1.5   | Excel file handling              |
| Chardet        | 5.2.0   | Encoding detection               |


📁 **Project Structure**

```
forecastpro-app/
│
├── app.py                      # Main application code
├── model.py                    # Forecasting models
├── requirements.txt            # Python dependencies
├── packages.txt                # Linux system dependencies
├── .python-version             # Python version (3.12)
├── .gitignore                  # Git ignore file
├── README.md                   # This file
├── LICENSE                     # MIT License
│
├── .streamlit/
│   └── config.toml             # Streamlit configuration
│
└── data/
    └── Sample - Superstore.csv  # Sample dataset
```

---

🚀 **Quick Start**

**Prerequisites**
- Python 3.12 or higher
- pip package manager

**One-Line Setup**
```bash
# Clone repository
git clone https://github.com/srirammulukuntla11/forecastpro-app.git
cd forecastpro-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```
Your app will open at http://localhost:8501

🔥 **Firebase Setup (Optional - for cloud deployment)**
1. Create Firebase Project at console.firebase.google.com
2. Enable Firestore (Native mode)
3. Generate service account key and download as `firebase-key.json` (do not commit)
4. Add same credentials to Streamlit secrets in TOML format under `[firebase]`

☁️ **Deploy to Streamlit Cloud**
1. Push code to GitHub
2. Create new app on share.streamlit.io pointing to your repo
3. Add Firebase secrets in app settings
4. Deploy and enjoy!

---

🎯 **How to Use**
1. Login/Signup
2. Upload Data (sample or your own)
3. Explore tabs: Data Explorer, Visual Analytics, Generate Forecast, Model Comparison, AutoML, Anomaly Detection, Forecast History
4. Export results as needed

---

🔧 **Configuration Files**

```txt
# requirements.txt excerpt
streamlit==1.40.2
pandas==2.2.3
...
```

(see full README for details)

---

🐛 **Troubleshooting**
| Issue                            | Solution                                      |
|----------------------------------|-----------------------------------------------|
| Firebase not installed           | Run `pip install firebase-admin`             |
| Python version error             | Add `.python-version` file with 3.12         |
| Firebase API not enabled         | Enable Firestore API in Google Cloud Console |
| Database not created             | Create Firestore database in Native mode     |
| Secrets missing                  | Add Firebase secrets to Streamlit Cloud     |
| Box plot error                   | Fixed in code (uses proper rgba format)      |
| Local Firebase error             | App falls back to JSON automatically        |

---

🤝 **Contributing**
1. Fork repository
2. Create feature branch
3. Commit and push changes
4. Open a PR

---

📝 **License**
This project is licensed under the MIT License - see the LICENSE file for details.

📧 **Contact**
Sriram Mulukuntla - GitHub: @srirammulukuntla11

---

🙏 **Acknowledgments**
Streamlit • Firebase • Scikit-learn • Plotly • Pandas

📊 **Feature Summary**
| Category       | Features                                    |
|----------------|---------------------------------------------|
| Database       | Firebase Firestore, JSON fallback           |
| Authentication | Login, Signup, Password hashing             |
| Data Handling  | Upload, Auto-clean, Export (CSV/Excel/JSON) |
| Visualization  | 5 chart types, Legends, Heatmaps            |
| Forecasting    | 4 Models, Comparison, AutoML                |
| Analysis       | Anomaly Detection (3 methods)               |
| UX             | Dark/Light theme, Responsive                |
| Deployment     | Streamlit Cloud ready                       |

⭐ **Support**
If you find this project useful, please star it on GitHub!

https://img.shields.io/badge/Made%2520with-Streamlit-ff4b4b.svg  
https://img.shields.io/badge/Python-3.12-3776AB.svg  
https://img.shields.io/badge/Firebase-FFCA28.svg  
https://img.shields.io/badge/Maintained%253F-yes-green.svg  
https://img.shields.io/badge/license-MIT-blue.svg

© 2026 Sriram Mulukuntla. All rights reserved.

