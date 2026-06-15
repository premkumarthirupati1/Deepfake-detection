# Deepfake Detection System

## Overview

The Deepfake Detection System is a web-based application developed using the Django framework that enables users to determine whether a given video is real or artificially generated (deepfake). The system leverages an ensemble machine learning approach trained on a dataset of over 22,900 labeled video metadata records to deliver accurate predictions.

The application supports two types of users — Remote Users who submit videos for prediction, and Service Providers who monitor system activity and manage users through an analytics dashboard.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [License](#license)

---

## Features

- Classification of videos as **FAKE** or **REAL** using an ensemble Voting Classifier
- Two-role user system: **Remote User** (submit predictions) and **Service Provider** (analytics and administration)
- Session-based authentication with user registration and profile management
- Geo-metadata logging — stores country, IP address, latitude, and longitude for every prediction
- Analytics dashboard with charts for prediction ratios and deepfake type distribution
- Downloadable trained datasets for Service Providers

---

## Tech Stack

| Layer            | Technology                                              |
|------------------|---------------------------------------------------------|
| Backend          | Python, Django 3.0                                      |
| Machine Learning | Scikit-learn (Naive Bayes, Linear SVC, Logistic Regression, VotingClassifier) |
| Data Processing  | Pandas, NumPy, CountVectorizer                          |
| Database         | MySQL                                                   |
| Frontend         | HTML, CSS (Django Templates)                            |

---

## Project Structure

```
deepfake_detection/
├── deepfake_detection/
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── Remote_User/
│   ├── models.py
│   ├── views.py
│   ├── forms.py
│   └── apps.py
├── Service_Provider/
│   └── views.py
├── Template/
│   ├── htmls/
│   ├── images/
│   └── media/
├── Datasets.csv
└── manage.py
```

---

## Installation

### Prerequisites

- Python 3.8 or above
- MySQL Server
- pip (Python package manager)

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/premkumarthirupati1/deepfake-detection.git
cd deepfake-detection
```

**2. Install required dependencies**
```bash
pip install django scikit-learn pandas numpy openpyxl mysqlclient
```

**3. Configure the database**

Create a MySQL database named `deepfake_detection` and update the `DATABASES` section in `settings.py` with your credentials:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'deepfake_detection',
        'USER': 'your_mysql_username',
        'PASSWORD': 'your_mysql_password',
        'HOST': '127.0.0.1',
        'PORT': '3306',
    }
}
```

**4. Apply database migrations**
```bash
python manage.py makemigrations
python manage.py migrate
```

**5. Start the development server**
```bash
python manage.py runserver
```

**6. Access the application**

Open your browser and navigate to `http://127.0.0.1:8000`

---

## How It Works

1. The system loads `Datasets.csv` containing labeled video metadata records
2. Labels are mapped numerically: `FAKE → 0`, `REAL → 1`
3. Video names are vectorized using `CountVectorizer` to extract text features
4. The dataset is split into 80% training and 20% testing sets
5. Three individual classifiers are trained — Multinomial Naive Bayes, Linear SVC, and Logistic Regression
6. A `VotingClassifier` combines all three models to produce the final prediction
7. The submitted video name is transformed and passed through the ensemble model
8. The prediction result (`FAKE` or `REAL`) is returned to the user and stored in the database with full geo-metadata

---

## Dataset

| Property  | Details                                                              |
|-----------|----------------------------------------------------------------------|
| Records   | 22,900+                                                              |
| Features  | URL, IP Address, Video Name, Resolution, Country, Locale, Latitude, Longitude |
| Labels    | FAKE / REAL                                                          |
| Format    | CSV (`Datasets.csv`)                                                 |

---

## License

This project is licensed under the MIT License.
