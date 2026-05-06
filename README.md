# 🏠 House Price Prediction System — Nepal

A machine learning project that predicts residential house sale prices in Nepal using real-world property listing data scraped from [Basobaas](https://basobaas.com). The model is built with **Linear Regression** and covers the full ML pipeline — from raw data cleaning to model evaluation.

---

## 📌 Project Overview

The Nepali real estate market is rapidly growing, yet price estimation remains largely informal and opaque. This project aims to bring data-driven insights into house pricing across major cities of Nepal (Kathmandu, Lalitpur, Bhaktapur, and beyond) by training a regression model on real property listings.

**Goal:** Given property features (area, location, road access, bedrooms, etc.), predict the sale price of a house.

---

## 📁 Project Structure

```
house-prediction/
│
├── HousePrice.ipynb          # Main Jupyter Notebook (full ML pipeline)
├── HouseDetails.csv          # Raw dataset (~14,800+ property listings)
└── README.md                 # Project documentation
```

---

## 📊 Dataset

- **Source:** [Basobaas.com](https://basobaas.com) — a Nepal-based real estate platform
- **File:** `HouseDetails.csv`
- **Size:** ~14,800+ property listings
- **Coverage:** Residential properties across Nepal (Kathmandu Valley and beyond)

### Key Raw Features

| Feature | Description |
|---|---|
| `price` | Listing price (target variable) |
| `status` | Sale or Rent |
| `category_name` | Property type (House, Flat, Land, etc.) |
| `bedroom_count` | Number of bedrooms |
| `bathroom_count` | Number of bathrooms |
| `kitchen_count` | Number of kitchens |
| `livingroom_count` | Number of living rooms |
| `parking_space_count` | Number of parking spaces |
| `floors` | Number of floors |
| `area_covered` | Land area (in various Nepali/SI units) |
| `access_road` | Road access width |
| `road_type` | Type of road (Blacktopped, Gravelled, etc.) |
| `property_face` | Cardinal direction the property faces |
| `location` | City/area (JSON-encoded) |
| `gpslat` / `gpslng` | GPS coordinates |

---

## 🔧 Methodology

### 1. Data Cleaning & Feature Selection

Irrelevant and metadata columns are dropped, including:
- Administrative fields: `id`, `slug`, `tags`, `featured`, `verified`, `premium`
- User/owner info: `user_id`, `owner_id`, `developer_id`
- Text fields: `title`, `property_name`, `description`, `video`
- High-null columns: `furnishing`, `built_year`, `builtup_area`

**Filtering:** Only **residential houses for sale** are retained.

### 2. Feature Engineering

| Task | Detail |
|---|---|
| **Location Extraction** | City name extracted from JSON-encoded `location` field using regex |
| **Area Normalization** | All land area units converted to **sq. meters** — supports Aana, Ropani, Paisa, Dam, Kattha, Dhur, Bigha, Sq. Feet, Sq. Meter, Haat |
| **Road Width Normalization** | Road widths converted to **meters** (Feet → Meters conversion) |
| **Categorical Encoding** | `property_face`, `road_type`, `type`, and `location` encoded using `OrdinalEncoder` |

### 3. Missing Value Imputation

Missing values (especially road width) are handled using **Iterative Imputer** (`sklearn.impute.IterativeImputer`) with `random_state=42` for reproducibility — a multivariate imputation strategy that models each feature as a function of others.

### 4. Model Training

```
Algorithm : Linear Regression (sklearn.linear_model.LinearRegression)
Train/Test : 80% / 20% split (random_state=5)
Target     : price (house sale price in NPR)
```

**Features used for prediction:**
- `bedroom_count`, `bathroom_count`, `kitchen_count`, `livingroom_count`, `parking_space_count`
- `floors`
- `area` (normalized to sq. meters)
- `Road` (road width in meters)
- `property_face_code`, `road_type`, `type_code`, `location_code`

### 5. Evaluation

Model performance is reported using the **Variance Score (R²)**:
```python
reg.score(X_test, y_test)
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas scikit-learn regex numpy
```

If running on **Google Colab** (original environment):
- Upload `HouseDetails.csv` to your Google Drive under `My Drive/ML Project/`
- Mount Drive and run all cells in order

### Running Locally

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd house-prediction
   ```

2. Update the dataset path in the notebook:
   ```python
   # Change this line:
   df = pd.read_csv('/content/drive/MyDrive/ML Project/HouseDetails.csv')
   # To:
   df = pd.read_csv('HouseDetails.csv')
   ```

3. Launch Jupyter:
   ```bash
   jupyter notebook HousePrice.ipynb
   ```

4. Run all cells sequentially.

---

## 🌏 Nepali Land Measurement Units

A key challenge in this project is Nepal's use of traditional land measurement units. The following conversions are implemented:

| Unit | Equivalent in sq. meters |
|---|---|
| 1 Ropani | 508.737 sq. m |
| 1 Aana | 31.796 sq. m |
| 1 Paisa | 7.949 sq. m |
| 1 Dam | 1.987 sq. m |
| 1 Bigha | 6,772.63 sq. m |
| 1 Kattha | 338.63 sq. m |
| 1 Dhur | 16.93 sq. m |
| 1 Sq. Feet | 0.0929 sq. m |
| 1 Haat | 0.209 sq. m |

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `scikit-learn` | Preprocessing, imputation, modelling |
| `regex` | Parsing mixed-format strings (area, location) |
| `numpy` | Numerical operations |
| Google Colab | Development environment |

---

## 📈 Future Improvements

- [ ] Try advanced regressors: **Random Forest**, **XGBoost**, **Gradient Boosting**
- [ ] Add cross-validation for more robust evaluation
- [ ] Include price-per-unit-area as a derived feature
- [ ] Filter outliers (unusually high/low prices)
- [ ] Build a web interface for interactive price prediction
- [ ] Expand dataset to include rental price prediction

---

## 👤 Author

**Bishal Maharjan**  
Machine Learning Project — House Price Prediction System, Nepal

---

## 📄 License

This project is for educational purposes. Dataset sourced from publicly available listings on [Basobaas.com](https://basobaas.com).
