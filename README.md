# Customer Segmentation Analysis Project 🎯

## Overview 📊
This project performs comprehensive segmentation analysis on the Northwind database, utilizing advanced clustering techniques to identify customer, product, supplier, and country-based segments. The analysis employs DBSCAN algorithm to detect unusual patterns and significant segments within the data.

## Project Team 👥
- Elif Barutçu
- Elif Erdal
- Didar Arslan
- Elif Duymaz Yılmaz
- Deniz Tunç
- Hatice Nur Eriş
- Elif Özbay

## Project Components 🧩

### 1. Product Segmentation (`sample2.py`) 📦
- Product performance analysis based on sales metrics
- Identification of low-performing products and unusual product combinations
- Sales pattern analysis and trend identification

### 2. Supplier Segmentation (`sample3.py`) 🏭
- Supplier performance evaluation based on product sales
- Detection of underperforming suppliers
- Analysis of supplier contribution patterns

### 3. Country Segmentation (`sample4.py`) 🌍
- Order pattern analysis by country
- Identification of countries with unusual ordering behaviors
- Regional market analysis

### 4. Unified API (`api.py`) 🔌
- Single endpoint for all segmentation analyses
- Swagger UI integration for easy interaction
- Comprehensive documentation

## Technology Stack 💻
- **Backend**: Python, FastAPI
- **Database**: PostgreSQL
- **Data Analysis**: scikit-learn (DBSCAN, KMeans), pandas
- **Visualization**: matplotlib
- **ORM**: SQLAlchemy

## Project Workflow 🔄

### 1. Data Collection 📥
- Extraction of relevant data from PostgreSQL database
- Custom SQL queries for each segmentation type
- Data validation and integrity checks

### 2. Data Preprocessing 🧹
- Missing value handling
- Feature selection and engineering
- Data normalization using StandardScaler
- Outlier detection and treatment

### 3. Clustering Analysis 📊
- Implementation of DBSCAN algorithm
- Optimal parameter selection using KneeLocator
- Cluster quality assessment using Silhouette scores
- Performance optimization

### 4. Visualization 📈
- 2D cluster visualization
- Silhouette score plots
- Cluster distribution analysis
- Interactive visualizations

### 5. API Integration 🔄
- RESTful API development with FastAPI
- Swagger UI documentation
- JSON response formatting
- Error handling and validation

## Installation ⚙️

### Prerequisites 📋
- Python 3.8+
- PostgreSQL
- pip or conda package manager

### Setup Instructions 📝

#### Using pip:
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

#### Using conda:
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate problem_env
```

### Database Configuration 🔧
```python
# Configure in config.py
DB_CONFIG = {
    'user': 'your_username',
    'password': 'your_password',
    'host': 'localhost',
    'port': '5432',
    'database': 'northwind'
}
```

## Usage 🚀

1. Start the API server:
```bash
uvicorn api:app --reload
```

2. Access the API documentation:
- Open `http://localhost:8000/docs` in your browser
- Explore available endpoints and their documentation

## API Endpoints 🌐

| Endpoint | Description | Parameters |
|----------|-------------|------------|
| `/api/product-clustering` | Product segmentation analysis | Optional: min_samples, eps |
| `/api/supplier-clustering` | Supplier segmentation analysis | Optional: min_samples, eps |
| `/api/country-clustering` | Country segmentation analysis | Optional: min_samples, eps |

## Output Format 📄
Each endpoint returns a JSON response containing:
- Cluster assignments
- Outlier detection results
- Statistical summaries
- Visualization data

## Contributing 🤝
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📜
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏
- Northwind database
- scikit-learn development team
- FastAPI community

## Turkish Version
You can find the Turkish version of this README file [here](https://github.com/elfbrtc/Turkcell-GYK1-ClusterVision/tree/main).
