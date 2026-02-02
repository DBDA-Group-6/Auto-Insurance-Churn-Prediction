from flask import Flask, request, render_template, jsonify, make_response
import pymysql
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import io
import os
import traceback
import joblib

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp'

# RDS Connection details – override via environment variables in production
RDS_HOST     = os.getenv('RDS_HOST',     'autodb.cr2mkcsuu4rc.ap-south-1.rds.amazonaws.com')
RDS_USER     = os.getenv('RDS_USER',     'admin')
RDS_PASSWORD = os.getenv('RDS_PASSWORD', 'sqlNeguveng')
RDS_DB       = os.getenv('RDS_DB',       'autodb')
MODEL_PATH   = os.getenv('MODEL_PATH',   'my_model.joblib')

# SQLAlchemy engine
try:
    engine = create_engine(
        f"mysql+pymysql://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:3306/{RDS_DB}",
        pool_pre_ping=True,
        pool_recycle=3600
    )
    print("Database engine created successfully")
except Exception as e:
    print(f"Error creating database engine: {e}")
    engine = None

# ===========================================================================
# PREPROCESSING CONSTANTS (must match final_model_pandas.py exactly)
# ===========================================================================
COLUMNS_TO_DROP = [
    'individual_id', 'address_id', 'cust_orig_date',
    'date_of_birth', 'latitude', 'longitude',
    'acct_suspd_date', 'state'
]

HOME_MARKET_LABEL_MAP = {
    '1000 - 24999': 0,  '25000 - 49999': 1,  '50000 - 74999': 2,
    '75000 - 99999': 3, '100000 - 124999': 4, '125000 - 149999': 5,
    '150000 - 174999': 6, '175000 - 199999': 7, '200000 - 224999': 8,
    '225000 - 249999': 9, '250000 - 274999': 10, '275000 - 299999': 11,
    '300000 - 349999': 12, '350000 - 399999': 13, '400000 - 449999': 14,
    '450000 - 499999': 15, '500000 - 749999': 16, '750000 - 999999': 17,
    '1000000 Plus': 18
}

OUTLIER_COLS = ['curr_ann_amt', 'days_tenure', 'age_in_years',
                'length_of_residence', 'income']

# Fixed categorical values (same as final_model_pandas.py)
CITY_VALUES = [
    'Kaufman', 'Grand Prairie', 'Dallas', 'Arlington', 'Fort Worth',
    'Carrollton', 'Allen', 'Bedford', 'The Colony', 'Mckinney',
    'Irving', 'Mesquite', 'Hurst', 'Garland', 'Sachse', 'Euless',
    'Plano', 'Frisco', 'Grapevine', 'Cedar Hill', 'Keller',
    'Justin', 'Wylie', 'Aledo', 'Waxahachie', 'Seagoville',
    'North Richland Hills', 'Desoto', 'Roanoke', 'Southlake',
    'Lancaster', 'Kemp', 'Mansfield', 'Richardson', 'Rice',
    'Caddo Mills', 'Red Oak', 'Weatherford', 'Flower Mound', 'Denton',
    'Ennis', 'Midlothian', 'Coppell', 'Sanger', 'Aubrey', 'Burleson',
    'Duncanville', 'Crowley', 'Rockwall', 'Rowlett', 'Colleyville',
    'Lewisville', 'Balch Springs', 'Argyle', 'Lake Dallas', 'Haslet',
    'Terrell', 'Forney', 'Haltom City', 'Azle', 'Addison', 'Italy',
    'Springtown', 'Joshua', 'Princeton', 'Anna', 'Little Elm',
    'Crandall', 'Ponder', 'Royse City', 'Valley View', 'Ferris',
    'Scurry', 'Farmersville', 'Prosper', 'Kennedale', 'Lavon',
    'Sunnyvale', 'Celina', 'Pilot Point', 'Blue Ridge', 'Melissa',
    'Hutchins', 'Palmer', 'Wilmer', 'Krum', 'Tioga', 'Nevada',
    'Maypearl', 'Era', 'Milford', 'Mertens', 'Forreston', 'Chatfield',
    'Naval Air Station Jrb'
]

COUNTY_VALUES = [
    'Kaufman', 'Dallas', 'Tarrant', 'Denton', 'Collin',
    'Parker', 'Ellis', 'Navarro', 'Hunt', 'Johnson',
    'Rockwall', 'Cooke', 'Grayson', 'Hill'
]

MARITAL_STATUS_VALUES = ['Married', 'Single']


# ===========================================================================
# PANDAS PREPROCESSING FUNCTIONS
# ===========================================================================
def cap_outliers(df, target_cols):
    """Cap outliers using IQR × 1.5 method"""
    df = df.copy()
    
    for column in target_cols:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        # Cap values at bounds
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    return df


def fill_null_values(df, col_name):
    """Fill nulls: numeric → median, categorical → mode"""
    df = df.copy()
    
    # Check if column is numeric or categorical
    if pd.api.types.is_numeric_dtype(df[col_name]):
        # Fill with median for numeric columns
        median = df[col_name].median()
        df[col_name] = df[col_name].fillna(median)
    else:
        # Fill with mode for categorical columns
        mode = df[col_name].mode()
        if len(mode) > 0:
            df[col_name] = df[col_name].fillna(mode[0])
    
    return df


def cat_encoding_fixed(df, column, valid_values):
    """
    One-hot encode categorical column with FIXED set of values.
    Creates a column for EVERY value in valid_values, ensuring
    consistent columns between training and inference.
    """
    df = df.copy()
    
    for val in valid_values:
        clean_name = f"{column}_{str(val).replace(' ', '_')}"
        df[clean_name] = (df[column] == val).astype(int)
    
    df = df.drop(columns=[column])
    return df


# ===========================================================================
# PREPROCESSING FUNCTION (Pandas – mirrors final_model_pandas.py exactly)
# ===========================================================================
def preprocess(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw CSV DataFrame (pandas) and returns the preprocessed DataFrame
    ready for model.predict(). Steps match final_model_pandas.py exactly.
    """
    # Make a copy to avoid modifying original
    df = df_pandas.copy()
    
    # 1) Drop identifier / date columns (NOT city/county)
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    
    # 2) Drop the Churn column – it must NOT be fed to the model
    df = df.drop(columns=['Churn'], errors='ignore')
    
    # 3) Label-encode home_market_value
    df['home_market_value'] = df['home_market_value'].map(HOME_MARKET_LABEL_MAP)
    
    # 4) Cap outliers (IQR × 1.5) on the 5 numeric columns
    df = cap_outliers(df, OUTLIER_COLS)
    
    # 5) Fill remaining nulls (median for numeric, mode for categorical)
    for col_name in df.columns:
        df = fill_null_values(df, col_name)
    
    # 6) One-hot encode city, county, marital_status with FIXED column sets
    df = cat_encoding_fixed(df, 'city', CITY_VALUES)
    df = cat_encoding_fixed(df, 'county', COUNTY_VALUES)
    df = cat_encoding_fixed(df, 'marital_status', MARITAL_STATUS_VALUES)
    
    return df


# ===========================================================================
# LOAD MODEL ONCE AT STARTUP
# ===========================================================================
print("Loading trained model artifact …")
try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    scaler = artifact["scaler"]
    expected_columns = artifact["columns"]
    print(f"✅ Model loaded: {len(expected_columns)} features expected")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None
    expected_columns = []


# ===========================================================================
# DATABASE SETUP
# ===========================================================================
def setup_database():
    """Create prediction table if it doesn't exist"""
    if engine is None:
        print("❌ Database engine not available.")
        return False
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS prediction (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    individual_id VARCHAR(255),
                    address_id VARCHAR(255),
                    curr_ann_amt FLOAT,
                    age_in_years INT,
                    income FLOAT,
                    city VARCHAR(255),
                    county VARCHAR(255),
                    marital_status VARCHAR(50),
                    Churn_Prediction INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
        print("✅ Database table 'prediction' ready")
        return True
    except SQLAlchemyError as e:
        print(f"❌ Database setup error: {e}")
        return False


# ===========================================================================
# ROUTES
# ===========================================================================
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Auto Insurance Churn Prediction</title>
        <style>
            * {margin: 0; padding: 0; box-sizing: border-box;}
            body {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px;}
            .container {background: white; border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                        padding: 50px; max-width: 600px; text-align: center; animation: slideIn 0.5s ease;}
            @keyframes slideIn {from {opacity: 0; transform: translateY(-30px);} to {opacity: 1; transform: translateY(0);}}
            h1 {color: #667eea; margin-bottom: 15px; font-size: 32px; font-weight: 700;}
            p {color: #666; margin-bottom: 30px; line-height: 1.6;}
            .features {text-align: left; margin: 30px 0; padding: 25px; background: #f8f9fa; border-radius: 15px;}
            .features h3 {color: #444; margin-bottom: 15px; font-size: 18px;}
            .features ul {list-style: none;}
            .features li {padding: 8px 0; color: #555; padding-left: 25px; position: relative;}
            .features li:before {content: "✓"; position: absolute; left: 0; color: #4CAF50; font-weight: bold;}
            a {display: inline-block; margin: 10px; padding: 15px 35px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               color: white; text-decoration: none; border-radius: 30px; font-weight: 600; transition: all 0.3s;
               box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);}
            a:hover {transform: translateY(-3px); box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 Auto Insurance Churn Prediction</h1>
            <p>Upload your customer data and get instant churn predictions powered by XGBoost ML model</p>
            
            <div class="features">
                <h3>🎯 What This System Does:</h3>
                <ul>
                    <li>Processes raw CSV data with pandas</li>
                    <li>Applies advanced feature engineering</li>
                    <li>Predicts customer churn probability</li>
                    <li>Stores results in MySQL database</li>
                    <li>Provides downloadable predictions</li>
                </ul>
            </div>
            
            <a href="/ml-model">📊 Start Prediction</a>
            <a href="/view_predictions">👁️ View Results</a>
        </div>
    </body>
    </html>
    """


@app.route('/ml-model')
def ml_model():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload Data for Prediction</title>
        <style>
            body {font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px;}
            .container {background: white; border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                        padding: 40px; max-width: 600px; text-align: center;}
            h2 {color: #667eea; margin-bottom: 20px;}
            p {color: #666; margin-bottom: 25px;}
            .upload-area {border: 3px dashed #667eea; border-radius: 15px; padding: 40px; margin: 20px 0;
                          background: #f8f9ff; cursor: pointer; transition: all 0.3s;}
            .upload-area:hover {background: #e7ebff; border-color: #764ba2;}
            input[type="file"] {display: none;}
            label {cursor: pointer; color: #667eea; font-weight: 600; font-size: 18px;}
            button {background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white; border: none;
                    padding: 15px 40px; font-size: 16px; border-radius: 30px; cursor: pointer; margin-top: 20px;
                    font-weight: 600; transition: all 0.3s; box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);}
            button:hover {transform: translateY(-3px); box-shadow: 0 6px 20px rgba(76, 175, 80, 0.6);}
            .file-name {margin-top: 15px; color: #4CAF50; font-weight: 600;}
            a {display: inline-block; margin-top: 20px; color: #667eea; text-decoration: none;}
        </style>
        <script>
            function showFileName() {
                const input = document.getElementById('csvfile');
                const fileNameDisplay = document.getElementById('fileName');
                if (input.files.length > 0) {
                    fileNameDisplay.textContent = '📄 Selected: ' + input.files[0].name;
                }
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h2>📤 Upload CSV for Churn Prediction</h2>
            <p>Upload your customer data file (CSV format, max 500 MB)</p>
            
            <form action="/predict" method="post" enctype="multipart/form-data">
                <div class="upload-area" onclick="document.getElementById('csvfile').click();">
                    <label for="csvfile">🗂️ Click to Browse or Drag & Drop</label>
                    <input type="file" id="csvfile" name="csvfile" accept=".csv" required onchange="showFileName()">
                    <div id="fileName" class="file-name"></div>
                </div>
                <button type="submit">🚀 Upload & Predict</button>
            </form>
            
            <a href="/">← Back to Home</a>
        </div>
    </body>
    </html>
    """


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if engine is None:
            return error_page("Database not available."), 500
        
        if model is None:
            return error_page("Model not loaded."), 500

        # 1) Read uploaded CSV
        file = request.files.get('csvfile')
        if not file or file.filename == '':
            return error_page("No file uploaded."), 400

        df_raw = pd.read_csv(file)
        print(f"Received file: {file.filename}, shape: {df_raw.shape}")

        # 2) Preprocess with pandas (mirrors final_model_pandas.py)
        df_processed = preprocess(df_raw)
        print(f"After preprocessing: {df_processed.shape}")

        # 3) Ensure column alignment with training
        for col in expected_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[expected_columns]

        # 4) Scale features
        X_scaled = scaler.transform(df_processed)

        # 5) Predict
        predictions = model.predict(X_scaled)
        print(f"Generated {len(predictions)} predictions")

        # 6) Build result DataFrame (original data + prediction)
        df_result = df_raw.copy()
        df_result['Churn_Prediction'] = predictions

        # 7) Store in database (only key columns + prediction)
        df_to_db = df_result[[
            'individual_id', 'address_id', 'curr_ann_amt', 'age_in_years',
            'income', 'city', 'county', 'marital_status', 'Churn_Prediction'
        ]].copy()

        df_to_db.to_sql('prediction', engine, if_exists='replace', index=False)
        print(f"Stored {len(df_to_db)} predictions in database")

        # 8) Count predictions
        unique_preds = df_result['Churn_Prediction'].value_counts().to_dict()

        # 9) Return success page
        return success_page(file.filename, df_raw, df_result, unique_preds)

    except Exception as e:
        traceback.print_exc()
        return error_page(f"Prediction error: {str(e)}"), 500


@app.route('/view_predictions')
def view_predictions():
    try:
        if engine is None:
            return error_page("Database not available."), 500

        df_predictions = pd.read_sql("SELECT * FROM prediction", engine)
        
        if df_predictions.empty:
            return error_page("No predictions found in database."), 404

        total_count = len(df_predictions)
        
        # Show first 100 rows
        df_display = df_predictions.head(100)
        table_html = df_display.to_html(classes='prediction-table', index=False)

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
        <title>View Predictions</title>
        <style>
            body {{font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  min-height: 100vh; padding: 20px;}}
            .container {{background: white; border-radius: 20px; padding: 40px; max-width: 1200px; margin: 0 auto;}}
            h2 {{color: #2196F3; text-align: center;}}
            .prediction-table {{width: 100%; border-collapse: collapse; margin: 20px 0;}}
            .prediction-table th {{background: #2196F3; color: white; padding: 12px; text-align: left;}}
            .prediction-table td {{padding: 10px; border-bottom: 1px solid #ddd;}}
            .prediction-table tr:hover {{background: #f5f5f5;}}
            .buttons {{text-align: center; margin-top: 30px;}}
            a {{display: inline-block; margin: 5px; padding: 12px 25px; background: #4CAF50; color: white; text-decoration: none; border-radius: 10px;}}
            a.secondary {{background: #2196F3;}}
            a.download {{background: #FF9800;}}
        </style>
        </head>
        <body>
            <div class="container">
                <h2>📋 Prediction Results</h2>
                <p style="text-align: center; color: #666;">Showing {len(df_display)} of {total_count} predictions</p>
                <div style="overflow-x: auto;">{table_html}</div>
                <div class="buttons">
                    <a href="/">Home</a>
                    <a href="/ml-model">Upload New Data</a>
                    <a href="/download_predictions" class="download">📥 Download All ({total_count} rows)</a>
                </div>
            </div>
        </body>
        </html>
        """
    except Exception as e:
        traceback.print_exc()
        return error_page(f"Error: {str(e)}"), 500


@app.route('/download_predictions')
def download_predictions():
    try:
        if engine is None:
            return error_page("Database not available."), 500

        df_predictions = pd.read_sql("SELECT * FROM prediction", engine)
        if df_predictions.empty:
            return error_page("No predictions found."), 404

        csv_data = df_predictions.to_csv(index=False)
        response = make_response(csv_data)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = 'attachment; filename=predictions.csv'
        return response
    except Exception as e:
        traceback.print_exc()
        return error_page(f"Download error: {str(e)}"), 500


# ---------------------------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------------------------
@app.route('/health')
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


# ===========================================================================
# PAGE HELPERS
# ===========================================================================
def error_page(message):
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Error</title>
        <style>
            body {{font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px;}}
            .container {{background: white; border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); padding: 40px; max-width: 500px; text-align: center;}}
            h2 {{color: #e53935; margin-bottom: 20px;}}
            p {{color: #666; margin-bottom: 30px;}}
            a {{display: inline-block; padding: 12px 30px; background: #4CAF50; color: white; text-decoration: none; border-radius: 10px; font-weight: 600;}}
            a:hover {{background: #45a049; transform: translateY(-2px);}}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>⚠️ Error</h2>
            <p>{message}</p>
            <a href="/ml-model">Try Again</a>
        </div>
    </body>
    </html>
    """


def success_page(filename, df_raw, df_result, unique_preds):
    """Rendered after a successful upload → preprocess → predict cycle."""
    total = len(df_result)

    # Build prediction-distribution rows
    stats_html = "".join([
        f'<div class="stat-item">'
        f'<span>{"Churned" if k == 1 else "Not Churned"} ({k})</span>'
        f'<span>{v} ({v / total * 100:.1f}%)</span>'
        f'</div>'
        for k, v in sorted(unique_preds.items())
    ])

    # Show first 10 rows of the combined result as a preview table
    preview_html = df_result.head(10).to_html(classes='preview-table', index=False)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Success</title>
        <style>
            body {{font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px;}}
            .container {{background: white; border-radius: 20px; padding: 40px; max-width: 900px; width: 100%;}}
            h2 {{color: #4CAF50; text-align: center; margin-bottom: 10px;}}
            .subtitle {{text-align: center; color: #666; margin-bottom: 25px; font-size: 14px;}}
            .info {{background: #f5f5f5; padding: 20px; border-radius: 10px; margin-bottom: 20px;}}
            .stats {{background: #e7f3ff; padding: 20px; border-radius: 10px; margin: 20px 0;}}
            .stat-item {{display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #ddd;}}
            .stat-item:last-child {{border-bottom: none;}}
            .preview-table {{width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 13px; overflow-x: auto; display: block;}}
            .preview-table th {{background: #667eea; color: white; padding: 10px; text-align: left; white-space: nowrap;}}
            .preview-table td {{padding: 8px 10px; border-bottom: 1px solid #eee; white-space: nowrap;}}
            .preview-table tr:hover {{background: #f9f9ff;}}
            .buttons {{text-align: center; margin-top: 30px;}}
            a {{display: inline-block; margin: 5px; padding: 12px 25px; background: #4CAF50; color: white; text-decoration: none; border-radius: 10px; font-weight: 600;}}
            a.secondary {{background: #2196F3;}}
            a.download  {{background: #FF9800;}}
            .section-title {{color: #555; font-weight: 600; margin: 20px 0 8px; font-size: 15px;}}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>✅ Upload & Prediction Successful!</h2>
            <p class="subtitle">Pipeline: Raw data → Pandas Preprocessing → XGBoost Model → Predictions</p>

            <div class="info">
                <p><strong>📄 {filename}</strong><br>
                   {df_raw.shape[0]} rows × {df_raw.shape[1]} cols (raw) → {total} predictions generated</p>
            </div>

            <p class="section-title">📊 Prediction Distribution</p>
            <div class="stats">{stats_html}</div>

            <p class="section-title">📋 Preview (first 10 rows with Churn_Prediction)</p>
            <div style="overflow-x: auto;">{preview_html}</div>

            <div class="buttons">
                <a href="/ml-model">Upload New Data</a>
                <a href="/view_predictions" class="secondary">View All Predictions</a>
                <a href="/download_predictions" class="download">📥 Download CSV</a>
            </div>
        </div>
    </body>
    </html>
    """


# ===========================================================================
# ERROR HANDLERS
# ===========================================================================
@app.errorhandler(413)
def request_entity_too_large(error):
    return error_page("File too large. Maximum allowed: 500 MB."), 413

@app.errorhandler(400)
def bad_request(error):
    return error_page("Bad request."), 400


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == '__main__':
    print("=" * 50)
    print("Auto Insurance Churn Prediction System Starting…")
    print("=" * 50)
    if setup_database():
        print("Database ready!  Server starting at http://0.0.0.0:5000")
        print("=" * 50)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Database setup failed.")
        print("=" * 50)
