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
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max file size
app.config['UPLOAD_FOLDER'] = '/tmp'

# RDS Connection details – override via environment variables in production
RDS_HOST     = os.getenv('RDS_HOST',     'autodb.cr2mkcsuu4rc.ap-south-1.rds.amazonaws.com')
RDS_USER     = os.getenv('RDS_USER',     'admin')
RDS_PASSWORD = os.getenv('RDS_PASSWORD', '123456789')
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
# PREPROCESSING CONSTANTS  →  KEPT EXACTLY THE SAME
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

CITY_VALUES = [ ... ]   # ← keeping original long list (omitted here for brevity)

COUNTY_VALUES = [ ... ] # ← keeping original

MARITAL_STATUS_VALUES = ['Married', 'Single']

# ===========================================================================
# PREPROCESSING FUNCTIONS  →  KEPT EXACTLY THE SAME
# ===========================================================================
def cap_outliers(df, target_cols): ...
def fill_null_values(df, col_name): ...
def cat_encoding_fixed(df, column, valid_values): ...
def preprocess(df_pandas: pd.DataFrame) -> pd.DataFrame: ...
    # ↑↑↑  ALL LOGIC REMAINS UNCHANGED  ↑↑↑

# ===========================================================================
# DATABASE SETUP  →  unchanged
# ===========================================================================
def setup_database(): ...

# ===========================================================================
# ROUTES
# ===========================================================================
@app.route('/')
def index():
    return render_template('home.html')   # ← assume you keep your original home

@app.route('/ml-model')
def ml_model():
    return render_template('index.html')  # ← your upload form (single file)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# ---------------------------------------------------------------------------
# UPLOAD → PREPROCESS → PREDICT  (single file)
# ---------------------------------------------------------------------------
@app.route('/upload', methods=['POST'])
def upload():
    try:
        if engine is None:
            return error_page("Database connection not available."), 500

        if not request.files or 'csv1' not in request.files:
            return error_page("Please select a raw CSV file."), 400

        csv1 = request.files['csv1']
        if not csv1 or csv1.filename == '' or not csv1.filename.lower().endswith('.csv'):
            return error_page("Please upload a valid .csv file."), 400

        csv1_content = csv1.read()
        if len(csv1_content) == 0:
            return error_page("Uploaded file is empty."), 400

        df_raw = pd.read_csv(io.BytesIO(csv1_content))
        if df_raw.empty:
            return error_page("CSV contains no data rows."), 400

        # Save raw
        df_raw.to_sql(name='rawdata', con=engine, if_exists='replace', index=False)

        # Preprocess (THIS PART REMAINS UNTOUCHED)
        df_preprocessed = preprocess(df_raw)

        df_preprocessed.to_sql(name='processed', con=engine, if_exists='replace', index=False)

        # Load model + predict (THIS BLOCK REMAINS UNTOUCHED)
        if not os.path.exists(MODEL_PATH):
            return error_page(f"Model file not found: {MODEL_PATH}"), 404

        artifact = joblib.load(MODEL_PATH)
        model      = artifact["model"]
        scaler     = artifact["scaler"]
        train_cols = artifact["columns"]

        df_aligned = df_preprocessed.reindex(columns=train_cols, fill_value=0)
        df_scaled  = scaler.transform(df_aligned)
        predictions = model.predict(df_scaled)

        # Save result
        df_result = df_raw.copy()
        df_result['Churn_Prediction'] = predictions
        df_result.to_sql(name='prediction', con=engine, if_exists='replace', index=False)

        unique_preds = pd.Series(predictions).value_counts().to_dict()

        return success_page(csv1.filename, df_raw, df_result, unique_preds)

    except Exception as e:
        traceback.print_exc()
        return error_page(f"Unexpected error: {str(e)}"), 500


# ---------------------------------------------------------------------------
# PREDICTION SUCCESS PAGE  (styled like app.py)
# ---------------------------------------------------------------------------
def success_page(filename, df_raw, df_result, unique_preds):
    total = len(df_result)

    # Format prediction stats like app.py
    stats_html = ""
    for k, v in sorted(unique_preds.items()):
        label = "Churned" if k == 1 else "Not Churned"
        percentage = v / total * 100 if total > 0 else 0
        stats_html += f"""
        <div class="summary-item">
            <span class="summary-label">{label} ({k})</span>
            <span class="summary-value">{v} ({percentage:.1f}%)</span>
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload Success</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }}
            .container {{
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
                max-width: 700px;
                width: 100%;
            }}
            .success-icon {{
                text-align: center;
                font-size: 60px;
                margin-bottom: 20px;
            }}
            h2 {{ 
                color: #4CAF50;
                text-align: center;
                margin-bottom: 10px;
            }}
            .subtitle {{
                text-align: center;
                color: #666;
                margin-bottom: 25px;
                font-size: 15px;
            }}
            .info {{
                background: #f5f5f5;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 25px;
            }}
            .file-info {{
                margin-bottom: 15px;
                padding-bottom: 15px;
                border-bottom: 1px solid #ddd;
            }}
            .file-info:last-child {{
                border-bottom: none;
            }}
            .file-name {{
                font-weight: 600;
                color: #333;
                margin-bottom: 6px;
            }}
            .file-stats {{
                color: #666;
                font-size: 14px;
            }}
            .summary {{
                background: #e7f3ff;
                padding: 20px;
                border-radius: 10px;
                margin: 25px 0;
            }}
            .summary h3 {{
                margin: 0 0 15px 0;
                color: #2196F3;
                font-size: 18px;
            }}
            .summary-item {{
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid rgba(33,150,243,0.15);
            }}
            .summary-item:last-child {{
                border-bottom: none;
            }}
            .summary-label {{
                font-weight: 600;
                color: #333;
            }}
            .summary-value {{
                color: #2196F3;
                font-weight: 600;
            }}
            .buttons {{
                text-align: center;
                margin-top: 30px;
            }}
            a {{ 
                display: inline-block;
                margin: 6px;
                padding: 12px 24px;
                background: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 10px;
                font-weight: 600;
                transition: all 0.3s ease;
            }}
            a:hover {{ 
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(76,175,80,0.4);
            }}
            a.secondary {{
                background: #2196F3;
            }}
            a.download {{
                background: #FF9800;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="success-icon">✅</div>
            <h2>Upload & Prediction Successful!</h2>
            <p class="subtitle">Processed {total:,} records from {filename}</p>

            <div class="info">
                <div class="file-info">
                    <div class="file-name">📄 Uploaded file</div>
                    <div class="file-stats">
                        {filename}<br>
                        {len(df_raw):,} rows × {len(df_raw.columns)} columns (raw)
                    </div>
                </div>
                <div class="file-info">
                    <div class="file-name">🎯 Predictions generated</div>
                    <div class="file-stats">
                        Saved to <strong>prediction</strong> table in database
                    </div>
                </div>
            </div>

            <div class="summary">
                <h3>Prediction Distribution</h3>
                {stats_html}
            </div>

            <div class="buttons">
                <a href="/ml-model">Upload Another File</a>
                <a href="/view_predictions" class="secondary">View Predictions</a>
                <a href="/download_predictions" class="download">📥 Download CSV</a>
            </div>
        </div>
    </body>
    </html>
    """


# ---------------------------------------------------------------------------
# VIEW PREDICTIONS  (styled like app.py)
# ---------------------------------------------------------------------------
@app.route('/view_predictions')
def view_predictions():
    try:
        if engine is None:
            return error_page("Database not available."), 500

        df_predictions = pd.read_sql("SELECT * FROM prediction LIMIT 50", engine)
        if df_predictions.empty:
            return error_page("No predictions found. Upload a CSV first."), 404

        total_count = pd.read_sql("SELECT COUNT(*) as total FROM prediction", engine)['total'][0]
        table_html  = df_predictions.to_html(classes='prediction-table', index=False)

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Results</title>
            <style>
                body {{font-family:Arial,sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;padding:20px;}}
                .container {{background:white;border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,0.3);padding:40px;max-width:1200px;margin:0 auto;}}
                h2 {{color:#2196F3;text-align:center;}}
                .subtitle {{text-align:center;color:#666;margin-bottom:30px;}}
                .prediction-table {{width:100%;border-collapse:collapse;margin:20px 0;}}
                .prediction-table th {{background:#2196F3;color:white;padding:12px;text-align:left;}}
                .prediction-table td {{padding:10px;border-bottom:1px solid #ddd;}}
                .prediction-table tr:hover {{background:#f5f5f5;}}
                .buttons {{text-align:center;margin-top:30px;}}
                a {{display:inline-block;margin:5px;padding:12px 25px;background:#4CAF50;color:white;text-decoration:none;border-radius:10px;font-weight:600;}}
                a.secondary {{background:#2196F3;}}
                a.download {{background:#FF9800;}}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>📋 Prediction Results</h2>
                <p class="subtitle">Showing {len(df_predictions)} of {total_count} rows</p>
                <div style="overflow-x:auto;">{table_html}</div>
                <div class="buttons">
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


# ---------------------------------------------------------------------------
# DOWNLOAD PREDICTIONS
# ---------------------------------------------------------------------------
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
        response.headers['Content-Disposition'] = 'attachment; filename=churn_predictions.csv'
        return response
    except Exception as e:
        traceback.print_exc()
        return error_page(f"Download error: {str(e)}"), 500


# ---------------------------------------------------------------------------
# ERROR & HEALTH  (unchanged logic, error page style updated slightly)
# ---------------------------------------------------------------------------
def error_page(message):
    return f"""
    <!DOCTYPE html>
    <html>
    <head><title>Error</title>
    <style>
        body {{font-family:Arial,sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;display:flex;justify-content:center;align-items:center;padding:20px;}}
        .container {{background:white;border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,0.3);padding:40px;max-width:500px;text-align:center;}}
        h2 {{color:#e53935;margin-bottom:20px;}}
        p {{color:#666;margin-bottom:30px;}}
        a {{display:inline-block;padding:12px 30px;background:#4CAF50;color:white;text-decoration:none;border-radius:10px;font-weight:600;}}
        a:hover {{background:#45a049;transform:translateY(-2px);}}
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


@app.route('/health')
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return error_page("File too large. Maximum allowed: 1GB."), 413


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == '__main__':
    print("=" * 50)
    print("Auto Insurance Churn Prediction System Starting…")
    print("=" * 50)
    if setup_database():
        print("Database ready!  Server starting at http://0.0.0.0:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Database setup failed.")
