import json
import os
import boto3
import pandas as pd
import numpy as np
from decimal import Decimal
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Environment variables
PROCESSED_TABLE_NAME = os.environ.get("PROCESSED_TABLE_NAME", "strava_processed_pau")
PREDICTIONS_TABLE_NAME = os.environ.get("PREDICTIONS_TABLE_NAME", "strava_predictions_pau")
REGION_NAME = os.environ.get("REGION_NAME", "eu-central-1")

dynamodb = boto3.resource("dynamodb", region_name=REGION_NAME)


def decimal_to_float(obj):
    """Recursively convert Decimal to float for DynamoDB compatibility"""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decimal_to_float(x) for x in obj]
    return obj


def convert_to_dynamodb_decimal(obj):
    """Recursively convert float to Decimal for DynamoDB storage"""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return Decimal(str(round(obj, 6)))
    elif isinstance(obj, dict):
        return {k: convert_to_dynamodb_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dynamodb_decimal(v) for v in obj]
    return obj


def compute_zones_from_max_hr(max_hr):
    """Calculate HR zones from user's max HR"""
    max_hr = int(max_hr)
    return {
        "z1": {"min": int(max_hr * 0.50), "max": int(max_hr * 0.60), "name": "Recovery"},
        "z2": {"min": int(max_hr * 0.60), "max": int(max_hr * 0.70), "name": "Aerobic"},
        "z3": {"min": int(max_hr * 0.70), "max": int(max_hr * 0.80), "name": "Tempo"},
        "z4": {"min": int(max_hr * 0.80), "max": int(max_hr * 0.90), "name": "Threshold"},
        "z5": {"min": int(max_hr * 0.90), "max": max_hr, "name": "Maximum"}
    }


def assign_hr_zone(avg_hr, max_hr):
    """Assign HR zone based on percentage of max HR"""
    if pd.isna(avg_hr) or avg_hr <= 0:
        return None
    frac = avg_hr / max_hr
    if frac < 0.60:
        return "z1"
    elif frac < 0.70:
        return "z2"
    elif frac < 0.80:
        return "z3"
    elif frac < 0.90:
        return "z4"
    else:
        return "z5"


def pace_to_time(distance_km, pace_min_per_km):
    """Convert pace and distance to time string (HH:MM:SS)"""
    total_minutes = distance_km * pace_min_per_km
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    seconds = int((total_minutes - int(total_minutes)) * 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def pace_decimal_to_mmss(pace_dec):
    """Convert decimal pace (e.g., 4.8 = 4m48s) to mm:ss string"""
    minutes = int(pace_dec)
    seconds = int(round((pace_dec - minutes) * 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}"


EFFORT_HR_FRAC = {
    "easy": 0.62,
    "moderate": 0.72,
    "tempo": 0.82,
    "hard": 0.90,
    "race": 0.96
}

TERRAIN_ELEV_GAIN_PER_10K = {
    "flat": 15,
    "rolling": 60,
    "hilly": 150
}

EFFORT_TO_ZONES = {
    "easy": ["z1", "z2"],
    "moderate": ["z2", "z3"],
    "tempo": ["z3", "z4"],
    "hard": ["z4", "z5"],
    "race": ["z5"]
}


def build_effort_score_column(df_zone, max_hr):
    """Calculate effort score combining HR and pace"""
    df = df_zone.copy()
    
    # Check if HR data exists
    has_hr_data = 'average_heartrate' in df.columns and df['average_heartrate'].notna().any()
    
    if has_hr_data:
        df["hr_effort"] = (df["average_heartrate"] / max_hr).clip(0.5, 1.0)
        high_hr = df[df["average_heartrate"] > max_hr * 0.85]
    else:
        # No HR data - use default effort based on pace
        df["hr_effort"] = 0.75
        high_hr = df.nsmallest(max(5, len(df) // 10), 'pace_min_per_km')
    
    if len(high_hr) >= 5:
        threshold_pace = high_hr["pace_min_per_km"].median()
    else:
        threshold_pace = df["pace_min_per_km"].median()
    
    df["pace_effort"] = (threshold_pace / df["pace_min_per_km"]).clip(0.5, 1.2)
    
    if has_hr_data:
        df["effort_score"] = (0.6 * df["hr_effort"] + 0.4 * df["pace_effort"]).clip(0.5, 1.0)
    else:
        # No HR - use pace only
        df["effort_score"] = df["pace_effort"].clip(0.5, 1.0)
    
    return df


def filter_training_rows_by_effort(df, effort, max_hr):
    """Filter training data to match target effort level"""
    df = df.copy()
    
    # Check if HR data exists
    has_hr_data = 'average_heartrate' in df.columns and df['average_heartrate'].notna().any()
    
    if not has_hr_data:
        # No HR data - use all data and filter by pace instead
        print("⚠️ No HR data - using pace-based filtering")
        df = df.dropna(subset=["pace_min_per_km"])
        # Use fastest 50% of runs as proxy for effort zones
        median_pace = df["pace_min_per_km"].median()
        df_zone = df[df["pace_min_per_km"] <= median_pace * 1.2].copy()
        if len(df_zone) < 10:
            df_zone = df.copy()
        return df_zone
    
    # Has HR data - use original logic
    df = df.dropna(subset=["average_heartrate", "pace_min_per_km"])
    df["hr_zone"] = df["average_heartrate"].apply(lambda hr: assign_hr_zone(hr, max_hr))
    target_zones = EFFORT_TO_ZONES.get(effort, ["z2", "z3"])
    df_zone = df[df["hr_zone"].isin(target_zones)]
    if len(df_zone) < 10:
        relaxed = set(target_zones)
        relaxed.update(["z2", "z3", "z4"])
        df_zone = df[df["hr_zone"].isin(list(relaxed))]
    if len(df_zone) < 10:
        df_zone = df.copy()
    return df_zone


def prepare_training_data(df, effort, max_hr):
    """Prepare features and target for model training"""
    df_zone = filter_training_rows_by_effort(df, effort, max_hr)
    df_zone = build_effort_score_column(df_zone, max_hr)
    if "recency_weight" not in df_zone.columns:
        df_zone["recency_weight"] = 1.0
    feature_cols = [
        "distance",
        "total_elevation_gain",
        "elevation_per_km",
        "day_of_week",
        "hour_of_day",
        "recency_weight",
        "effort_score"
    ]
    feature_cols = [f for f in feature_cols if f in df_zone.columns]
    df_clean = df_zone[feature_cols + ["pace_min_per_km"]].dropna()
    if len(df_clean) < 10:
        raise ValueError(f"Not enough training data: {len(df_clean)} rows (need at least 10)")
    X = df_clean[feature_cols]
    y = df_clean["pace_min_per_km"]
    return X, y, feature_cols, df_zone


def train_rf_model(X, y):
    """Train Random Forest model and calculate metrics"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, shuffle=True
    )
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = float(np.mean(np.abs(y_pred - y_test)))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    ss_res = float(((y_test - y_pred) ** 2).sum())
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }
    return model, metrics, fi


def build_user_row(df_zone, distance_km, effort, terrain, feature_cols, max_hr):
    """Build feature row for prediction based on user input"""
    df = df_zone.copy()
    base = df[feature_cols].median(numeric_only=True).to_dict()
    distance_km = float(distance_km)
    distance_m = distance_km * 1000.0
    base["distance"] = distance_m
    gain_per_10k = TERRAIN_ELEV_GAIN_PER_10K.get(terrain.lower(), 15)
    total_gain = gain_per_10k * (distance_km / 10.0)
    base["total_elevation_gain"] = total_gain
    base["elevation_per_km"] = total_gain / max(distance_km, 1e-6)
    if "recency_weight" in base:
        base["recency_weight"] = 1.0
    effort_frac = EFFORT_HR_FRAC.get(effort.lower(), 0.72)
    if "effort_score" in base:
        base["effort_score"] = effort_frac
    row = {f: float(base.get(f, 0.0)) for f in feature_cols}
    return pd.DataFrame([row])


def lambda_handler(event, context):
    """
    Main Lambda handler for running performance prediction
    """
    print("Event received:", json.dumps(event))
    
    if "body" in event and isinstance(event["body"], str):
        body = json.loads(event["body"])
    else:
        body = event

    athlete_id = body.get("athlete_id")
    distance_km = body.get("distance_km")
    effort = body.get("effort", "").lower()
    terrain = body.get("terrain", "").lower()
    max_hr = body.get("max_hr")

    if not all([athlete_id, distance_km, effort, terrain, max_hr]):
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({
                "error": "Missing required parameters",
                "required": ["athlete_id", "distance_km", "effort", "terrain", "max_hr"]
            })
        }

    is_csv_user = str(athlete_id).startswith('csv_')

    if is_csv_user:
        processed_table_name = os.environ.get("PROCESSED_TABLE_CSV", "strava_processed_csv")
        predictions_table_name = os.environ.get("PREDICTIONS_TABLE_CSV", "strava_predictions_csv")
        print(f"Using CSV tables for {athlete_id}")
    else:
        processed_table_name = PROCESSED_TABLE_NAME
        predictions_table_name = PREDICTIONS_TABLE_NAME
        print(f"Using OAuth tables for {athlete_id}")
    
    processed_table = dynamodb.Table(processed_table_name)
    predictions_table = dynamodb.Table(predictions_table_name)

    try:
        max_hr = float(max_hr)
        distance_km = float(distance_km)
        
        if not (180 <= max_hr <= 210):
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "max_hr must be between 180 and 210"})
            }
        
        if not (1 <= distance_km <= 100):
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "distance_km must be between 1 and 100"})
            }
            
        if effort not in EFFORT_HR_FRAC:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({
                    "error": f"Invalid effort level: {effort}",
                    "valid_options": list(EFFORT_HR_FRAC.keys())
                })
            }
            
        if terrain not in TERRAIN_ELEV_GAIN_PER_10K:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({
                    "error": f"Invalid terrain: {terrain}",
                    "valid_options": list(TERRAIN_ELEV_GAIN_PER_10K.keys())
                })
            }
            
    except ValueError as e:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": f"Invalid numeric parameter: {str(e)}"})
        }

    print(f"Loading processed data for athlete {athlete_id}...")
    try:
        if is_csv_user:
            resp = processed_table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key("athlete_id").eq(athlete_id)
            )
        else:
            resp = processed_table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key("athlete_id").eq(int(athlete_id))
            )
        
        items = [decimal_to_float(i) for i in resp.get("Items", [])]
        df = pd.DataFrame(items)

        if df.empty:
            return {
                "statusCode": 404,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({
                    "error": "No training data found for this athlete",
                    "message": "Please sync your Strava activities first"
                })
            }
            
        print(f"Loaded {len(df)} activities")
        
    except Exception as e:
        print(f"Error loading processed data: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": "Failed to load training data"})
        }

    # Check for required columns - HR is optional for CSV users
    required_cols = ["pace_min_per_km", "distance"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": f"Missing columns: {missing}"})
        }
    
    # Check if HR data is available
    has_hr_data = 'average_heartrate' in df.columns and df['average_heartrate'].notna().any()
    
    if not has_hr_data:
        print("⚠️ No heart rate data available - using pace-only prediction")
        # Add default HR column for compatibility
        df['average_heartrate'] = max_hr * 0.72  # Default moderate effort

    print(f"Preparing training data for effort: {effort}...")
    try:
        X, y, feature_cols, df_zone = prepare_training_data(df, effort, max_hr)
        print(f"Training samples: {len(X)}, Zone-filtered: {len(df_zone)}")
    except ValueError as e:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({
                "error": str(e),
                "message": "Need more training data for accurate predictions"
            })
        }
    except Exception as e:
        print(f"Training data prep error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": "Training data preparation failed"})
        }

    print("Training model...")
    try:
        model, metrics, fi = train_rf_model(X, y)
        print(f"Model trained - MAE: {metrics['mae']:.2f}, R2: {metrics['r2']:.3f}")
    except Exception as e:
        print(f"Model training error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": "Model training failed"})
        }

    print(f"Predicting for {distance_km}km, {effort}, {terrain}...")
    try:
        user_row = build_user_row(df_zone, distance_km, effort, terrain, feature_cols, max_hr)
        rf_pred_pace = float(model.predict(user_row)[0])
        zone_mean_pace = float(df_zone["pace_min_per_km"].mean())
        final_pace_dec = 0.7 * rf_pred_pace + 0.3 * zone_mean_pace
        print(f"Predicted pace: {final_pace_dec:.2f} min/km")
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": "Prediction failed"})
        }

    estimated_time = pace_to_time(distance_km, final_pace_dec)
    pace_display = pace_decimal_to_mmss(final_pace_dec)
    effort_frac = EFFORT_HR_FRAC.get(effort, 0.72)
    expected_avg_hr = int(effort_frac * max_hr)
    expected_max_hr = int(min(expected_avg_hr + 15, max_hr))
    hr_zones = compute_zones_from_max_hr(max_hr)
    top_features = fi.head(3).to_dict('records')
    top_features_formatted = {
        item['feature']: round(float(item['importance']), 3) 
        for item in top_features
    }

    result = {
        "recommended_pace": pace_display,
        "recommended_pace_decimal": round(final_pace_dec, 2),
        "estimated_time": estimated_time,
        "expected_avg_hr": expected_avg_hr,
        "expected_max_hr": expected_max_hr,
        "hr_zones": hr_zones,
        "model_metrics": {
            "mae": round(metrics["mae"], 2),
            "rmse": round(metrics["rmse"], 2),
            "r2": round(metrics["r2"], 3),
            "mape": round(metrics["mape"], 1),
            "training_samples": metrics["train_samples"],
            "test_samples": metrics["test_samples"],
            "zone_filtered_rows": len(df_zone)
        },
        "top_features": top_features_formatted
    }

    try:
        save_item = {
            "athlete_id": athlete_id if is_csv_user else int(athlete_id),
            "prediction_id": f"{athlete_id}_{int(pd.Timestamp.utcnow().timestamp())}",
            "prediction_timestamp": pd.Timestamp.utcnow().isoformat(),
            "inputs": {
                "distance_km": distance_km,
                "effort": effort,
                "terrain": terrain,
                "max_hr": int(max_hr)
            },
            "outputs": result
        }
        save_item = convert_to_dynamodb_decimal(save_item)
        predictions_table.put_item(Item=save_item)
        print("Prediction saved to DynamoDB")
    except Exception as e:
        print(f"Warning: failed to save prediction: {str(e)}")

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(result, default=str)
    }