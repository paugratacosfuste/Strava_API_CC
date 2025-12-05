import json
import os
import boto3
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from decimal import Decimal
from datetime import datetime, timezone
from collections import defaultdict
from boto3.dynamodb.conditions import Key

# Default table names (will be overridden for CSV users)
ACTIVITIES_TABLE_NAME = os.environ.get('ACTIVITIES_TABLE_NAME', 'strava_activities_pau')
PROCESSED_TABLE_NAME = os.environ.get('PROCESSED_TABLE_NAME', 'strava_processed_pau')
REGION_NAME = os.environ.get('REGION_NAME', 'eu-central-1')

dynamodb = boto3.resource('dynamodb', region_name=REGION_NAME)


# ========================================
# BEST EFFORTS CALCULATION (NEW - FOR CSV USERS)
# ========================================

def calculate_best_efforts_from_activities(athlete_id, activities_data):
    """
    Calculate best efforts for standard distances from activities.
    This replicates what OAuth gets from Strava API for CSV users.
    Saves to strava_best_efforts_csv table.
    
    Args:
        athlete_id: CSV athlete ID (e.g., 'csv_mati')
        activities_data: DataFrame with processed activities
    
    Returns:
        Number of best efforts saved
    """
    from decimal import Decimal
    
    # Connect to best_efforts CSV table
    best_efforts_table_name = "strava_best_efforts_csv"
    best_efforts_table = dynamodb.Table(best_efforts_table_name)
    
    # Standard distances to track (meters, tolerance)
    STANDARD_DISTANCES = {
        '400m': (400, 0.05),
        '1K': (1000, 0.05),
        '5K': (5000, 0.05),
        '10K': (10000, 0.05),
        'Half Marathon': (21097.5, 0.05),
        'Marathon': (42195, 0.05)
    }
    
    print(f"\n{'='*60}")
    print(f"CALCULATING BEST EFFORTS FOR: {athlete_id}")
    print(f"{'='*60}")
    print(f"Total activities to analyze: {len(activities_data)}")
    
    best_efforts_saved = 0
    
    # Convert DataFrame to list of dicts for easier processing
    activities = activities_data.to_dict('records')
    
    for distance_name, (target_meters, tolerance) in STANDARD_DISTANCES.items():
        min_distance = target_meters * (1 - tolerance)
        max_distance = target_meters * (1 + tolerance)
        
        # Find activities in range
        candidates = []
        for activity in activities:
            distance = activity.get('distance', 0)
            elapsed_time = activity.get('elapsed_time', 0)
            
            # Activity must be within distance range and have valid time
            if min_distance <= distance <= max_distance and elapsed_time > 0:
                candidates.append(activity)
        
        if candidates:
            # Find fastest (minimum elapsed_time)
            best = min(candidates, key=lambda x: x['elapsed_time'])
            
            # Calculate pace
            pace_min_per_km = (best['elapsed_time'] / 60.0) / (best['distance'] / 1000.0)
            
            # Save to DynamoDB
            try:
                best_efforts_table.put_item(Item={
                    'athlete_id': athlete_id,
                    'effort_name': distance_name,
                    'distance': Decimal(str(best['distance'])),
                    'elapsed_time': Decimal(str(best['elapsed_time'])),
                    'moving_time': Decimal(str(best.get('moving_time', best['elapsed_time']))),
                    'start_date_local': str(best.get('start_date_local', '')),
                    'name': str(best.get('name', f'Best {distance_name}')),
                    'pace_min_per_km': Decimal(str(round(pace_min_per_km, 2)))
                })
                
                # Format time for display
                seconds = int(best['elapsed_time'])
                if distance_name in ['400m', '1K']:
                    minutes = seconds // 60
                    secs = seconds % 60
                    time_str = f"{minutes}:{secs:02d}"
                else:
                    hours = seconds // 3600
                    minutes = (seconds % 3600) // 60
                    secs = seconds % 60
                    time_str = f"{hours}:{minutes:02d}:{secs:02d}"
                
                date_str = str(best.get('start_date_local', 'N/A'))
                if len(date_str) > 10:
                    date_str = date_str[:10]
                
                print(f"  ✓ {distance_name:15s} | {time_str:10s} | {pace_min_per_km:.2f} min/km | {date_str}")
                best_efforts_saved += 1
                
            except Exception as e:
                print(f"  ✗ {distance_name}: Error saving - {e}")
        else:
            print(f"  - {distance_name:15s} | No activities found in range ({min_distance:.0f}m - {max_distance:.0f}m)")
    
    print(f"{'='*60}")
    print(f"✓ SAVED {best_efforts_saved}/6 BEST EFFORTS TO strava_best_efforts_csv")
    print(f"{'='*60}\n")
    
    return best_efforts_saved


# ========================================
# AUTOMATIC CLEANUP FUNCTION
# ========================================

def cleanup_duplicates(athlete_id, is_csv_user, activities_table, processed_table):
    """
    Automatically remove duplicate activities.
    Returns: number of duplicates removed
    """
    try:
        print(f"🧹 Checking for duplicates for athlete {athlete_id}...")
        
        # Query all activities for this athlete
        if is_csv_user:
            response = activities_table.query(
                KeyConditionExpression=Key('athlete_id').eq(athlete_id)
            )
        else:
            response = activities_table.query(
                KeyConditionExpression=Key('athlete_id').eq(int(athlete_id))
            )
        
        items = response.get('Items', [])
        
        # Continue pagination if needed
        while 'LastEvaluatedKey' in response:
            if is_csv_user:
                response = activities_table.query(
                    KeyConditionExpression=Key('athlete_id').eq(athlete_id),
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
            else:
                response = activities_table.query(
                    KeyConditionExpression=Key('athlete_id').eq(int(athlete_id)),
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
            items.extend(response.get('Items', []))
        
        print(f"Found {len(items)} total activities in raw table")
        
        # Find duplicates
        if is_csv_user:
            # For CSV: duplicates by date + distance
            seen = {}
            duplicates = []
            for item in items:
                key = f"{item.get('start_date_local', '')}_{item.get('distance', '')}"
                if key in seen:
                    duplicates.append(item)
                else:
                    seen[key] = item
        else:
            # For OAuth: duplicates by activity_id
            activity_groups = defaultdict(list)
            for item in items:
                activity_groups[item.get('activity_id')].append(item)
            
            duplicates = []
            for activity_id, group in activity_groups.items():
                if len(group) > 1:
                    # Keep first occurrence, mark rest as duplicates
                    duplicates.extend(group[1:])
        
        if len(duplicates) == 0:
            print("✓ No duplicates found - data is clean")
            return 0
        
        print(f"Found {len(duplicates)} duplicate activities - removing...")
        
        # Delete duplicates from activities table
        with activities_table.batch_writer() as batch:
            for item in duplicates:
                batch.delete_item(Key={
                    'athlete_id': item['athlete_id'],
                    'activity_id': item['activity_id']
                })
        
        # Delete duplicates from processed table (may not exist yet)
        deleted_from_processed = 0
        with processed_table.batch_writer() as batch:
            for item in duplicates:
                try:
                    batch.delete_item(Key={
                        'athlete_id': item['athlete_id'],
                        'activity_id': item['activity_id']
                    })
                    deleted_from_processed += 1
                except:
                    pass  # Activity may not exist in processed table yet
        
        print(f"✓ Cleaned {len(duplicates)} duplicates from activities table")
        print(f"✓ Cleaned {deleted_from_processed} duplicates from processed table")
        
        return len(duplicates)
        
    except Exception as e:
        print(f"⚠ Cleanup error (non-critical): {e}")
        import traceback
        traceback.print_exc()
        return 0


# ========================================
# DATA CONVERSION FUNCTIONS
# ========================================

def decimal_to_float(obj):
    """Recursively convert Decimal to float"""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decimal_to_float(x) for x in obj]
    return obj


def convert_to_dynamodb_decimal(obj):
    """Recursively convert float to Decimal, handling NaN and inf"""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return Decimal(str(round(float(obj), 6)))
    
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        return Decimal(str(round(float(obj), 6)))
    elif isinstance(obj, dict):
        return {k: convert_to_dynamodb_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dynamodb_decimal(v) for v in obj]
    return obj


# ========================================
# FEATURE ENGINEERING FUNCTIONS
# ========================================

def engineer_features(df_raw):
    """Create derived features from raw activity data"""
    print(f"Engineering features for {len(df_raw)} activities...")
    df = df_raw.copy()

    # Convert all numeric columns
    numeric_cols = [
        col for col in df.columns
        if col not in ['activity_id', 'athlete_id', 'start_date_local', 'name', 'type']
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Pace (min/km)
    df['pace_min_per_km'] = np.where(
        (df['average_speed'] > 0),
        1000 / (df['average_speed'] * 60),
        np.nan
    )

    # Elevation per km
    df['elevation_per_km'] = np.where(
        (df['distance'] > 0),
        (df['total_elevation_gain'] * 1000) / df['distance'],
        np.nan
    )

    # Moving efficiency (km/h)
    df['moving_efficiency'] = np.where(
        (df['moving_time'] > 0),
        (df['distance'] / 1000) / (df['moving_time'] / 3600),
        np.nan
    )

    # Rest ratio (elapsed / moving time)
    df['rest_ratio'] = np.where(
        (df['moving_time'] > 0),
        df['elapsed_time'] / df['moving_time'],
        np.nan
    )

    # Time-based features
    if 'start_date_local' in df.columns:
        df['start_date_local'] = pd.to_datetime(df['start_date_local'], errors='coerce')
        df['day_of_week'] = df['start_date_local'].dt.dayofweek
        df['hour_of_day'] = df['start_date_local'].dt.hour
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['month'] = df['start_date_local'].dt.month
        df['week_of_year'] = df['start_date_local'].dt.isocalendar().week

    # Replace inf values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    print(f"Features engineered: pace, elevation_per_km, moving_efficiency, rest_ratio, time features")

    return df


def perform_imputation(df):
    """Impute missing values using KNN"""
    impute_cols = [
        'average_heartrate', 'max_heartrate', 'average_cadence',
        'pace_min_per_km', 'moving_time', 'distance', 'total_elevation_gain'
    ]
    available = [c for c in impute_cols if c in df.columns]

    if len(available) < 3:
        print("Not enough columns for imputation, skipping...")
        return df

    print(f"Imputing {len(available)} columns using KNN...")
    
    # Only impute rows with at least 50% non-null values
    null_pct = df[available].isnull().sum(axis=1) / len(available)
    impute_mask = null_pct < 0.5
    
    if impute_mask.sum() > 0:
        df.loc[impute_mask, available] = KNNImputer(n_neighbors=5).fit_transform(
            df.loc[impute_mask, available]
        )
    
    print(f"Imputed {impute_mask.sum()} rows")

    return df


def add_rolling_features(df):
    """Add rolling averages and trends"""
    print("Adding rolling features...")
    
    if 'start_date_local' in df.columns:
        df = df.sort_values('start_date_local')

    cols = ['distance', 'moving_time', 'average_heartrate', 'pace_min_per_km', 'elevation_per_km']
    cols = [c for c in cols if c in df.columns]

    for col in cols:
        # 5-run rolling average
        df[f'{col}_rolling_avg_5'] = df[col].rolling(5, min_periods=1).mean()
        
        # 10-run rolling average (if enough data)
        if len(df) >= 10:
            df[f'{col}_rolling_avg_10'] = df[col].rolling(10, min_periods=3).mean()

    # Pace improvement (negative = getting faster)
    if 'pace_min_per_km' in cols:
        df['pace_improvement'] = df['pace_min_per_km_rolling_avg_5'] - df['pace_min_per_km']
        
    print(f"Rolling features added for {len(cols)} metrics")

    return df


def add_recency_weight(df):
    """Add recency weight to prioritize recent training"""
    print("Adding recency weights...")
    
    df['activity_date'] = df['start_date_local']
    now = datetime.now(timezone.utc)
    
    # Make activity_date timezone-aware if it isn't
    if df['activity_date'].dt.tz is None:
        df['activity_date'] = df['activity_date'].dt.tz_localize('UTC')
    
    df['days_ago'] = (now - df['activity_date']).dt.days

    def weight(d):
        if pd.isna(d):
            return 0.3
        if d <= 30:
            return 1.0
        elif d <= 90:
            return 0.8
        elif d <= 180:
            return 0.5
        else:
            return 0.3

    df['recency_weight'] = df['days_ago'].apply(weight)
    
    print(f"Recency weights assigned (avg: {df['recency_weight'].mean():.2f})")

    return df


def compute_effort_score(df, max_hr=None):
    """Compute effort score combining HR and pace - handles missing HR data"""
    print("Computing effort scores...")
    
    # Determine max HR
    if max_hr is None:
        if 'age' in df.columns and not df['age'].isna().all():
            avg_age = df['age'].median()
            max_hr = 220 - avg_age
            print(f"Using age-estimated max HR: {max_hr}")
        else:
            max_hr = 190  # Conservative default
            print(f"Using default max HR: {max_hr}")

    # Check if HR data exists before using it
    has_hr_data = 'average_heartrate' in df.columns and df['average_heartrate'].notna().any()
    
    if has_hr_data:
        # HR effort (normalized to max HR)
        df['hr_effort'] = (df['average_heartrate'] / max_hr).clip(0.5, 1.0)
        print("✓ Calculated HR effort from heart rate data")
    else:
        # No heart rate data available - use pace as proxy
        df['hr_effort'] = 0.75  # Default moderate effort
        print("⚠ No heart rate data - using default effort (0.75)")

    # Pace effort (relative to fast runs)
    recent_cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=60)
    
    if 'activity_date' in df.columns:
        recent_df = df[df['activity_date'] >= recent_cutoff]
    else:
        recent_df = df
    
    # Handle fast runs detection with or without HR data
    if has_hr_data:
        # High-effort runs (>85% max HR)
        fast_runs = recent_df[recent_df['average_heartrate'] > max_hr * 0.85]
    else:
        # Use pace as proxy - take fastest 10% of runs
        fast_runs = recent_df.nsmallest(max(5, len(recent_df) // 10), 'pace_min_per_km')
    
    if len(fast_runs) >= 5:
        threshold_pace = fast_runs['pace_min_per_km'].median()
        print(f"Threshold pace from {len(fast_runs)} fast runs: {threshold_pace:.2f} min/km")
    else:
        threshold_pace = df['pace_min_per_km'].median()
        print(f"Threshold pace from all runs: {threshold_pace:.2f} min/km")

    df['pace_effort'] = (threshold_pace / df['pace_min_per_km']).clip(0.5, 1.2)

    # Blend: 60% HR, 40% pace (if HR available) or 100% pace (if no HR)
    if has_hr_data:
        df['effort_score'] = (0.6 * df['hr_effort'] + 0.4 * df['pace_effort']).clip(0.5, 1.0)
    else:
        df['effort_score'] = df['pace_effort'].clip(0.5, 1.0)
    
    print(f"Effort scores computed (avg: {df['effort_score'].mean():.2f})")

    return df


def clean_and_filter(df):
    """Remove invalid/outlier activities"""
    print(f"Cleaning data...")
    initial_count = len(df)
    
    # Remove very short activities (< 1 minute)
    df = df[df["moving_time"] >= 60]
    
    # Remove unrealistic paces (< 2 min/km or > 12 min/km)
    if 'pace_min_per_km' in df.columns:
        df = df[(df["pace_min_per_km"] >= 2) & (df["pace_min_per_km"] <= 12)]
    
    # Remove unrealistic distances (< 0.5 km or > 150 km)
    if 'distance' in df.columns:
        df = df[(df["distance"] >= 500) & (df["distance"] <= 150000)]
    
    # Remove unrealistic HR values only if column exists and has data
    if 'average_heartrate' in df.columns:
        # Keep rows with NaN OR valid HR values
        df = df[(df["average_heartrate"].isna()) | ((df["average_heartrate"] >= 60) & (df["average_heartrate"] <= 220))]
    
    removed = initial_count - len(df)
    print(f"Removed {removed} invalid activities ({removed/initial_count*100:.1f}%)")
    
    return df


# ========================================
# MAIN HANDLER
# ========================================

def lambda_handler(event, context):
    """
    Process raw Strava activities into ML-ready features.
    NOW WITH:
    - Automatic duplicate cleanup
    - Best efforts calculation for CSV users (NEW!)
    
    Input: {"athlete_id": "csv_mati", "max_hr": 192, "source": "csv"}
    """
    print("=" * 60)
    print("DATA PROCESSOR LAMBDA STARTED")
    print("=" * 60)
    
    # Get athlete_id FIRST
    athlete_id = event.get("athlete_id")
    max_hr = event.get("max_hr")
    source = event.get('source', 'oauth')
    
    if not athlete_id:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing athlete_id"})
        }
    
    # Check if it's a CSV user
    is_csv_user = str(athlete_id).startswith('csv_')
    
    # Determine which tables to use
    if is_csv_user or source == 'csv':
        # Use CSV tables
        activities_table_name = os.environ.get('ACTIVITIES_TABLE_CSV', 'strava_activities_csv')
        processed_table_name = os.environ.get('PROCESSED_TABLE_CSV', 'strava_processed_csv')
        print(f"✓ Using CSV tables for {athlete_id}")
    else:
        # Use OAuth tables
        activities_table_name = ACTIVITIES_TABLE_NAME
        processed_table_name = PROCESSED_TABLE_NAME
        print(f"✓ Using OAuth tables for {athlete_id}")
    
    # Create table references
    activities_table = dynamodb.Table(activities_table_name)
    processed_table = dynamodb.Table(processed_table_name)

    # ========================================
    # STEP 1: AUTOMATIC CLEANUP
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 1: AUTOMATIC DUPLICATE CLEANUP")
    print("=" * 60)
    
    try:
        duplicates_removed = cleanup_duplicates(athlete_id, is_csv_user, activities_table, processed_table)
        if duplicates_removed > 0:
            print(f"\n✅ Cleaned {duplicates_removed} duplicate activities")
        else:
            print("\n✅ No duplicates found - data is clean")
    except Exception as e:
        print(f"\n⚠️ Cleanup failed (non-critical): {e}")
        print("Continuing with processing...")
    
    # ========================================
    # STEP 2: LOAD ACTIVITIES
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 2: LOADING ACTIVITIES")
    print("=" * 60)
    
    print(f"\nLoading activities for athlete {athlete_id}...")
    try:
        # Handle both string (csv_maria) and int (156086407) athlete IDs
        if is_csv_user:
            resp = activities_table.query(
                KeyConditionExpression=Key('athlete_id').eq(athlete_id)
            )
        else:
            resp = activities_table.query(
                KeyConditionExpression=Key('athlete_id').eq(int(athlete_id))
            )
        
        items = [decimal_to_float(i) for i in resp.get("Items", [])]
        df = pd.DataFrame(items)

        if df.empty:
            return {
                "statusCode": 404,
                "body": json.dumps({
                    "message": "No activities found for this athlete",
                    "athlete_id": athlete_id
                })
            }
            
        print(f"✓ Loaded {len(df)} activities")
        
    except Exception as e:
        print(f"✗ Error loading activities: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to load activities: {str(e)}"})
        }

    # ========================================
    # STEP 3: PROCESSING PIPELINE
    # ========================================
    try:
        print("\n" + "=" * 60)
        print("STEP 3: PROCESSING PIPELINE")
        print("=" * 60)
        
        df = engineer_features(df)
        df = perform_imputation(df)
        df = add_rolling_features(df)
        df = add_recency_weight(df)
        df = compute_effort_score(df, max_hr=max_hr)
        df = clean_and_filter(df)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Final dataset: {len(df)} activities")
        print(f"Features: {len(df.columns)} columns")
        
    except Exception as e:
        print(f"✗ Processing pipeline error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Processing failed: {str(e)}"})
        }

    # ========================================
    # STEP 3.5: CALCULATE BEST EFFORTS (CSV USERS ONLY - NEW!)
    # ========================================
    best_efforts_count = 0
    if is_csv_user:
        print("\n" + "=" * 60)
        print("STEP 3.5: CALCULATE BEST EFFORTS (CSV USER)")
        print("=" * 60)
        
        try:
            best_efforts_count = calculate_best_efforts_from_activities(athlete_id, df)
            print(f"\n✅ Best efforts calculated: {best_efforts_count}/6")
        except Exception as e:
            print(f"\n⚠️ Best efforts calculation failed (non-critical): {e}")
            import traceback
            traceback.print_exc()

    # ========================================
    # STEP 4: SAVE TO DYNAMODB
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 4: SAVING TO DYNAMODB")
    print("=" * 60)
    
    # Convert timestamps to strings
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Save processed activities
    print(f"Saving {len(df)} processed activities to {processed_table_name}...")
    saved = 0
    failed = 0
    
    try:
        with processed_table.batch_writer() as batch:
            for idx, row in df.iterrows():
                try:
                    item = convert_to_dynamodb_decimal(row.to_dict())
                    # Handle both string and int athlete_ids
                    if is_csv_user:
                        item['athlete_id'] = str(item['athlete_id'])
                    else:
                        item['athlete_id'] = int(item['athlete_id'])
                    item['activity_id'] = int(item['activity_id'])
                    batch.put_item(Item=item)
                    saved += 1
                except Exception as e:
                    failed += 1
                    if failed <= 5:
                        print(f"Failed to save activity {row.get('activity_id')}: {str(e)}")
        
        print(f"\n✓ Saved {saved} activities")
        if failed > 0:
            print(f"✗ Failed to save {failed} activities")
            
    except Exception as e:
        print(f"✗ Batch write error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Failed to save processed data"})
        }

    # ========================================
    # FINAL SUMMARY
    # ========================================
    summary = {
        "athlete_id": athlete_id,
        "duplicates_removed": duplicates_removed,
        "activities_processed": saved,
        "activities_failed": failed,
        "best_efforts_calculated": best_efforts_count,  # NEW!
        "source": 'csv' if is_csv_user else 'oauth',
        "tables_used": {
            'activities': activities_table_name,
            'processed': processed_table_name,
            'best_efforts': 'strava_best_efforts_csv' if is_csv_user else 'strava_best_efforts_pau'
        },
        "features": list(df.columns),
        "date_range": {
            "earliest": str(df['start_date_local'].min()) if 'start_date_local' in df.columns and len(df) > 0 else None,
            "latest": str(df['start_date_local'].max()) if 'start_date_local' in df.columns and len(df) > 0 else None
        },
        "summary_stats": {
            "total_distance_km": float(df['distance'].sum() / 1000) if 'distance' in df.columns else 0,
            "total_time_hours": float(df['moving_time'].sum() / 3600) if 'moving_time' in df.columns else 0,
            "avg_pace_min_per_km": float(df['pace_min_per_km'].mean()) if 'pace_min_per_km' in df.columns else 0,
            "avg_hr": float(df['average_heartrate'].mean()) if 'average_heartrate' in df.columns and df['average_heartrate'].notna().any() else 0
        }
    }

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE ✅")
    print("=" * 60)
    if duplicates_removed > 0:
        print(f"🧹 Cleaned {duplicates_removed} duplicates")
    print(f"📊 Processed {saved} activities")
    if best_efforts_count > 0:
        print(f"🏆 Calculated {best_efforts_count} best efforts")
    print(f"📦 Saved to {processed_table_name}")
    print("=" * 60)
    print(json.dumps(summary, indent=2, default=str))

    return {
        "statusCode": 200,
        "body": json.dumps(summary, default=str)
    }