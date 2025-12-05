import json
import os
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from decimal import Decimal

import boto3

REGION_NAME = os.environ.get("REGION_NAME", "eu-central-1")
dynamodb = boto3.resource("dynamodb", region_name=REGION_NAME)


def _decimal_to_float(obj):
    if isinstance(obj, list):
        return [_decimal_to_float(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _decimal_to_float(v) for k, v in obj.items()}
    if isinstance(obj, Decimal):
        return float(obj)
    return obj


def _parse_date(item):
    date_str = (
        item.get("start_date_local")
        or item.get("start_date")
        or item.get("start_date_utc")
    )
    if not date_str:
        return None

    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _floor_to_monday(dt: datetime) -> datetime:
    return dt - timedelta(days=dt.weekday())


def _interpolate_nulls(series):
    """Fill None values with linear interpolation between valid points"""
    if not series:
        return series
    
    series = list(series)
    n = len(series)
    
    # Find first and last non-null indices
    first_valid = None
    last_valid = None
    for i in range(n):
        if series[i] is not None:
            if first_valid is None:
                first_valid = i
            last_valid = i
    
    # If no valid values, return zeros
    if first_valid is None:
        return [0.0] * n
    
    # Fill beginning nulls with first valid value
    for i in range(first_valid):
        series[i] = series[first_valid]
    
    # Fill end nulls with last valid value
    for i in range(last_valid + 1, n):
        series[i] = series[last_valid]
    
    # Interpolate middle nulls
    i = first_valid
    while i < last_valid:
        if series[i] is None:
            # Find next non-null
            j = i + 1
            while j <= last_valid and series[j] is None:
                j += 1
            
            # Linear interpolation
            if j <= last_valid:
                start_val = series[i - 1]
                end_val = series[j]
                gap = j - i + 1
                for k in range(1, gap):
                    series[i - 1 + k] = start_val + (end_val - start_val) * k / gap
            i = j
        else:
            i += 1
    
    return series


def _linear_forecast(series, k):
    """Simple linear regression forecast for next k points"""
    # Interpolate nulls first
    series = _interpolate_nulls(series)
    
    # Use only last 8 weeks for trend
    recent_series = series[-8:] if len(series) > 8 else series
    
    xy = [(i, v) for i, v in enumerate(recent_series) if v is not None and v > 0]
    n = len(xy)
    
    if n == 0:
        return [0.0] * k
    if n == 1:
        return [xy[0][1]] * k

    sx = sum(x for x, _ in xy)
    sy = sum(y for _, y in xy)
    sxx = sum(x * x for x, _ in xy)
    sxy = sum(x * y for x, y in xy)

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-10:
        return [recent_series[-1]] * k
    
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n

    start_idx = len(series)
    preds = []
    for i in range(start_idx, start_idx + k):
        pred = a * i + b
        pred = max(pred, 0.0)
        preds.append(pred)
    
    return preds


def lambda_handler(event, context):
    # ✅ FIX: Get athlete_id FIRST
    qs = event.get("queryStringParameters") or {}
    athlete_id = qs.get("athlete_id") or os.environ.get("DEFAULT_ATHLETE_ID")
    weeks_back = int(os.environ.get("WEEKS_BACK", "12"))
    forecast_weeks = int(os.environ.get("FORECAST_WEEKS", "4"))

    if not athlete_id:
        return {
            "statusCode": 400,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": "athlete_id is required"}),
        }

    # ✅ FIX: NOW check if CSV user
    is_csv_user = str(athlete_id).startswith('csv_')

    # Determine which table to use
    if is_csv_user:
        processed_table_name = os.environ.get("PROCESSED_TABLE_CSV", "strava_processed_csv")
        print(f"✓ Using CSV table for {athlete_id}")
    else:
        processed_table_name = os.environ.get("PROCESSED_TABLE", "strava_processed_pau")
        print(f"✓ Using OAuth table for {athlete_id}")
    
    processed_table = dynamodb.Table(processed_table_name)

    print(f"Calculating trends for athlete {athlete_id}")
    
    now = datetime.now(timezone.utc)
    start_window = now - timedelta(weeks=weeks_back)

    # Scan processed table
    items = []
    try:
        # Handle both string and int athlete IDs
        if is_csv_user:
            scan_kwargs = {
                'FilterExpression': 'athlete_id = :aid',
                'ExpressionAttributeValues': {':aid': athlete_id}
            }
        else:
            scan_kwargs = {
                'FilterExpression': 'athlete_id = :aid',
                'ExpressionAttributeValues': {':aid': int(athlete_id)}
            }
        
        resp = processed_table.scan(**scan_kwargs)
        items.extend(resp.get("Items", []))
        
        while "LastEvaluatedKey" in resp:
            scan_kwargs['ExclusiveStartKey'] = resp["LastEvaluatedKey"]
            resp = processed_table.scan(**scan_kwargs)
            items.extend(resp.get("Items", []))
            
        print(f"Found {len(items)} activities")
        
    except Exception as e:
        print(f"Error scanning table: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": "Failed to load data"})
        }

    weekly = defaultdict(
        lambda: {
            "distance_km": 0.0,
            "time_s": 0.0,
            "hr_sum": 0.0,
            "hr_time": 0.0,
            "elev_m": 0.0,
        }
    )

    for it in items:
        dt = _parse_date(it)
        if not dt or dt < start_window:
            continue

        week_start = _floor_to_monday(dt).date().isoformat()
        w = weekly[week_start]

        # Distance
        if "distance_km" in it:
            dist_km = float(it["distance_km"])
        else:
            dist_m = float(it.get("distance", 0.0))
            dist_km = dist_m / 1000.0

        time_s = float(it.get("moving_time", 0.0))
        elev_m = float(it.get("total_elevation_gain", 0.0))

        w["distance_km"] += dist_km
        w["time_s"] += time_s
        w["elev_m"] += elev_m

        # HR
        avg_hr = it.get("average_heartrate")
        if avg_hr is not None:
            avg_hr = float(avg_hr)
            if avg_hr > 0 and time_s > 0:
                w["hr_sum"] += avg_hr * time_s
                w["hr_time"] += time_s

    # Sort weeks chronologically
    week_keys = sorted(weekly.keys())
    
    if not week_keys:
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "athlete_id": athlete_id,
                "weeks": [],
                "forecast_weeks": [],
                "distance": {"historical": [], "forecast": []},
                "pace": {"historical": [], "forecast": []},
                "heartrate": {"historical": [], "forecast": []},
                "elevation": {"historical": [], "forecast": []}
            })
        }

    distance_hist = []
    pace_hist = []
    hr_hist = []
    elev_hist = []

    for wk in week_keys:
        data = weekly[wk]
        dist_km = data["distance_km"]
        time_s = data["time_s"]

        distance_hist.append(dist_km if dist_km > 0 else None)

        # Pace (seconds per km)
        if dist_km > 0 and time_s > 0:
            pace_hist.append(time_s / dist_km)
        else:
            pace_hist.append(None)

        # HR
        if data["hr_time"] > 0:
            hr_hist.append(data["hr_sum"] / data["hr_time"])
        else:
            hr_hist.append(None)

        elev_hist.append(data["elev_m"] if data["elev_m"] > 0 else None)

    # Generate forecasts
    print("Generating forecasts...")
    distance_forecast = _linear_forecast(distance_hist, forecast_weeks)
    pace_forecast = _linear_forecast(pace_hist, forecast_weeks)
    hr_forecast = _linear_forecast(hr_hist, forecast_weeks)
    elev_forecast = _linear_forecast(elev_hist, forecast_weeks)

    # Forecast week labels
    forecast_week_labels = []
    if week_keys:
        last_week = datetime.fromisoformat(week_keys[-1])
        for i in range(1, forecast_weeks + 1):
            forecast_week_labels.append(
                (last_week + timedelta(days=7 * i)).date().isoformat()
            )

    # Interpolate historical data
    distance_hist = _interpolate_nulls(distance_hist)
    pace_hist = _interpolate_nulls(pace_hist)
    hr_hist = _interpolate_nulls(hr_hist)
    elev_hist = _interpolate_nulls(elev_hist)

    result = {
        "athlete_id": athlete_id,
        "weeks": week_keys,
        "forecast_weeks": forecast_week_labels,
        "distance": {
            "historical": distance_hist,
            "forecast": distance_forecast,
        },
        "pace": {
            "historical": pace_hist,
            "forecast": pace_forecast,
        },
        "heartrate": {
            "historical": hr_hist,
            "forecast": hr_forecast,
        },
        "elevation": {
            "historical": elev_hist,
            "forecast": elev_forecast,
        },
    }

    result = _decimal_to_float(result)
    
    print(f"Trends calculated successfully")
    print(f"Historical weeks: {len(week_keys)}")
    print(f"Forecast weeks: {len(forecast_week_labels)}")

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(result),
    }