import json
import os
import boto3
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime, timezone
from boto3.dynamodb.conditions import Key

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

PROCESSED_TABLE_NAME = os.environ.get("PROCESSED_TABLE_NAME", "strava_processed_pau")
BEST_EFFORTS_TABLE_NAME = os.environ.get("BEST_EFFORTS_TABLE_NAME", "strava_best_efforts_pau")
REGION_NAME = os.environ.get("REGION_NAME", "eu-central-1")

dynamodb = boto3.resource("dynamodb", region_name=REGION_NAME)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decimal_to_float(x) for x in obj]
    return obj


def format_pace_from_decimal(pace_min_per_km):
    if pace_min_per_km is None or (isinstance(pace_min_per_km, float) and np.isnan(pace_min_per_km)):
        return "-"
    minutes = int(pace_min_per_km)
    seconds = int(round((pace_min_per_km - minutes) * 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}"


def format_time_from_seconds(total_seconds, distance_meters=None):
    if total_seconds is None or (isinstance(total_seconds, float) and np.isnan(total_seconds)):
        return "-"
    total_seconds = int(round(total_seconds))

    # For short efforts (400m, 1K) show mm:ss
    if distance_meters is not None and distance_meters < 5000:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}:{seconds:02d}"

    # Otherwise show hh:mm:ss
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def format_total_time_human(total_seconds):
    """Format long training time as Hh Mm"""
    if total_seconds is None or (isinstance(total_seconds, float) and np.isnan(total_seconds)):
        return "-"
    total_seconds = int(round(total_seconds))

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60

    return f"{hours}h {minutes}m"


def calculate_best_effort_approx(df, distance_meters):
    """Approximate best effort using whole-activity average pace"""
    if "distance" not in df.columns or "moving_time" not in df.columns:
        return None

    df_valid = df[
        (df["distance"].notna()) &
        (df["moving_time"].notna()) &
        (df["distance"] >= float(distance_meters))
    ].copy()

    if df_valid.empty:
        return None

    df_valid["sec_per_meter"] = df_valid["moving_time"] / df_valid["distance"]
    best_idx = df_valid["sec_per_meter"].idxmin()
    best_run = df_valid.loc[best_idx]

    best_time_seconds = best_run["sec_per_meter"] * distance_meters
    pace_min_per_km = (best_time_seconds / 60.0) / (distance_meters / 1000.0)

    time_str = format_time_from_seconds(best_time_seconds, distance_meters)
    pace_str = format_pace_from_decimal(pace_min_per_km)

    date_str = best_run.get("start_date_local", "Unknown")
    if isinstance(date_str, str):
        try:
            dt = pd.to_datetime(date_str)
            date_str = dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    return {
        "distance_meters": int(best_run.get("distance", distance_meters)),
        "time": time_str,
        "pace": pace_str,
        "date": date_str,
    }


def load_best_efforts_items(athlete_id, best_efforts_table):
    """Load best efforts from table - handles both int and string athlete IDs"""
    items = []
    try:
        # Try as integer first (OAuth users)
        try:
            resp = best_efforts_table.query(
                KeyConditionExpression=Key("athlete_id").eq(int(athlete_id))
            )
        except (ValueError, boto3.dynamodb.conditions.Key.exceptions):
            # If that fails, try as string (CSV users)
            resp = best_efforts_table.query(
                KeyConditionExpression=Key("athlete_id").eq(str(athlete_id))
            )
        
        items.extend(resp.get("Items", []))

        while "LastEvaluatedKey" in resp:
            try:
                resp = best_efforts_table.query(
                    KeyConditionExpression=Key("athlete_id").eq(int(athlete_id)),
                    ExclusiveStartKey=resp["LastEvaluatedKey"]
                )
            except (ValueError, boto3.dynamodb.conditions.Key.exceptions):
                resp = best_efforts_table.query(
                    KeyConditionExpression=Key("athlete_id").eq(str(athlete_id)),
                    ExclusiveStartKey=resp["LastEvaluatedKey"]
                )
            items.extend(resp.get("Items", []))

    except Exception as e:
        print(f"Error loading best efforts table: {e}")
        return []

    return [decimal_to_float(i) for i in items]


def best_efforts_from_strava_table(athlete_id, df_processed, best_efforts_table):
    """Build best efforts using Strava best_efforts table when available"""
    be_items = load_best_efforts_items(athlete_id, best_efforts_table)
    distances = [
        (400, "400m"),
        (1000, "1K"),
        (5000, "5K"),
        (10000, "10K"),
        (21097, "Half Marathon"),
        (42195, "Marathon"),
    ]

    if not be_items:
        print("No Strava best_efforts in table; using approximate method for all distances.")
        results = []
        for d_m, label in distances:
            approx = calculate_best_effort_approx(df_processed, d_m)
            if approx:
                approx["distance_label"] = label
            else:
                approx = {
                    "distance_label": label,
                    "distance_meters": d_m,
                    "time": "-",
                    "pace": "-",
                    "date": "-"
                }
            results.append(approx)
        return results

    label_name_map = {
        "400m": ["400m"],
        "1K": ["1k", "1K", "1 km"],
        "5K": ["5k", "5K", "5 km"],
        "10K": ["10k", "10K", "10 km"],
        "Half Marathon": ["Half-Marathon", "Half Marathon"],
        "Marathon": ["Marathon"],
    }

    best_by_label = {label: None for _, label in distances}

    for item in be_items:
        name = item.get("effort_name")
        elapsed = item.get("elapsed_time")
        distance = item.get("distance")
        act_id = item.get("activity_id")
        start_date_local = item.get("start_date_local", "Unknown")

        if name is None or elapsed is None or distance is None:
            continue

        for label, name_list in label_name_map.items():
            if name in name_list:
                current = best_by_label.get(label)
                if current is None or elapsed < current["elapsed_time"]:
                    best_by_label[label] = {
                        "elapsed_time": elapsed,
                        "distance_meters": int(distance),
                        "activity_id": act_id,
                        "start_date_local": start_date_local,
                    }

    results = []
    for d_m, label in distances:
        best = best_by_label.get(label)
        if best is not None:
            elapsed = best["elapsed_time"]
            dist = best["distance_meters"]
            time_str = format_time_from_seconds(elapsed, d_m)
            pace_min_per_km = (elapsed / 60.0) / (dist / 1000.0)
            pace_str = format_pace_from_decimal(pace_min_per_km)

            date_str = best.get("start_date_local", "Unknown")
            if isinstance(date_str, str):
                try:
                    dt = pd.to_datetime(date_str)
                    date_str = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass

            results.append({
                "distance_label": label,
                "distance_meters": dist,
                "time": time_str,
                "pace": pace_str,
                "date": date_str,
            })
            print(f"Strava best effort {label}: {time_str} @ {pace_str}/km")
        else:
            print(f"No Strava best effort for {label}; using approximate method.")
            approx = calculate_best_effort_approx(df_processed, d_m)
            if approx:
                approx["distance_label"] = label
            else:
                approx = {
                    "distance_label": label,
                    "distance_meters": d_m,
                    "time": "-",
                    "pace": "-",
                    "date": "-"
                }
            results.append(approx)

    return results


# ------------------------------------------------------------------
# Lambda handler
# ------------------------------------------------------------------

def lambda_handler(event, context):
    print("Stats & Best Efforts Lambda started")

    # ✅ FIX: Get athlete_id FIRST
    if "body" in event and isinstance(event["body"], str):
        try:
            body = json.loads(event["body"])
        except json.JSONDecodeError:
            body = {}
    else:
        body = event or {}

    athlete_id = body.get("athlete_id")

    if not athlete_id and "queryStringParameters" in event and event["queryStringParameters"]:
        athlete_id = event["queryStringParameters"].get("athlete_id")

    if not athlete_id:
        return {
            "statusCode": 400,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": "Missing athlete_id"}),
        }

    # ✅ FIX: NOW check if CSV user
    is_csv_user = str(athlete_id).startswith('csv_')

    # Determine which tables to use
    if is_csv_user:
        processed_table_name = os.environ.get("PROCESSED_TABLE_CSV", "strava_processed_csv")
        best_efforts_table_name = os.environ.get("BEST_EFFORTS_TABLE_CSV", "strava_best_efforts_csv")
        print(f"✓ Using CSV tables for {athlete_id}")
    else:
        processed_table_name = PROCESSED_TABLE_NAME
        best_efforts_table_name = BEST_EFFORTS_TABLE_NAME
        print(f"✓ Using OAuth tables for {athlete_id}")

    # Create table references
    processed_table = dynamodb.Table(processed_table_name)
    best_efforts_table = dynamodb.Table(best_efforts_table_name)

    print(f"Loading processed data for athlete {athlete_id}")

    try:
        # Query with appropriate type
        if is_csv_user:
            resp = processed_table.query(
                KeyConditionExpression=Key("athlete_id").eq(athlete_id)
            )
        else:
            resp = processed_table.query(
                KeyConditionExpression=Key("athlete_id").eq(int(athlete_id))
            )
        
        items = [decimal_to_float(i) for i in resp.get("Items", [])]
        df = pd.DataFrame(items)

        if df.empty:
            print("No rows in processed table for this athlete")
            return {
                "statusCode": 404,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
                "body": json.dumps({"error": "No data found for this athlete"}),
            }

        print(f"Loaded {len(df)} processed activities")

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": "Failed to load data"}),
        }

    # ---------- Statistics ----------
    stats = {}
    try:
        total_runs = len(df)

        if "distance" in df.columns:
            total_distance_km = float(df["distance"].sum() / 1000.0)
            total_distance_km = round(total_distance_km, 1)
        else:
            total_distance_km = 0.0

        if "moving_time" in df.columns:
            total_time_seconds = float(df["moving_time"].sum())
            total_time_hours = round(total_time_seconds / 3600.0, 1)
            total_time_formatted = format_total_time_human(total_time_seconds)
        else:
            total_time_seconds = 0.0
            total_time_hours = 0.0
            total_time_formatted = "-"

        total_elevation_m = int(df["total_elevation_gain"].sum()) if "total_elevation_gain" in df.columns else 0

        avg_hr = int(round(df["average_heartrate"].mean())) if "average_heartrate" in df.columns else 0

        avg_pace_decimal = df["pace_min_per_km"].mean() if "pace_min_per_km" in df.columns else np.nan
        avg_pace_str = format_pace_from_decimal(avg_pace_decimal)

        stats = {
            "total_runs": int(total_runs),
            "total_distance_km": total_distance_km,
            "total_time_hours": float(total_time_hours),
            "total_time_formatted": total_time_formatted,
            "total_elevation_m": total_elevation_m,
            "avg_heart_rate": int(avg_hr) if avg_hr else 0,
            "avg_pace": avg_pace_str,
        }

        print(f"Stats: {stats}")

    except Exception as e:
        print(f"Error calculating statistics: {e}")
        stats = {}

    # ---------- Best efforts ----------
    print("Building best efforts using Strava best_efforts table (with fallback)...")
    best_efforts = best_efforts_from_strava_table(athlete_id, df, best_efforts_table)

    result = {
        "athlete_id": athlete_id,
        "statistics": stats,
        "best_efforts": best_efforts,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

    print("Stats & Best Efforts calculation complete")

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,OPTIONS,POST",
        },
        "body": json.dumps(result, default=str),
    }