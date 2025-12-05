import json
import os
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

REGION_NAME = os.environ.get("REGION_NAME", "eu-central-1")
PROCESSED_TABLE_NAME = os.environ.get("PROCESSED_TABLE_NAME", "strava_processed_pau")
PROCESSED_TABLE_CSV = os.environ.get("PROCESSED_TABLE_CSV", "strava_processed_csv")

dynamodb = boto3.resource("dynamodb", region_name=REGION_NAME)


def decimal_to_float(value):
    """Recursively convert Decimal objects coming from DynamoDB to float."""
    if isinstance(value, list):
        return [decimal_to_float(v) for v in value]
    if isinstance(value, dict):
        return {k: decimal_to_float(v) for k, v in value.items()}
    if isinstance(value, Decimal):
        return float(value)
    return value


def _parse_date(item):
    """Robustly parse a Strava date field into a timezone-aware datetime."""
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
    """Return the Monday of the week of dt."""
    return dt - timedelta(days=dt.weekday())


def _load_processed_activities(athlete_id: str):
    """Load all processed activities for the given athlete_id."""
    is_csv_user = str(athlete_id).startswith("csv_")

    if is_csv_user:
        table_name = PROCESSED_TABLE_CSV
    else:
        table_name = PROCESSED_TABLE_NAME

    table = dynamodb.Table(table_name)

    items = []
    try:
        # Try integer athlete_id (OAuth users)
        if not is_csv_user:
            resp = table.query(
                KeyConditionExpression=Key("athlete_id").eq(int(athlete_id))
            )
        else:
            resp = table.query(
                KeyConditionExpression=Key("athlete_id").eq(athlete_id)
            )
    except Exception:
        # Fallback to string key
        resp = table.query(
            KeyConditionExpression=Key("athlete_id").eq(str(athlete_id))
        )

    items.extend(resp.get("Items", []))

    # Handle pagination
    while "LastEvaluatedKey" in resp:
        if not is_csv_user:
            resp = table.query(
                KeyConditionExpression=Key("athlete_id").eq(int(athlete_id)),
                ExclusiveStartKey=resp["LastEvaluatedKey"],
            )
        else:
            resp = table.query(
                KeyConditionExpression=Key("athlete_id").eq(athlete_id),
                ExclusiveStartKey=resp["LastEvaluatedKey"],
            )
        items.extend(resp.get("Items", []))

    return [decimal_to_float(i) for i in items]


def _aggregate_by_week(items):
    """
    Aggregate activities into weekly distance/time and
    count distinct running days per week.
    """
    weekly = defaultdict(lambda: {"distance_km": 0.0, "time_s": 0.0, "days": set()})

    for it in items:
        dt = _parse_date(it)
        if not dt:
            continue

        week_start = _floor_to_monday(dt).date().isoformat()

        dist_m = float(it.get("distance", 0.0) or 0.0)
        time_s = float(it.get("moving_time", 0.0) or 0.0)

        weekly[week_start]["distance_km"] += dist_m / 1000.0
        weekly[week_start]["time_s"] += time_s
        weekly[week_start]["days"].add(dt.date())

    week_keys = sorted(weekly.keys())
    return week_keys, weekly


def _compute_volume_context(week_keys, weekly):
    """
    Compute base week distance (last non-zero),
    chronic distance (up to last 6 weeks),
    acute distance (last week),
    acute/chronic ratio and average run days.
    """
    base_week_distance = 30.0  # fallback
    # last non-zero week distance
    for wk in reversed(week_keys):
        dist = weekly[wk]["distance_km"]
        if dist > 0:
            base_week_distance = dist
            break

    # chronic & run days over last up to 6 weeks
    last_weeks = week_keys[-6:]
    distances = [weekly[w]["distance_km"] for w in last_weeks]
    run_days_list = [len(weekly[w]["days"]) for w in last_weeks]

    chronic_distance = (
        sum(distances) / len(distances) if distances and sum(distances) > 0 else base_week_distance
    )

    last_week_key = week_keys[-1]
    acute_distance = weekly[last_week_key]["distance_km"]

    if chronic_distance > 0:
        acr = acute_distance / chronic_distance
    else:
        acr = 1.0

    avg_run_days = (
        sum(run_days_list) / len(run_days_list) if run_days_list else 4.0
    )

    return base_week_distance, chronic_distance, acute_distance, acr, avg_run_days


def _determine_progression_factor(acute_distance, chronic_distance, acr):
    """
    Decide how aggressively to progress based on acute/chronic ratio.
    """
    # No training last week -> gentle restart
    if acute_distance == 0 and chronic_distance > 0:
        return 0.6

    # If training much higher than usual -> deload a bit
    if acr > 1.3:
        return 0.9

    # If training much lower than usual -> can increase more
    if acr < 0.8:
        return 1.08

    # Normal case: modest progression
    return 1.03


def _choose_run_days(avg_run_days):
    """
    Determine how many days per week the athlete usually runs and
    choose which days will be planned sessions.
    We clamp between 3 and 5 run days.
    """
    run_days_target = int(round(avg_run_days))
    run_days_target = max(3, min(run_days_target, 5))

    # day_index: 0=Mon ... 6=Sun
    # We always keep Tue (1), Thu (3), Sat (5) as core days
    patterns = {
        3: {1, 3, 5},            # Tue, Thu, Sat
        4: {1, 3, 5, 6},         # + Sun (recovery)
        5: {1, 2, 3, 5, 6},      # + Wed (easy)
    }
    return run_days_target, patterns[run_days_target]


def _build_next_week_plan(week_keys, weekly, athlete_id: str):
    """
    Create a training plan for next week, adapting:
    - total distance using acute/chronic load
    - number of running days using historical pattern.
    """
    now = datetime.now(timezone.utc)
    today = now.date()
    current_monday = today - timedelta(days=today.weekday())
    plan_monday = current_monday + timedelta(days=7)

    (
        base_week_distance,
        chronic_distance,
        acute_distance,
        acr,
        avg_run_days,
    ) = _compute_volume_context(week_keys, weekly)

    # Progression factor from acute/chronic
    factor = _determine_progression_factor(acute_distance, chronic_distance, acr)

    # Target distance based on base week and factor
    raw_target = base_week_distance * factor
    # Safety bounds: not less than 50% and not more than 130% of base
    lower_bound = 0.5 * base_week_distance
    upper_bound = 1.3 * base_week_distance
    target_week_distance = max(lower_bound, min(raw_target, upper_bound))
    target_week_distance = round(target_week_distance, 1)

    # Decide how many run days we plan
    run_days_target, allowed_run_days = _choose_run_days(avg_run_days)

    # Session template (maximum-intensity version for 5 run days)
    template = [
        {
            "day_index": 0,
            "day_name": "Monday",
            "type": "Rest",
            "intensity": "Off",
            "ratio": 0.0,
            "description": "Full rest or very light mobility work.",
        },
        {
            "day_index": 1,
            "day_name": "Tuesday",
            "type": "Intervals",
            "intensity": "Z4",
            "ratio": 0.20,  # core quality day
            "description": "Warm-up + short intervals (e.g. 6×400m) with jog recoveries.",
        },
        {
            "day_index": 2,
            "day_name": "Wednesday",
            "type": "Easy Run",
            "intensity": "Z2",
            "ratio": 0.15,  # added when we allow 5 days
            "description": "Easy conversational run to build aerobic base.",
        },
        {
            "day_index": 3,
            "day_name": "Thursday",
            "type": "Tempo Run",
            "intensity": "Z3–Z4",
            "ratio": 0.20,
            "description": "Continuous tempo at comfortably hard pace.",
        },
        {
            "day_index": 4,
            "day_name": "Friday",
            "type": "Rest / Cross-training",
            "intensity": "Off/Z1",
            "ratio": 0.0,
            "description": "Rest day or low-impact cross-training (bike, swim, walking).",
        },
        {
            "day_index": 5,
            "day_name": "Saturday",
            "type": "Long Run",
            "intensity": "Z2",
            "ratio": 0.30,
            "description": "Long steady run at easy pace.",
        },
        {
            "day_index": 6,
            "day_name": "Sunday",
            "type": "Recovery Run",
            "intensity": "Z1–Z2",
            "ratio": 0.15,
            "description": "Very easy shake-out to recover from the long run.",
        },
    ]

    # Zero out ratios for days we are not planning as running days
    effective_ratios = []
    for s in template:
        if s["day_index"] in allowed_run_days and s["ratio"] > 0:
            effective_ratios.append(s["ratio"])
        else:
            effective_ratios.append(0.0)

    positive_ratios_sum = sum(effective_ratios) or 1.0  # avoid division by zero

    sessions = []
    for s, eff_ratio in zip(template, effective_ratios):
        day_date = plan_monday + timedelta(days=s["day_index"])
        if eff_ratio > 0:
            dist = round(target_week_distance * (eff_ratio / positive_ratios_sum), 1)
        else:
            dist = 0.0

        sessions.append(
            {
                "date": day_date.isoformat(),
                "day": s["day_name"],
                "type": s["type"],
                "intensity": s["intensity"],
                "planned_distance_km": dist,
                "description": s["description"],
            }
        )

    # Recent weeks (for context)
    recent_weeks = []
    for wk in week_keys[-4:]:
        recent_weeks.append(
            {
                "week_start": wk,
                "distance_km": round(weekly[wk]["distance_km"], 1),
                "time_hours": round(weekly[wk]["time_s"] / 3600.0, 1),
            }
        )

    # Extra debug info – frontend ignores this but useful for you
    debug_info = {
        "last_week_distance_km": round(acute_distance, 1),
        "chronic_distance_6w_km": round(chronic_distance, 1),
        "acute_chronic_ratio": round(acr, 2),
        "avg_run_days_per_week": round(avg_run_days, 2),
        "planned_run_days": run_days_target,
    }

    return {
        "athlete_id": athlete_id,
        "plan_week_start": plan_monday.isoformat(),
        "generated_at": now.isoformat(),
        "base_week_distance_km": round(base_week_distance, 1),
        "target_week_distance_km": target_week_distance,
        "recent_weeks": recent_weeks,
        "sessions": sessions,
        "debug": debug_info,
    }


def lambda_handler(event, context):
    # CORS preflight
    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
            "body": json.dumps({"message": "OK"}),
        }

    # Get athlete_id from query string or JSON body
    athlete_id = None
    if "queryStringParameters" in event and event["queryStringParameters"]:
        athlete_id = event["queryStringParameters"].get("athlete_id")

    if not athlete_id and event.get("body"):
        try:
            body = json.loads(event["body"])
            athlete_id = body.get("athlete_id")
        except Exception:
            pass

    if not athlete_id:
        return {
            "statusCode": 400,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": "Missing athlete_id"}),
        }

    try:
        items = _load_processed_activities(athlete_id)

        if not items:
            return {
                "statusCode": 404,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
                "body": json.dumps(
                    {"error": "No processed activities found for this athlete"}
                ),
            }

        week_keys, weekly = _aggregate_by_week(items)
        if not week_keys:
            return {
                "statusCode": 404,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
                "body": json.dumps(
                    {"error": "No weekly data available to build a plan"}
                ),
            }

        plan = _build_next_week_plan(week_keys, weekly, str(athlete_id))

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,OPTIONS",
            },
            "body": json.dumps(plan),
        }

    except Exception as e:
        print("Error building training plan:", e)
        import traceback

        traceback.print_exc()
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": "Internal error building training plan"}),
        }
