import json
import os
import boto3
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timedelta
from decimal import Decimal

# ------------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------------
CLIENT_ID = os.environ.get('STRAVA_CLIENT_ID')
CLIENT_SECRET = os.environ.get('STRAVA_CLIENT_SECRET')
TOKENS_TABLE_NAME = os.environ.get('TOKENS_TABLE_NAME')
ACTIVITIES_TABLE_NAME = os.environ.get('ACTIVITIES_TABLE_NAME')
BEST_EFFORTS_TABLE_NAME = os.environ.get('BEST_EFFORTS_TABLE_NAME', 'strava_best_efforts_pau')

# Strava API endpoints
TOKEN_URL = "https://www.strava.com/oauth/token"
ACTIVITIES_URL = "https://www.strava.com/api/v3/athlete/activities"
ACTIVITY_DETAIL_URL = "https://www.strava.com/api/v3/activities/{id}"

# AWS clients/resources
REGION = "eu-central-1"
dynamodb = boto3.resource('dynamodb', region_name=REGION)
tokens_table = dynamodb.Table(TOKENS_TABLE_NAME)
activities_table = dynamodb.Table(ACTIVITIES_TABLE_NAME)
best_efforts_table = dynamodb.Table(BEST_EFFORTS_TABLE_NAME)

lambda_client = boto3.client('lambda', region_name=REGION)
PROCESSOR_LAMBDA_NAME = 'StravaDataProcessor_pau'


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def get_access_token(refresh_token: str) -> str:
    """
    Exchange refresh token for a fresh access token.
    """
    payload = urllib.parse.urlencode({
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'refresh_token': refresh_token,
        'grant_type': 'refresh_token'
    }).encode('ascii')

    req = urllib.request.Request(TOKEN_URL, data=payload, method='POST')

    with urllib.request.urlopen(req) as response:
        token_data = json.loads(response.read().decode('utf-8'))

    return token_data['access_token']


def fetch_all_activities(access_token: str, after_timestamp: int):
    """
    Fetch all activities from Strava API (handles pagination).
    """
    all_activities = []
    page = 1
    per_page = 200  # Max allowed by Strava

    while True:
        params = urllib.parse.urlencode({
            'after': after_timestamp,
            'per_page': per_page,
            'page': page
        })

        url = f"{ACTIVITIES_URL}?{params}"
        req = urllib.request.Request(url)
        req.add_header('Authorization', f'Bearer {access_token}')

        with urllib.request.urlopen(req) as response:
            activities = json.loads(response.read().decode('utf-8'))

        if not activities:
            break

        all_activities.extend(activities)
        print(f"Fetched page {page}: {len(activities)} activities")

        if len(activities) < per_page:
            break

        page += 1

    return all_activities


def fetch_activity_details(access_token: str, activity_id: int) -> dict:
    """
    Fetch a single activity with full details (including best_efforts).
    """
    url = ACTIVITY_DETAIL_URL.format(id=activity_id)
    req = urllib.request.Request(url)
    req.add_header('Authorization', f'Bearer {access_token}')

    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))

    return data


def convert_to_dynamodb_format(obj):
    """
    Recursively convert Python objects to DynamoDB-compatible format.
    Converts all floats to Decimal, handles nested structures.
    """
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: convert_to_dynamodb_format(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dynamodb_format(item) for item in obj]
    else:
        return obj


def save_activities_to_dynamodb(athlete_id, activities):
    """
    Save activities to DynamoDB table with bulletproof type conversion.
    """
    saved_count = 0
    for activity in activities:
        try:
            converted_activity = convert_to_dynamodb_format(activity)

            item = {
                'athlete_id': int(athlete_id),
                'activity_id': int(converted_activity['id']),
                'activity_name': converted_activity.get('name', 'Unnamed'),
                'activity_type': converted_activity.get('type', 'Run'),
                'sport_type': converted_activity.get('sport_type', 'Run'),
                'start_date': converted_activity.get('start_date', ''),
                'start_date_local': converted_activity.get('start_date_local', ''),
                'timezone': converted_activity.get('timezone', ''),
                'distance': converted_activity.get('distance', Decimal('0')),
                'moving_time': int(converted_activity.get('moving_time', 0)),
                'elapsed_time': int(converted_activity.get('elapsed_time', 0)),
                'total_elevation_gain': converted_activity.get('total_elevation_gain', Decimal('0')),
                'average_speed': converted_activity.get('average_speed', Decimal('0')),
                'max_speed': converted_activity.get('max_speed', Decimal('0')),
                'average_heartrate': converted_activity.get('average_heartrate'),
                'max_heartrate': converted_activity.get('max_heartrate'),
                'elev_high': converted_activity.get('elev_high'),
                'elev_low': converted_activity.get('elev_low'),
                'start_latlng': converted_activity.get('start_latlng', []),
                'end_latlng': converted_activity.get('end_latlng', []),
                'achievement_count': int(converted_activity.get('achievement_count', 0)),
                'kudos_count': int(converted_activity.get('kudos_count', 0)),
                'suffer_score': converted_activity.get('suffer_score'),
                'average_cadence': converted_activity.get('average_cadence'),
                'average_watts': converted_activity.get('average_watts'),
                'weighted_average_watts': converted_activity.get('weighted_average_watts'),
                'kilojoules': converted_activity.get('kilojoules'),
                'device_watts': converted_activity.get('device_watts'),
                'has_heartrate': converted_activity.get('has_heartrate', False),
                'max_watts': converted_activity.get('max_watts'),
                'device_name': converted_activity.get('device_name', ''),
                'workout_type': converted_activity.get('workout_type')
            }

            # Remove None / empty strings to reduce size
            item = {k: v for k, v in item.items() if v is not None and v != ''}

            activities_table.put_item(Item=item)
            saved_count += 1

        except Exception as e:
            print(f"Error saving activity {activity.get('id')}: {e}")
            continue

    return saved_count


def save_best_efforts_to_dynamodb(athlete_id, access_token, activities):
    """
    For each running activity, fetch full details and store best_efforts
    in a dedicated table.
    Table schema (recommended):
      - Partition key: athlete_id (Number)
      - Sort key: effort_key (String) = "name#activity_id#distance"
    """
    saved_efforts = 0
    athlete_id_int = int(athlete_id)

    from decimal import Decimal as D

    with best_efforts_table.batch_writer() as batch:
        for activity in activities:
            act_id = activity.get('id')
            if not act_id:
                continue

            try:
                details = fetch_activity_details(access_token, act_id)
                best_efforts = details.get('best_efforts', [])
                if not best_efforts:
                    continue

                for effort in best_efforts:
                    name = effort.get('name')
                    elapsed_time = effort.get('elapsed_time')
                    distance = effort.get('distance')

                    if not name or elapsed_time is None or distance is None:
                        continue

                    effort_key = f"{name}#{act_id}#{int(distance)}"

                    item = {
                        'athlete_id': athlete_id_int,
                        'effort_key': effort_key,
                        'effort_name': name,
                        'distance': D(str(distance)),
                        'elapsed_time': int(elapsed_time),
                        'activity_id': int(act_id),
                        'start_date': details.get('start_date', ''),
                        'start_date_local': details.get('start_date_local', '')
                    }

                    # Remove empty fields
                    item = {k: v for k, v in item.items() if v is not None and v != ''}

                    batch.put_item(Item=item)
                    saved_efforts += 1

                print(f"Saved {len(best_efforts)} best efforts for activity {act_id}")

            except urllib.error.HTTPError as e:
                print(f"HTTP error for activity {act_id}: {e.code} {e.reason}")
                continue
            except Exception as e:
                print(f"Error fetching/saving best efforts for activity {act_id}: {e}")
                continue

    print(f"Total best efforts saved: {saved_efforts}")
    return saved_efforts


# ------------------------------------------------------------------
# Lambda handler
# ------------------------------------------------------------------

def lambda_handler(event, context):
    """
    Main handler: Fetches athlete's activities for the last 365 days,
    saves them to DynamoDB, and stores Strava best_efforts in a separate table.
    """
    athlete_id = event.get('athlete_id')
    if not athlete_id:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'athlete_id is required'})
        }

    print(f"Fetching activities for athlete: {athlete_id}")

    # 1) Get refresh token
    try:
        response = tokens_table.get_item(Key={'athlete_id': int(athlete_id)})
        if 'Item' not in response:
            return {
                'statusCode': 404,
                'body': json.dumps({'error': 'Athlete not found in database'})
            }

        refresh_token = response['Item']['refresh_token']
        print("Retrieved refresh token from DynamoDB")

    except Exception as e:
        print(f"Error retrieving tokens: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Database error retrieving tokens'})
        }

    # 2) Get access token
    try:
        access_token = get_access_token(refresh_token)
        print("Successfully obtained access token")
    except Exception as e:
        print(f"Error getting access token: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Failed to get access token'})
        }

    # 3) Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    after_timestamp = int(start_date.timestamp())
    print(f"Fetching activities from {start_date.date()} to {end_date.date()}")

    # 4) Fetch activities
    try:
        activities = fetch_all_activities(access_token, after_timestamp)
        print(f"Fetched {len(activities)} activities (all types)")
    except Exception as e:
        print(f"Error fetching activities: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Failed to fetch activities from Strava'})
        }

    # 5) Filter running activities & save
    running_activities = [a for a in activities if a.get('type') == 'Run']
    print(f"Found {len(running_activities)} running activities")

    try:
        saved_count = save_activities_to_dynamodb(athlete_id, running_activities)
        print(f"Saved {saved_count} running activities to DynamoDB")
    except Exception as e:
        print(f"Error saving activities: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Failed to save activities to database'})
        }

    # 6) Save best efforts for each run into separate table
    try:
        best_efforts_saved = save_best_efforts_to_dynamodb(athlete_id, access_token, running_activities)
    except Exception as e:
        print(f"WARNING: Failed to save best efforts: {e}")
        best_efforts_saved = 0

    # 7) Trigger processor lambda (asynchronous)
    try:
        processor_payload = json.dumps({'athlete_id': athlete_id})
        lambda_client.invoke(
            FunctionName=PROCESSOR_LAMBDA_NAME,
            InvocationType='Event',
            Payload=processor_payload
        )
        print(f"Successfully triggered {PROCESSOR_LAMBDA_NAME} for athlete {athlete_id}")
    except Exception as e:
        print(f"WARNING: Failed to trigger processor lambda: {e}")

    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Fetched & saved activities; best efforts stored; processing started.',
            'athlete_id': athlete_id,
            'total_activities': len(activities),
            'running_activities': len(running_activities),
            'saved_activities': saved_count,
            'saved_best_efforts': best_efforts_saved
        })
    }
