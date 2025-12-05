import json
import os
import boto3
from decimal import Decimal
from boto3.dynamodb.conditions import Key

REGION_NAME = os.environ.get("REGION_NAME", "eu-central-1")
dynamodb = boto3.resource("dynamodb", region_name=REGION_NAME)

def decimal_to_float(obj):
    """Convert Decimal objects to float for JSON serialization"""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decimal_to_float(x) for x in obj]
    return obj

def format_athlete_name(athlete_id):
    """
    Format athlete name for display:
    - 156086407 → "pau_gratacos" (your OAuth ID)
    - csv_mati → "mati" (CSV users - remove prefix)
    - Other numeric IDs → keep as is
    """
    athlete_id_str = str(athlete_id)
    
    # Your OAuth ID - display as pau_gratacos
    if athlete_id_str == "156086407":
        return "pau_gratacos"
    
    # CSV users - remove "csv_" prefix
    elif athlete_id_str.startswith("csv_"):
        return athlete_id_str[4:]  # Remove "csv_" prefix (e.g., csv_mati → mati)
    
    # Other OAuth users - keep numeric ID
    else:
        return athlete_id_str

def lambda_handler(event, context):
    """
    Get top 5 users for each distance from BOTH pau and csv tables.
    Combines OAuth users (from strava_best_efforts_pau) and CSV users (from strava_best_efforts_csv).
    Returns competitive leaderboard with human-readable names.
    """
    
    # Table names
    best_efforts_pau_table = dynamodb.Table("strava_best_efforts_pau")
    best_efforts_csv_table = dynamodb.Table("strava_best_efforts_csv")
    
    # Standard distances we track
    distances = [
        ("400m", ["400m"]),
        ("1K", ["1k", "1K", "1 km"]),
        ("5K", ["5k", "5K", "5 km"]),
        ("10K", ["10k", "10K", "10 km"]),
        ("Half Marathon", ["Half-Marathon", "Half Marathon"]),
        ("Marathon", ["Marathon"])
    ]
    
    leaderboard = {}
    
    for distance_label, effort_names in distances:
        print(f"Processing distance: {distance_label}")
        
        all_efforts = []
        
        # ========== FETCH FROM PAU TABLE (OAuth users) ==========
        try:
            print(f"  Scanning strava_best_efforts_pau...")
            response_pau = best_efforts_pau_table.scan()
            items_pau = response_pau.get('Items', [])
            
            # Continue scanning if there are more pages
            while 'LastEvaluatedKey' in response_pau:
                response_pau = best_efforts_pau_table.scan(
                    ExclusiveStartKey=response_pau['LastEvaluatedKey']
                )
                items_pau.extend(response_pau.get('Items', []))
            
            print(f"  Found {len(items_pau)} total items in pau table")
            
            # Filter for this distance
            for item in items_pau:
                effort_name = item.get('effort_name')
                if effort_name in effort_names:
                    athlete_id = item.get('athlete_id')
                    elapsed_time = item.get('elapsed_time')
                    distance_meters = item.get('distance', 0)
                    
                    if elapsed_time and distance_meters:
                        # Convert Decimal to float BEFORE math operations
                        elapsed_time = float(elapsed_time)
                        distance_meters = float(distance_meters)
                        
                        # Calculate pace
                        pace_min_per_km = (elapsed_time / 60.0) / (distance_meters / 1000.0)
                        
                        all_efforts.append({
                            'athlete_id': athlete_id,
                            'athlete_name': format_athlete_name(athlete_id),
                            'time_seconds': elapsed_time,
                            'pace_min_per_km': round(pace_min_per_km, 2),
                            'date': item.get('start_date_local', 'Unknown'),
                            'source': 'oauth'
                        })
            
            print(f"  Added {len([e for e in all_efforts if e['source'] == 'oauth'])} efforts from pau table")
            
        except Exception as e:
            print(f"  Error scanning pau table for {distance_label}: {e}")
        
        # ========== FETCH FROM CSV TABLE (CSV users) ==========
        try:
            print(f"  Scanning strava_best_efforts_csv...")
            response_csv = best_efforts_csv_table.scan()
            items_csv = response_csv.get('Items', [])
            
            # Continue scanning if there are more pages
            while 'LastEvaluatedKey' in response_csv:
                response_csv = best_efforts_csv_table.scan(
                    ExclusiveStartKey=response_csv['LastEvaluatedKey']
                )
                items_csv.extend(response_csv.get('Items', []))
            
            print(f"  Found {len(items_csv)} total items in csv table")
            
            # Filter for this distance
            for item in items_csv:
                effort_name = item.get('effort_name')
                if effort_name in effort_names:
                    athlete_id = item.get('athlete_id')
                    elapsed_time = item.get('elapsed_time')
                    distance_meters = item.get('distance', 0)
                    
                    if elapsed_time and distance_meters:
                        # Convert Decimal to float BEFORE math operations
                        elapsed_time = float(elapsed_time)
                        distance_meters = float(distance_meters)
                        
                        # Calculate pace
                        pace_min_per_km = (elapsed_time / 60.0) / (distance_meters / 1000.0)
                        
                        all_efforts.append({
                            'athlete_id': athlete_id,
                            'athlete_name': format_athlete_name(athlete_id),
                            'time_seconds': elapsed_time,
                            'pace_min_per_km': round(pace_min_per_km, 2),
                            'date': item.get('start_date_local', 'Unknown'),
                            'source': 'csv'
                        })
            
            print(f"  Added {len([e for e in all_efforts if e['source'] == 'csv'])} efforts from csv table")
            
        except Exception as e:
            print(f"  Error scanning csv table for {distance_label}: {e}")
        
        # ========== COMBINE AND SORT ==========
        print(f"  Total efforts for {distance_label}: {len(all_efforts)}")
        
        # Sort by time (fastest first) and take top 5
        all_efforts.sort(key=lambda x: x['time_seconds'])
        top_5 = all_efforts[:5]
        
        # Format times for display
        for rank, effort in enumerate(top_5, start=1):
            seconds = effort['time_seconds']
            
            # For short distances (< 5K), show mm:ss
            if distance_label in ['400m', '1K']:
                minutes = int(seconds // 60)
                secs = int(seconds % 60)
                effort['time_formatted'] = f"{minutes}:{secs:02d}"
            else:
                # For longer distances, show hh:mm:ss
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                effort['time_formatted'] = f"{hours}:{minutes:02d}:{secs:02d}"
            
            # Format pace
            pace = effort['pace_min_per_km']
            pace_min = int(pace)
            pace_sec = int((pace - pace_min) * 60)
            effort['pace_formatted'] = f"{pace_min}:{pace_sec:02d}"
            
            # Add rank
            effort['rank'] = rank
        
        print(f"  Top 5 for {distance_label}:")
        for effort in top_5:
            print(f"    {effort['rank']}. {effort['athlete_name']} - {effort['time_formatted']} ({effort['source']})")
        
        leaderboard[distance_label] = top_5
    
    # Convert Decimal to float for JSON
    result = decimal_to_float(leaderboard)
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET,OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        },
        'body': json.dumps(result)
    }