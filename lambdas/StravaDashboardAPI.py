import json
import os
import boto3
from decimal import Decimal

# Environment variables
PROCESSED_TABLE_NAME = os.environ.get('PROCESSED_TABLE_NAME', 'strava_processed_pau')
PREDICTIONS_TABLE_NAME = os.environ.get('PREDICTIONS_TABLE_NAME', 'strava_predictions_pau')
REGION_NAME = os.environ.get('REGION_NAME', 'eu-central-1')

# Initialize DynamoDB
dynamodb = boto3.resource('dynamodb', region_name=REGION_NAME)
processed_table = dynamodb.Table(PROCESSED_TABLE_NAME)
predictions_table = dynamodb.Table(PREDICTIONS_TABLE_NAME)

def decimal_default(obj):
    """JSON serializer for Decimal objects"""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

def lambda_handler(event, context):
    """
    API Gateway Lambda that returns dashboard data for a given athlete
    Supports CORS for browser access
    """
    
    # Handle CORS preflight
    if event.get('httpMethod') == 'OPTIONS':
        return cors_response(200, {'message': 'OK'})
    
    # Extract athlete_id from query parameters
    try:
        params = event.get('queryStringParameters', {})
        athlete_id = params.get('athlete_id')
        
        if not athlete_id:
            return cors_response(400, {'error': 'athlete_id query parameter is required'})
            
        athlete_id = int(athlete_id)
        
    except (ValueError, TypeError) as e:
        return cors_response(400, {'error': 'Invalid athlete_id format'})
    
    # Determine what data to return
    data_type = params.get('type', 'all')  # 'activities', 'predictions', or 'all'
    
    response_data = {}
    
    # Fetch processed activities
    if data_type in ['activities', 'all']:
        try:
            activities_response = processed_table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('athlete_id').eq(athlete_id)
            )
            activities = activities_response.get('Items', [])
            
            # Sort by date (most recent first)
            activities = sorted(
                activities, 
                key=lambda x: x.get('start_date_local', ''), 
                reverse=True
            )
            
            response_data['activities'] = activities
            response_data['activities_count'] = len(activities)
            
        except Exception as e:
            print(f"Error fetching activities: {e}")
            return cors_response(500, {'error': 'Failed to fetch activities data'})
    
    # Fetch predictions
    if data_type in ['predictions', 'all']:
        try:
            predictions_response = predictions_table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('athlete_id').eq(athlete_id)
            )
            predictions = predictions_response.get('Items', [])
            
            # Sort by timestamp (most recent first)
            predictions = sorted(
                predictions,
                key=lambda x: x.get('prediction_timestamp', ''),
                reverse=True
            )
            
            # Return only the most recent prediction
            response_data['latest_prediction'] = predictions[0] if predictions else None
            response_data['prediction_history'] = predictions[:10]  # Last 10 predictions
            
        except Exception as e:
            print(f"Error fetching predictions: {e}")
            return cors_response(500, {'error': 'Failed to fetch predictions data'})
    
    # Calculate summary statistics
    if 'activities' in response_data:
        try:
            response_data['summary'] = calculate_summary_stats(response_data['activities'])
        except Exception as e:
            print(f"Error calculating summary: {e}")
    
    return cors_response(200, response_data)


def calculate_summary_stats(activities):
    """Calculate summary statistics from activities"""
    
    if not activities:
        return {}
    
    total_distance = sum(float(a.get('distance', 0)) for a in activities)
    total_time = sum(float(a.get('moving_time', 0)) for a in activities)
    total_elevation = sum(float(a.get('total_elevation_gain', 0)) for a in activities)
    
    # Average metrics (only from activities that have the value)
    avg_hr_activities = [a for a in activities if a.get('average_heartrate')]
    avg_hr = sum(float(a['average_heartrate']) for a in avg_hr_activities) / len(avg_hr_activities) if avg_hr_activities else 0
    
    avg_pace_activities = [a for a in activities if a.get('pace_min_per_km')]
    avg_pace = sum(float(a['pace_min_per_km']) for a in avg_pace_activities) / len(avg_pace_activities) if avg_pace_activities else 0
    
    return {
        'total_activities': len(activities),
        'total_distance_km': round(total_distance / 1000, 2),
        'total_time_hours': round(total_time / 3600, 2),
        'total_elevation_m': round(total_elevation, 2),
        'avg_heartrate_bpm': round(avg_hr, 1),
        'avg_pace_min_per_km': round(avg_pace, 2),
        'avg_distance_per_run_km': round(total_distance / 1000 / len(activities), 2)
    }


def cors_response(status_code, body):
    """Create API Gateway response with CORS headers"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',  # In production, restrict this to your CloudFront domain
            'Access-Control-Allow-Methods': 'GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        },
        'body': json.dumps(body, default=decimal_default)
    }