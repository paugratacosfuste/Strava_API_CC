import json
import os
import boto3
import time
import urllib.request
import urllib.parse

# Environment variables
CLIENT_ID = os.environ.get('STRAVA_CLIENT_ID', '184259') 
CLIENT_SECRET = os.environ.get('STRAVA_CLIENT_SECRET', '63241b77907cfa7fbbd0f93d0c815e981386ba07') 
REDIRECT_URL = os.environ.get('REDIRECT_URL', 'https://dd9ssrwk58hno.cloudfront.net') 
DYNAMO_TABLE_NAME = os.environ.get('DYNAMO_TABLE_NAME', 'strava_tokens_pau')

TOKEN_EXCHANGE_URL = "https://www.strava.com/oauth/token"

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb', region_name='eu-central-1') 
table = dynamodb.Table(DYNAMO_TABLE_NAME)

lambda_client = boto3.client('lambda', region_name='eu-central-1') 
FETCHER_LAMBDA_NAME = 'StravaDataFetcher_pau' 

def lambda_handler(event, context):
    """
    Handles the redirect from Strava. Exchanges the temporary code for tokens,
    saves to DynamoDB, triggers data fetcher, and redirects to dashboard.
    """
    
    # Get the temporary code from URL parameters
    try:
        query_params = event.get('queryStringParameters', {})
        auth_code = query_params.get('code')
        error = query_params.get('error')
        
        if error:
            print(f"Authorization Denied: {error}")
            return redirect_to_page(REDIRECT_URL, 'index.html', 'error', error)
        if not auth_code:
            print("Missing authorization code")
            return redirect_to_page(REDIRECT_URL, 'index.html', 'error', 'Missing_Authorization_Code')
            
    except Exception as e:
        print(f"Error parsing event: {e}")
        return redirect_to_page(REDIRECT_URL, 'index.html', 'error', 'Internal_Server_Error')

    # Exchange code for tokens
    payload = urllib.parse.urlencode({
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': auth_code,
        'grant_type': 'authorization_code' 
    }).encode('ascii')

    try:
        req = urllib.request.Request(TOKEN_EXCHANGE_URL, data=payload, method='POST')
        with urllib.request.urlopen(req) as response:
            token_data = json.loads(response.read().decode('utf-8'))
            
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        print(f"Strava Token Exchange Error {e.code}: {error_body}")
        return redirect_to_page(REDIRECT_URL, 'index.html', 'error', 'Token_Exchange_Failed')
            
    except Exception as e:
        print(f"Token Exchange Error: {e}")
        return redirect_to_page(REDIRECT_URL, 'index.html', 'error', 'Token_Exchange_Failed')

    # Extract athlete ID and tokens
    athlete_id = token_data.get('athlete', {}).get('id')
    refresh_token = token_data.get('refresh_token')
    
    if not athlete_id or not refresh_token:
        print("Missing athlete ID or refresh token")
        return redirect_to_page(REDIRECT_URL, 'index.html', 'error', 'Invalid_Token_Response')
        
    # Save to DynamoDB
    try:
        table.put_item(
            Item={
                'athlete_id': int(athlete_id), 
                'refresh_token': refresh_token,
                'created_at': int(time.time()),
                'expires_at': token_data.get('expires_at')
            }
        )
        print(f"Saved tokens for athlete {athlete_id}")
        
    except Exception as e:
        print(f"DynamoDB Save Failed: {e}")
        return redirect_to_page(REDIRECT_URL, 'index.html', 'error', 'Database_Save_Failed')

    # Trigger Data Fetcher Lambda (which will trigger the chain)
    try:
        fetcher_payload = json.dumps({'athlete_id': int(athlete_id)}) 
        
        lambda_client.invoke(
            FunctionName=FETCHER_LAMBDA_NAME,
            InvocationType='Event',  # Asynchronous
            Payload=fetcher_payload
        )
        print(f"Triggered {FETCHER_LAMBDA_NAME} for athlete {athlete_id}")
        
    except Exception as e:
        print(f"WARNING: Failed to trigger fetcher: {e}") 
        
    # CRITICAL: Redirect to loading/dashboard page, NOT landing page
    # The dashboard will show loading state until data is ready
    return redirect_to_page(REDIRECT_URL, 'input_page.html', 'athlete_id', str(athlete_id))

def redirect_to_page(base_url, page, param_name, param_value):
    """
    Creates redirect response to a specific page with parameters
    """
    if base_url.endswith('/'):
        base_url = base_url.rstrip('/')
        
    redirect_url = f"{base_url}/{page}?{param_name}={param_value}"
    
    return {
        'statusCode': 302,
        'headers': {
            'Location': redirect_url
        },
        'body': 'Redirecting to dashboard...'
    }