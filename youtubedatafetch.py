import sys
import io
import pandas as pd
from googleapiclient.discovery import build
from datetime import datetime

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def is_complete_video(video_data):
    """Check if video has all required data (no missing critical fields)"""
    critical_fields = [
        'id', 'title', 'description', 'publishedAt', 'categoryId',
        'thumbnail_default', 'thumbnail_high', 'duration',
        'viewCount', 'likeCount', 'commentCount', 'privacyStatus'
    ]
    
    for field in critical_fields:
        value = video_data.get(field, '')
        if value == '' or value is None:
            return False
    return True

def get_channel_videos(api_key, channel_id, max_results=50):
    """Fetch only complete videos from YouTube channel (no missing data)"""
    
    youtube = build('youtube', 'v3', developerKey=api_key)
    all_videos = []
    skipped_count = 0
    
    # Get channel info
    channel_response = youtube.channels().list(
        part='snippet,statistics,contentDetails',
        id=channel_id
    ).execute()
    
    if not channel_response.get('items'):
        print("Channel not found")
        return []
    
    channel_info = channel_response['items'][0]
    uploads_playlist_id = channel_info['contentDetails']['relatedPlaylists']['uploads']
    channel_snippet = channel_info.get('snippet', {})
    channel_stats = channel_info.get('statistics', {})
    
    print(f"Channel: {channel_snippet.get('title', 'Unknown')}")
    print(f"Fetching videos with complete data...")
    
    # Get video IDs from uploads playlist (fetch more to ensure we get enough complete ones)
    playlist_items = []
    next_page_token = None
    fetch_limit = max_results * 3  # Fetch 3x to account for incomplete videos
    
    while len(playlist_items) < fetch_limit:
        response = youtube.playlistItems().list(
            part='snippet',
            playlistId=uploads_playlist_id,
            maxResults=min(50, fetch_limit - len(playlist_items)),
            pageToken=next_page_token
        ).execute()
        
        playlist_items.extend(response['items'])
        next_page_token = response.get('nextPageToken')
        
        if not next_page_token:
            break
    
    video_ids = [item['snippet']['resourceId']['videoId'] for item in playlist_items]
    
    # Get detailed video data
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i+50]
        
        videos_response = youtube.videos().list(
            part='snippet,statistics,contentDetails,status',
            id=','.join(batch_ids)
        ).execute()
        
        for video in videos_response['items']:
            video_id = video.get('id', '')
            stats = video.get('statistics', {})
            snippet = video.get('snippet', {})
            content_details = video.get('contentDetails', {})
            status = video.get('status', {})
            thumbnails = snippet.get('thumbnails', {})
            channel_thumbnails = channel_snippet.get('thumbnails', {})
            
            # Build video data
            video_data = {
                'id': video_id,
                'title': snippet.get('title', ''),
                'description': snippet.get('description', ''),
                'publishedAt': snippet.get('publishedAt', ''),
                'tags': '|'.join(snippet.get('tags', [])) if snippet.get('tags') else '',
                'categoryId': snippet.get('categoryId', ''),
                'defaultLanguage': snippet.get('defaultLanguage', ''),
                'defaultAudioLanguage': snippet.get('defaultAudioLanguage', ''),
                'thumbnail_default': thumbnails.get('default', {}).get('url', ''),
                'thumbnail_high': thumbnails.get('high', {}).get('url', ''),
                'duration': content_details.get('duration', ''),
                'viewCount': stats.get('viewCount', ''),
                'likeCount': stats.get('likeCount', ''),
                'commentCount': stats.get('commentCount', ''),
                'privacyStatus': status.get('privacyStatus', ''),
                'channel_id': channel_id,
                'channel_title': channel_snippet.get('title', ''),
                'channel_description': channel_snippet.get('description', ''),
                'channel_country': channel_snippet.get('country', ''),
                'channel_thumbnail': channel_thumbnails.get('high', {}).get('url', ''),
                'channel_subscriberCount': channel_stats.get('subscriberCount', ''),
                'channel_videoCount': channel_stats.get('videoCount', '')
            }
            
            # Only add videos with complete data
            if is_complete_video(video_data):
                all_videos.append(video_data)
                if len(all_videos) >= max_results:
                    break
            else:
                skipped_count += 1
        
        if len(all_videos) >= max_results:
            break
    
    print(f"Successfully fetched {len(all_videos)} complete videos")
    print(f"Skipped {skipped_count} videos with missing data")
    return all_videos

# Configuration
API_KEY = "AIzaSyAi26lpIsNQroAiwZEZT6M_9SAGFRAgc6o"
CHANNEL_ID = "UCtmn-DsF4BhPug-ff9LiDAA"  # FeelFreeToLearn
MAX_RESULTS = 50

# Fetch videos
videos = get_channel_videos(API_KEY, CHANNEL_ID, MAX_RESULTS)

if videos:
    # Create DataFrame
    df = pd.DataFrame(videos)
    
    # Convert numeric columns to proper types
    numeric_cols = ['viewCount', 'likeCount', 'commentCount', 
                    'channel_subscriberCount', 'channel_videoCount']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Save to CSV
    filename = "FeelFreeToLearn.csv"
    df.to_csv(filename, index=False, encoding='utf-8')
    
    print(f"\n✅ Data saved to: {filename}")
    print(f"Total videos: {len(df)}")
    print(f"Total views: {df['viewCount'].sum():,}")
    print(f"Average views: {df['viewCount'].mean():,.0f}")
    
    # Verify no critical missing data
    critical_cols = ['id', 'title', 'description', 'duration', 'viewCount', 'likeCount', 'commentCount']
    print(f"\n✅ All videos have complete data in critical fields")
    for col in critical_cols:
        empty = (df[col] == '').sum() if df[col].dtype == 'object' else 0
        print(f"  {col}: {len(df) - empty}/{len(df)} complete")
else:
    print("❌ No videos found")