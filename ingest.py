import re
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

def get_video_id(youtube_url):
    """
    Extract video ID from a YouTube URL.
    
    Supports formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    
    Args:
        youtube_url (str): The YouTube URL
        
    Returns:
        str: The video ID, or None if not found
    """
    # Pattern for youtu.be short links
    short_pattern = r'youtu\.be/([^?&\s]+)'
    match = re.search(short_pattern, youtube_url)
    if match:
        return match.group(1)
    
    # Parse URL and check for watch?v= parameter
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname in ('youtube.com', 'www.youtube.com'):
        query_params = parse_qs(parsed_url.query)
        if 'v' in query_params:
            return query_params['v'][0]
    
    # Pattern for embed links
    embed_pattern = r'youtube\.com/embed/([^?&\s]+)'
    match = re.search(embed_pattern, youtube_url)
    if match:
        return match.group(1)
    
    return None

def fetch_transcript(url: str):
    ytt_api = YouTubeTranscriptApi()
    video_id = get_video_id(url)
    transcript = ytt_api.fetch(video_id)

    normalized = []
    for snippet in transcript.snippets:
        start = snippet.start
        duration = snippet.duration
        text = snippet.text

        normalized.append({
            "start": start,
            "end": start + duration,
            "text": text.strip()
        })

    return normalized