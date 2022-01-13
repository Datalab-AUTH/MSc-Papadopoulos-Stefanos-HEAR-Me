from googleapiclient.discovery import build
# from utilities.keys import youtube_key
from Emotion.EmotionAnalysis import predict_emotion

# DEVELOPER_KEY = youtube_key
# YOUTUBE_API_SERVICE_NAME = "youtube"
# YOUTUBE_API_VERSION = "v3"
#
# youtube_object = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
#                        developerKey=DEVELOPER_KEY)

def search_track(youtube_object, query):
    search_keyword = youtube_object.search().list(q=query, part="id, snippet",
                                                  maxResults=1).execute()
    results = search_keyword.get("items", [])

    if not results[0]['id'].get('videoId') is None:
        id = results[0]['id']['videoId']
    else:
        return 0, 0

    # STATISTICS
    res = youtube_object.videos().list(id=id, part='statistics').execute()
    video_stats = res['items'][0]['statistics']
    view_count = video_stats["viewCount"]

    request = youtube_object.commentThreads().list(
        part="snippet,replies",
        maxResults=10,
        videoId=id,
        textFormat='plainText',
        order='relevance'
    )
    try:
        response = request.execute()
    except:
        print("Not Found!")
        return 0, 0

    if response is None:
        return 0, view_count

    comments_collection = []

    # while response:
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments_collection.append(comment)

    V_emotion = predict_emotion(comments_collection, False)

    return V_emotion, view_count
