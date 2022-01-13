def get_track_valence_arousal(sp, artist_name, track_name):
    result_track = sp.search(q=artist_name + ' ' + track_name,limit=1)
    try:
        if not result_track['tracks']['items'] == []:
            tid = result_track['tracks']['items'][0]['id']
            features = sp.audio_features(tid)
            valence = features[0]['valence']
            arousal = features[0]['energy']

            return valence, arousal

        else:
            print('Song Not Found! \n')
            return 0,0
    except:
        return 0,0