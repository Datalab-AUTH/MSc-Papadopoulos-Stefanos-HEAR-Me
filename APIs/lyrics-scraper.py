import pylyrics3
import azapi
API = azapi.AZlyrics('google', accuracy=0.5)

bon_iver_lyrics = pylyrics3.get_artist_lyrics('bon iver')
bon_iver_lyrics.keys()
bon_iver_lyrics['Skinny Love']

pylyrics3.get_song_lyrics('artist', 'song')

billie_eilish_albums = pylyrics3.get_artist_lyrics('billie eilish', albums=True)
billie_eilish_albums.keys()

pylyrics3.get_song_lyrics('gojira', 'stranded')