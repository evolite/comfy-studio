Put a track here as `<Song>.mp3` with a `<Song>.json` beat file next to it:

    {"start_time": 10.3, "beats": [0.116, 0.441, 0.766, ...]}

`beats` are the beat times in seconds, `start_time` is where the edit starts in the song. Set `DEFAULT_BEATS=audio/<Song>.json`
in `.env` to make it the house track; a project can pick another in Settings → Music.
