# StravaAnalysis
Code for some analysis

```python
payload = {
            'client_id': "",
            'client_secret': "",
            'refresh_token': "",
            'grant_type': "refresh_token",
            'f': 'json'}
strava_code = ""
```

# Setup Steps
Make a file called `strava_payload.py` with payload and code variables above. 

Go to this link and copy code from localhost url. Sub in your client id.

`https://www.strava.com/oauth/authorize?client_id=CLIENTID&redirect_uri=http://localhost&response_type=code&scope=read,activity:read_all`

Paste this code into the `strava_code` variable.

Run the `strava.py` file and use the data as you please.