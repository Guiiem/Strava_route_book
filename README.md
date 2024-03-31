When I am in multi-day routes, I like to write a small 
description of the day and upload some pictures in my 
Strava activities. I wanted a way to get all of that and put
everything together into a small book. 

This small code does it in a bit of a sloppy way, but it is 
good enough for my purpose :). 

It uses:

1. Strava API to retrieve the information and the pictures.
2. Folium to generate and save a map.
3. Matplotlib to plot velocity and elevation. 
4. LaTex to compile a pdf document. 

A `.env` file that contains the API credentials is necessary. 

```
CLIENT_ID = "xxxxxx"
CLIENT_SECRET = "xxxxxxxxx"
REFRESH_TOKEN = "xxxxxxxxx"
``` 

`output_example.pdf` shows the result obtained from the code. 
The formatting and structure of the document is very minimal, but
should be enough to apply some minor tweaks manually and end up
with a good memory book :). 

Strava API is limited to 100 requests per 15 minutes. To generate books
with many activities (>33), it needs to be done in batches. The activity data is stored in a 
`{book_folder}/data` folder, so when the code is executed again 
it is loaded from there instead of make a new request. 


