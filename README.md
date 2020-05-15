# A-Comeback-Story
Predicting MLB comeback wins using Statcast data 

![Baseball Header](https://github.com/josephshanks/UFO-SIGHTINGS/blob/master/media/ufo-header.png)

<p align="center">
  <img src="https://img.shields.io/badge/Maintained%3F-IN PROG-blue?style=flat-square"></img>
  <img src="https://img.shields.io/github/commit-activity/m/josephshanks/Swing-and-a-miss?style=flat-square"></img>
</p>

| [Joseph Shanks](https://github.com/josephshanks) |
---|---|---|

 
## Table of Contents

- [Basic Overview](#basic-overview)
  - [Context](#context)
  - [New Age Data](#new-age-data)
- [Exploring Data](#exploring-data)
  - [Initial Intake](#initial-intake)
  - [Import and Data Cleaning](#import-and-data-cleaning)
- [Modeling](#modeling)
  - [Tokenizing](#tokenizing)
  - [Visualizations](#visualizations)
- [Future Considerations](#future-considerations)
- [License](#license)


## Basic Overview

### Context

<img align="right" src="https://i.pinimg.com/236x/32/47/16/324716a77ab7183025a1ad46786de375--x-files-funny-love-puns.jpg">

Baseball is America's past time, a game that has been played for centries and a league that was formed way back in 1869. The game was simple, use the stick, hit the ball. Besides keeping score of course, the first statistic for Major League Baseball was created in 1872, simply hits per game. Fast forward to the 1980s and baseball statistics really had not progressed much further than your basic statistics that you may be aware of, batting average, hits, home runs, total bases, ERA (earned run average) etc. Then the godfather Bill James comes along who started the 'Moneyball' era in baseball. His philosophy was rather than projecting what players might be better than others based off scours intuition, that teams should really be looking at data of players and which players get on base more often. In a nutshell, his philosophy was more people on base=more runs. Bill James philosophy was first adopted by Billy Beane and the Oakland Athletics. With James' philosophy the Athletics were able to be in the bottom 5 in total player payroll yet compete with teams like the yankees, who were almost tripple the amount of money spent towards their players. In fact in 2002, the A's had the most consecutive wins out of any team in MLB history being the first to implement this new age philosphy. The A's were expected to be one of the worst teams in baseball yet became one of the best with all odds stacked against them. That is the power of statistics.  ![THE NATIONAL UFO REPORTING CENTER](http://www.nuforc.org/). 

### New Age Data

Statcast has taken the power of statistics to a new level. In 2015 the MLB implemented tracking technology in every stadium around the league to gather previously immeasurable data from every single pitch of every single game. Some of the most impactful statistics are Exit Velocity of balls off the bat, launch angle, pitch rotation, and pitch velocity. 

> Stemming off of Bill James, I want to know if I can use machine learning models to predict what features best contribute to a pitchers success using Statcast data. To do this I want to know what features of a play contributed towards a batter having "hard contact" vs "soft contact." If a team is able to minimize the likelihood of hard contact balls throughout a game they theroretically would be able to minimize the amount of runs given up. This would allow for teams to scout for pitchers that met a specific criteria of pitching statistics to join their baseball team. 


## Exploring Data

SOURCE             | TIMEFRAME | N_RECORDS
:-------------------------:|:-------------------------:|:-------------------------:|
![Statcast](https://baseballsavant.mlb.com/statcast_search)  | 2019 MLB SEASON |  244,393

<img align="right" src="https://raw.githubusercontent.com/boogiedev/UFO-SIGHTINGS/master/media/nuforc.PNG"></img>

The data I used was from the most recent baseball season (2019) Below is a preview of the format that the data comes in from the website.

<br/>

<p align="center">
  <img src="https://raw.githubusercontent.com/boogiedev/UFO-SIGHTINGS/master/media/dataexcerpt.PNG"></img>
</p>



### Initial Intake

Shown below are just a few of the features that I was able to work with:
- `pitch_type`: The type of pitch derived from Statcast.
- `release_pos_x`: Horizontal Release Position of the ball measured in feet from the catcher's perspective.
- `release_pos_z`: Vertical Release Position of the ball measured in feet from the catcher's perspective.
- `zone`: Zone location of the ball when it crosses the plate from the catcher's perspective.
- `stand`: Side of the plate batter is standing.
- `p_throws`: Hand pitcher throws with.
- `balls`: Pre-pitch number of balls in count.
- `strikes`: Pre-pitch number of strikes in count.
- `pfx_x`: Horizontal movement in feet from the catcher's perspective.
- `pfx_z`: Vertical movement in feet from the catcher's perpsective.
- `plate_x`: Horizontal position of the ball when it crosses home plate from the catcher's perspective.
- `plate_z`: Vertical position of the ball when it crosses home plate from the catcher's perspective.
- `on_1b`: Pre-pitch MLB Player Id of Runner on 1B.
- `vx0`: The velocity of the pitch, in feet per second, in x-dimension, determined at y=50 feet.
- `vy0`: The velocity of the pitch, in feet per second, in y-dimension, determined at y=50 feet.
- `ax`: The acceleration of the pitch, in feet per second per second, in x-dimension, determined at y=50 feet.
- `ay`: The acceleration of the pitch, in feet per second per second, in y-dimension, determined at y=50 feet.
- `sz_top`: Top of the batter's strike zone set by the operator when the ball is halfway to the plate.
- `release_spin`: Spin rate of pitch tracked by Statcast.
- `release_extension`: Release extension of pitch in feet as tracked by Statcast.
- `if_fielding_alignment`: Infield fielding alignment at the time of the pitch.
<p align="center">
  <img src="https://raw.githubusercontent.com/boogiedev/UFO-SIGHTINGS/master/media/dfbefore.PNG"></img>
</p>


### Import and Data Cleaning

Due to the large size of the data I used an S3 container and EC2 instead through AWS to help compute my modeling. Below is how I was able to import my 21 seperate csv files, as well as adding in columns for contact quality and the type of pitch thrown. After importing the data I created a cleaning function to run with my new data frame, here I had to drop all data that had already told me the result of the batted ball. Again, I am trying to predict how hard a ball might be hit based off pitching statistics, if I have data in my model that already captures how hard a ball is hit or if the play resulted in a out or homerun it would negatively impact my model. 

```python
def load_pitch_data_from_s3(csv_files, number_of_rows=None, bucket='qualitycontact'):
    '''
    Function to take a list of loan data CSV files that stored in an AWS S3 bucket and load and
    concatenate them into one dataframe.

    Returns:
        DataFrame: Returns a dataframe containing all loans contained within the list of CSV files.  
    '''
    datadict={}
    pitch_data = []
    for filename in csv_files:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket, Key=filename)
        data = obj['Body'].read()
        f = BytesIO(data)
        data = pd.read_csv(f, low_memory=False, na_values='n/a', nrows=number_of_rows) 
        
        #Creating contact quality column and pitch type column
        datadict[filename]=data
        if filename in ['data/barrell BREAKING.csv','data/barrell FAST.csv','data/barrell OFFSPEED.csv','data/flare:burner BREAKING.csv','data/flare:burner FAST.csv','data/flare:burner OFFSPEED.csv', 'data/solid contact BREAKING.csv','data/solid contact OFFSPEED.csv','data/solid contact FAST.csv']:
            datadict[filename]['contact quality'] = 1
        else:
            datadict[filename]['contact quality'] = 0
            
        if filename in ['data/barrell FAST.csv','data/flare:burner FAST.csv','data/no contact FAST.csv','data/poor:top FAST.csv','data/poor:under FAST.csv','data/poor:weak FAST.csv','data/solid contact FAST.csv']:
            datadict[filename]['fast ball'] = 1
            datadict[filename]['offspeed'] = 0
            datadict[filename]['breaking'] = 0
        elif filename in ['data/barrell OFFSPEED.csv','data/flare:burner OFFSPEED.csv','data/no contact OFFSPEED.csv','data/poor:top OFFSPEED.csv','data/poor:under OFFSPEED.csv','data/poor:weak OFFSPEED.csv','data/solid contact OFFSPEED.csv']:
            datadict[filename]['fast ball'] = 0
            datadict[filename]['offspeed'] = 1
            datadict[filename]['breaking'] = 0
        else:
            datadict[filename]['fast ball'] = 0
            datadict[filename]['offspeed'] = 0
            datadict[filename]['breaking'] = 1
            
        pitch_data.append(datadict[filename])
    pitch = pd.concat(pitch_data)

    return pitch
    
    
    
def cleaning(df):
    #dropping columns that contains information of the result of the play. I am trying to model the contact quality only by factors known before the batter hits the ball
    df=df.drop(['pitch_type','game_date','player_name','pitcher','batter','events','description','spin_dir','spin_rate_deprecated','break_angle_deprecated',
         'break_length_deprecated','des','game_type','home_team','away_team','type','hit_location','bb_type','game_year','hc_x','hc_y',
         'tfs_deprecated','tfs_zulu_deprecated','umpire','sv_id','hit_distance_sc','launch_speed','launch_angle','game_pk','pitcher',
        'estimated_ba_using_speedangle','estimated_woba_using_speedangle','woba_value','woba_denom','babip_value','iso_value','pitch_name',
         'launch_speed_angle','home_score','away_score','post_away_score','post_home_score','post_bat_score','post_fld_score'],axis=1)
    
    df[['on_3b','on_2b','on_1b']] = df[['on_3b','on_2b','on_1b']].fillna(value=0)
    df[['if_fielding_alignment','of_fielding_alignment']] = df[['if_fielding_alignment','of_fielding_alignment']].fillna(value='Standard')
    df.dropna()
    
    #Was there anybody on third base?
    df['on_3b']=df['on_3b'].apply(lambda x: 1 if x >= 1 else 0)
    
    #Was there anybody on first and second base?
    df['on_2b']=df['on_2b'].apply(lambda x: 1 if x >= 1 else 0)
    df['on_1b']=df['on_1b'].apply(lambda x: 1 if x >= 1 else 0)
    
    #batter stance and pitcher stance: 1 for Right, 0 for Left
    df['stand']=df['stand'].apply(lambda x: 1 if x=='R' else 0)
    df['p_throws']=df['p_throws'].apply(lambda x: 1 if x=='R' else 0)
    df['inning_topbot']=df['inning_topbot'].apply(lambda x: 1 if x=='Bot' else 0)
    df['if_fielding_alignment']=df['if_fielding_alignment'].apply(lambda x: 0 if x=='Standard' else 1)
    df['of_fielding_alignment']=df['of_fielding_alignment'].apply(lambda x: 0 if x=='Standard' else 1)
    
    #drop nulls
    df=df.dropna()
    
    return df
```

Below is a snapshot of the new data frame.

<p align="center">
  <img src="https://raw.githubusercontent.com/boogiedev/UFO-SIGHTINGS/master/media/dfafter.PNG"></img>
</p>


---
## Modeling

### Tokenizing

After creating our stop words list and removing punctuations we tokenized our documents. “Tokenize” means creating “tokens” which are atomic units of the text. These tokens are words extracted by splitting the document.We then used the “SnowballStemmer” to stem our tokenized words. We decided to use the snowball stemmer over the WordNetLemmatizer or the PorterStemmer. The reason for this is show below. 

```python

porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

docs_porter = [[porter.stem(word) for word in words]
               for words in doc_filter]
docs_snowball = [[snowball.stem(word) for word in words]
                 for words in doc_filter]
docs_wordnet = [[wordnet.lemmatize(word) for word in words]
                for words in doc_filter]


## Print the stemmed and lemmatized words from the first document
print(“%16s | %16s | %16s | %16s |” % (“WORD”, “PORTER”, “SNOWBALL”, “LEMMATIZER”))
for i in range(min(len(docs_porter[0]), len(docs_snowball[0]), len(docs_wordnet[0]))):
  p, s, w = docs_porter[0][i], docs_snowball[0][i], docs_wordnet[0][i]
  if len(set((p, s, w))) != 1:
    print(“%16s | %16s | %16s | %16s |” % (doc_filter[0][i], p, s, w))
```
```
            WORD |           PORTER |         SNOWBALL |       LEMMATIZER |
         hovered |            hover |            hover |          hovered |
          looked |             look |             look |           looked |
      helicopter |         helicopt |         helicopt |       helicopter |
          stayed |             stay |             stay |           stayed |
     disappeared |        disappear |        disappear |      disappeared |
         appears |           appear |           appear |          appears |
              us |               us |               us |                u |
      consistent |          consist |          consist |       consistent |
        sighting |            sight |            sight |         sighting |
           venus |             venu |            venus |            venus |

```

We chose to stem the words with the Snowball Stemmer due to its preservation of important words for this usecase such as ‘venus’. The Snowball Stemmmer normalizes these words from its appeared form into their root form. We now have our list of clean tokens for each document! We turned this into a pandas Series to compute the TF-IDF


### Visualizations

Choropleth Map: 

> The choropleth map shown in the notebook 'choropleth_map.ipynb' shows the number of reports from each state in the time period, as well as the three most common words from those reports of each state.
With this information, we see a detailed image of where the reports are coming from and what the reports are talking about.
15:51

Observed UFO Shapes:

The bar chart shows what the most common shapes are in the reports. We can see that 'Circles' and 'Teardrops' are common shapes, as well as individuals reporting just seeing 'Light'.


<p align="center">
  <img src="https://github.com/boogiedev/UFO-SIGHTINGS/blob/master/media/observed_ufo_shapes.png?raw=true"></img>
</p>

Changes in Vocabulary Size Per Minimum Document Frequency:

<p align="center">
  <img src="https://raw.githubusercontent.com/boogiedev/UFO-SIGHTINGS/master/media/mindf.PNG"></img>
</p>

```
0.1 -- vocabulary (len=157): ['light', 'look', 'nuforc', 'helicopt', 'first', 'bright', 'pd', 'disappear', 'way', 'went', 'sight', 'report', 'seen', 'us', 'one', 'like', 'east', 'appear', 'note', 'hover', 'could', 'sky', 'provid', 'elect', 'stationari', 'star', 'inform', 'anonym', 'contact', 'remain', 'sourc', 'travel', 'notic', 'fli', 'someth', 'approxim', 'clear', 'see', 'would', 'wit', 'sound', 'come', 'direct', 'near', 'craft', 'saw', 'west', 'air', 'north', 'feet', 'object', 'tree', 'mayb', 'shape', 'side', 'view', 'size', 'orang', 'circl', 'never', 'hous', 'gone', 'pass', 'time', 'seem', 'move', 'low', 'almost', 'straight', 'white', 'plane', 'still', 'anoth', 'know', 'quick', 'toward', 'made', 'outsid', 'normal', 'stop', 'make', 'flash', 'mile', 'distanc', 'high', 'insid', 'chang', 'thought', 'go', 'minut', 'back', 'second', 'watch', 'show', 'around', 'two', 'ball', 'even', 'away', 'night', 'south', 'thing', 'came', 'point', 'color', 'end', 'green', 'complet', 'take', 'drive', 'reflect', 'window', 'line', 'nois', 'noth', 'ufo', 'blue', 'left', 'speed', 'red', 'behind', 'live', 'area', 'aircraft', 'get', 'slowli', 'head', 'flew', 'glow', 'across', 'right', 'slow', 'phone', 'fast', 'also', 'larg', 'home', 'cloud', 'big', 'tri', 'photo', 'indic', 'turn', 'video', 'three', 'got', 'eye', 'float', 'moon', 'face', 'street', 'later', 'front', 'observ', 'start', 'visibl', 'think']
0.2 -- vocabulary (len=54): ['light', 'look', 'nuforc', 'first', 'bright', 'pd', 'disappear', 'went', 'sight', 'report', 'seen', 'one', 'like', 'east', 'appear', 'note', 'could', 'sky', 'provid', 'elect', 'inform', 'anonym', 'contact', 'remain', 'sourc', 'notic', 'see', 'would', 'wit', 'sound', 'direct', 'craft', 'saw', 'west', 'object', 'shape', 'time', 'seem', 'move', 'white', 'plane', 'still', 'stop', 'thought', 'go', 'minut', 'back', 'second', 'watch', 'around', 'two', 'night', 'south', 'get']
0.3 -- vocabulary (len=15): ['light', 'look', 'nuforc', 'bright', 'pd', 'report', 'one', 'like', 'note', 'could', 'sky', 'see', 'saw', 'object', 'move']
0.4 -- vocabulary (len=10): ['light', 'look', 'nuforc', 'pd', 'like', 'note', 'sky', 'saw', 'object', 'move']
0.5 -- vocabulary (len=5): ['light', 'look', 'sky', 'saw', 'move']
0.6 -- vocabulary (len=1): ['light']
0.7 -- vocabulary (len=0): []
0.8 -- vocabulary (len=0): []
0.9 -- vocabulary (len=0): []
1.0 -- vocabulary (len=0): []

```

Document Frequencies at Minimum of 0.5

```python
# See words with a high frequency threshhold 50%
thresh = 0.5
for word, freq in doc_freq.items():
    if freq >= thresh:
        print(f"{word}:  {freq}")

```
```
light:  0.6767676767676768
look:  0.5252525252525253
sky:  0.5656565656565656
saw:  0.5151515151515151
move:  0.5858585858585859
```


## Future Considerations

Using NaieveBayes to test comminalities of words used to derive if these occurences are related.


Do the U.F.O. sightings have a similar distribution of reports from states?


## License
[MIT ©](https://choosealicense.com/licenses/mit/)
