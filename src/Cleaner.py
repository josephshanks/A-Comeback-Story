import numpy as np
import pandas as pd


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