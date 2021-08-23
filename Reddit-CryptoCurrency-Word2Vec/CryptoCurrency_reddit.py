#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 09:54:02 2021

@author: lockiemichalski
"""
from IPython import display
import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', context='talk', palette='Dark2')
import praw
from psaw import PushshiftAPI
import datetime as dt

################################################################################
'''Connect to CryptoCurrency reddit forum, and retrieve all comments and 
repllies from the subreddit, Daily Discussion, back until it began in April 2019'''
################################################################################
'''Connect to reddit API'''

reddit = praw.Reddit(client_id='',
                     client_secret='',
                     user_agent='')

################################################################################
'''Use PSAW and PRAW to obtain historical submission id and then retrieve comments'''
################################################################################

api=PushshiftAPI() #connect to PushShift api for historical sub reddit submission ids

def DailyDiscussion_ids(start_datetime=dt.datetime(2019, 4, 2),
                        before_datetime=dt.datetime(2019, 4, 3),
                        end_date = None
                        ):
    dd_dict = {} #daily discussion dictionary - key = title, value = id for comment retrieval
    dd_df = pd.DataFrame()
    
    '''Daily Discussion begins April 2, 2019 (GMT+0)'''
    start_time=int(start_datetime.timestamp())
    before_time=int(before_datetime.timestamp())
    #start_datetime = dt.datetime(2019,4,2) #start April 2, 2019
    #before_datetime = dt.datetime(2021,4,3) #begin before April 3, 2019
    
    if end_date == None:
        today_date = dt.datetime.today() #todays date to end the search
    else:
        today_date = end_date #defined end date
    
    '''API call to retrieve all submissions from subreddit on the specific period'''
    while before_datetime < today_date:
    
        submissions = list(api.search_submissions(after=start_time,before=before_time,
                                                 subreddit='CryptoCurrency',
                                                 filter=['url','id','author','title','subreddit'],
                                                 limit=2000))
        try:
            '''Filter only the Daily Discussion thread posted by AutoModerator'''
            dd_sub = pd.DataFrame(submissions)
            dd = dd_sub[(dd_sub['title'].str.contains("Daily Discussion -")) & (dd_sub['author'].str.contains('AutoModerator'))]
            dd_id = list(dd['id'])[0] #id to be used to retrieve comments from the day
            dd_title = list(dd['title'])[0] #submission title
            dd_dict[dd_title] = dd_id
            dd_df = dd_df.append(dd)
            print(dd_title)
            
        except IndexError:
            print("No Daily Discussion in API on {} ".format(str(start_datetime)))
        except KeyError:
            print("No threads retrieved in API on {}".format(str(start_datetime)))

        '''Increment datetime to utc for loop over all historical threads'''
        start_datetime += timedelta(days=1) #add 1 day to previous date
        before_datetime += timedelta(days=1) #add 1 day to previous date
        start_time = int(start_datetime.timestamp())
        before_time = int(before_datetime.timestamp())
        
    dd_df.reset_index(drop=True, inplace=True)
    return dd_dict, dd_df

################################################################################
'''Run function to retrieve submission ids'''
dicussion_dict, disc_df = DailyDiscussion_ids(end_date=dt.datetime(2020, 6, 1))
dicussion_dict_2, disc_df_2 = DailyDiscussion_ids(start_datetime=dt.datetime(2020, 6, 1),
                        before_datetime=dt.datetime(2020, 6, 2))

disc_df_comb = pd.concat([disc_df, disc_df_2])
disc_df_comb.to_csv('daily_discussion_historical_id.csv')
discussion_dict_comb = {}
discussion_dict_comb.update(dicussion_dict)
discussion_dict_comb.update(dicussion_dict_2)

################################################################################
''' Function to retrieve comments and replies from April 2019 and save each month as pickle file'''
def DailyDiscussion_comments(df_ids):
    
    id_list = list(df_ids['id']) #list of sub ids to loop through   
    submissions_dict = {} #dict where each key = idx representing a thread on given day, value = df of comments on 

    '''Loop through retrieved comment IDS to obtain historical comments and replies'''
    for idx, comment_id in enumerate(id_list):
        
        sub_dict = {} #sub dict to store and then append to submissions_dict 
        submission = reddit.submission(id=comment_id) #obtain the DD thread on certain day based on id
        print('API obtained submission index {}'.format(str(idx)))
        submission.comments.replace_more(limit=None) #flatten the comments, MoreComments and replies to list
        print('Replaced comments more for submission index {}'.format(str(idx)))

        '''Loop over all the comments in the daily discussion thread for a given day'''
        for index, comment in enumerate(submission.comments.list()):
            
            sub = comment.submission.title #title of subreddit thread
            sub_created = comment.submission.created_utc #utc date subreddit thread made
            sub_upvotes = comment.submission.score #subreddit thread upvotes
            sub_upvote_ratio = comment.submission.upvote_ratio #subreddit thread upvote ratio
            
            if str(comment.author) == 'Err0r__4o4_':
                print('Comment 404 error, next iteration')
                continue
        
            '''Create df of comments and replies, then add df to dictionary for i comments'''
            '''df of a comment in the subreddit thread'''
            try:
                df = pd.DataFrame({'comment': [comment.body],
                               'author':comment.author,
                               'comment_utc_created': comment.created_utc,
                               'comment_index_track': index,
                               'comment_upvotes': comment.score,
                               'author_link_karma': (comment.author).link_karma,
                               'author_comment_karma': (comment.author).comment_karma,
                               'author_has_verified_email': (comment.author).has_verified_email,
                               'author_is_gold': (comment.author).is_gold, 
                               'author_account_created': (comment.author).created_utc, 
                               'sub': sub,
                               'sub_created': sub_created,
                               'sub_upvotes': sub_upvotes,
                               'sub_upvote_ratio': sub_upvote_ratio
                               })
            except AttributeError:
                print('Comment deleted, next iteration')
                continue
            
            except:
                print('Error with comment author, next iteration')
                continue
            
            try:
                '''If comment has replies, add the replies to the same df, otherwise just comment'''
                if len(comment.replies) > 0:
                        for reply in comment.replies:
                            
                            df = df.append({'comment': [reply.body][0],
                                'author': reply.author,
                                'comment_utc_created': reply.created_utc,
                                'comment_upvotes': reply.score,
                                'comment_index_track': index,
                                'author_link_karma': (reply.author).link_karma,
                                'author_comment_karma': (reply.author).comment_karma,
                                'author_has_verified_email': (reply.author).has_verified_email,
                                'author_is_gold': (reply.author).is_gold, 
                                'author_account_created': (reply.author).created_utc, 
                                'sub': sub,
                                'sub_created': sub_created,
                                'sub_upvotes': sub_upvotes,
                                'sub_upvote_ratio': sub_upvote_ratio
                                }, ignore_index=True)  
            
            except AttributeError:
                print('Error with post, next iteration')
                continue

            except:
                print('Error with reply author, next iteration')
                continue
            
            sub_dict[index] = df #append df to sub dict
            print('comment {} of index {} appended to dict'.format(str(index),str(idx)))
        
        output = open(sub+'.pkl', 'wb') 
        pickle.dump(sub_dict, output)
        print('{} pickled DONE'.format(sub))
  
        submissions_dict[idx] = sub_dict #sub dict append to main dict to return
        print('{} DONE'.format(sub))
        
    return submissions_dict

################################################################################
               
#loop through submission posts after from April 2019
comm_dd = DailyDiscussion_comments(discussion_dict_comb)    




