import sys, json
import pandas as pd, numpy as np

from collections import Counter


def smart_round(n):
    return np.round(n, 1) if np.floor(n) > 9 else np.round(n, 2)


try:
    fn_in = sys.argv[1]
except IndexError:
    fn_in = input('Enter input tab delimited file name:')


df = pd.read_csv(fn_in, delimiter='\t', dtype=str).drop_duplicates('id_str', keep='last')

# Parsing and converting to native datetime64 data type
df.created_at = pd.to_datetime(df.created_at)

df = df[df.created_at.between(pd.Timestamp(2021,1,4,16,0,0), pd.Timestamp(2021,1,5,0,0,0,0))]


print('==> TSV read is done')


# Filling NaNs with zero values and converting them to int64 data type
fields = ['in_reply_to_user_id_str',
          'from_user_id_str',
          'in_reply_to_status_id_str',
          'user_followers_count',
          'user_friends_count',
          'geo_coordinates'
]

df[fields] = df[fields].fillna(0).astype(np.int64)

# Converting tweets id fields to int64 data type
df.id_str = df.id_str.astype(np.int64)


print('==> Data types conversion is done\n')


# Retweets only including bots
retweets_bots = df.text.str[:2] == 'RT'

# Top retweeters and tweeters including bots
top_rt_bots = df[ retweets_bots].from_user.value_counts().sort_values(ascending=False)
top_tw_bots = df[~retweets_bots].from_user.value_counts().sort_values(ascending=False)

print('Top retweeters including bots')
print(top_rt_bots[:20], '\n')
print('Top tweeters including bots')
print(top_tw_bots[:20], '\n')

# Bots filtering using a predefined bot list in a separate bots.txt file

try:
    with open('bots.txt', 'r', encoding='utf-8') as f:
        bots = '|'.join(
            list(
                filter(
                    lambda b: b != '',
                    f.read().split(sep='\n')
                )
            )
        )

        df_bots_size = df.id_str.size

        if bots:
            df = df[~df.from_user.str.contains(bots, case=False, regex=True)]

        df_bots_num = df_bots_size - df.id_str.size

        print(f'Bot filtering list: {bots}. {df_bots_num} records were removed\n')

except FileNotFoundError:
    print('==> Bot list file <bots.txt> not found\n')

# Retweets without bots
retweets = df.text.str[:2] == 'RT'

# Top retweeters and tweeters without bots
top_rt = df[ retweets].from_user.value_counts().sort_values(ascending=False)
top_tw = df[~retweets].from_user.value_counts().sort_values(ascending=False)

# Top 10 retweeters and tweeters
top_rt_10 = top_rt[:10]
top_tw_10 = top_tw[:10]

print('Top 10 retweeters')
print(top_rt_10, '\n')
print('Top 10 tweeters')
print(top_tw_10, '\n')

# % of retweets/tweets which came from the top 10 retweeters/tweeters
pct_top_rt_10 = np.round(np.sum(top_rt_10) * 100 / np.sum( retweets), 1)
pct_top_tw_10 = np.round(np.sum(top_tw_10) * 100 / np.sum(~retweets), 1)

print(f'Top 10 retweeters {pct_top_rt_10}%\n')
print(f'Top 10 tweeters {pct_top_tw_10}%\n')

# Who tweeted from top 10 retweeters and vice versa
top_rt_tw = df.from_user[np.logical_and(~retweets, df.from_user.isin(top_rt_10.index))].value_counts().sort_values(ascending=False)
top_tw_rt = df.from_user[np.logical_and( retweets, df.from_user.isin(top_tw_10.index))].value_counts().sort_values(ascending=False)

# Who was the bot in top tweeters/retweeters lists
bots_rt = top_rt_bots[:20][top_rt_bots.index.symmetric_difference(top_rt.index)].dropna().sort_values(ascending=False)
bots_tw = top_tw_bots[:20][top_tw_bots.index.symmetric_difference(top_tw.index)].dropna().sort_values(ascending=False)

print('These top', bots_rt.index.size,'retweeters are bots:', bots_rt.index.tolist())
print('These top', bots_tw.index.size,'tweeters are bots:', bots_tw.index.tolist(),'\n')

# Retweets and tweets amount and percentage
cnt_rt = np.sum( retweets)
cnt_tw = np.sum(~retweets)

pct_rt = np.round(cnt_rt * 100 / df.id_str.size, 1)
pct_tw = np.round(100 - pct_rt, 1)

print(f'Amount of retweets: {cnt_rt} ({pct_rt}%)')
print(f'Amount of tweets: {cnt_tw} ({pct_tw}%)')

# Unique users and their posting rate
unique_users = df.from_user.unique().size
unique_users_rate = np.round(df.id_str.size / unique_users, 1)

print('\nUnique users total:', unique_users, 'with rate', unique_users_rate, 'retweets or tweets per unique user')

# Unique retweeters and their posting rate
unique_users_rt = df[retweets].from_user.unique().size
unique_users_rt_rate = np.round(df[retweets].id_str.size / unique_users_rt, 1)

print('\nUnique retweeters total:', unique_users_rt, 'with rate', unique_users_rt_rate, 'retweets per unique retweeter')

# Unique tweeters and their posting rate
unique_users_tw = df[~retweets].from_user.unique().size
unique_users_tw_rate = np.round(df[~retweets].id_str.size / unique_users_tw, 1)

print('\nUnique tweeters total:', unique_users_tw, 'with rate', unique_users_tw_rate, 'tweets per unique tweeter')

# Top five retweets
top_retweets = df[retweets].text.value_counts().sort_values(ascending=False)[:10]

print('\nTop retweets\n')
print(top_retweets[:5])

# % are geocoded
geo_total = np.sum(df.geo_coordinates)
geo_coded = np.round(geo_total * 100 / df.id_str.size, 3)

print(f'\nGeocoded tweets and retweets: {geo_total} ({geo_coded}%)')

# % of profiles have a location
have_location_total = df.user_location.dropna().size
have_location = np.round(have_location_total * 100 / df.id_str.size, 1)

print(f'\nProfiles have a location: {have_location}%')

# Top 10 followers
top_followers = df.sort_values(by=['user_followers_count', 'user_friends_count'], ascending=False).drop_duplicates('from_user_id_str', keep='first')[['from_user', 'user_followers_count', 'user_friends_count']][:10]

print('\nTop 10 user followers')
print(top_followers)

# Hashtags and mentions

hashtags = Counter()
mentions = Counter()

for ent in df.entities_str:
    try:
        e = json.loads(ent)

        if e['hashtags']:
            for h in e['hashtags']:
                hashtags.update(['#' + h['text']])

        if e['user_mentions']:
            for m in e['user_mentions']:
                mentions.update(['@' + m['screen_name']])

    except TypeError:
        print(ent)

hashtags = dict(hashtags.most_common(20))
mentions = dict(mentions.most_common(20))

print('\nMost popular hashtags')
print(hashtags)
print('\nMost mentioned users')
print(mentions)

# Timeseries for all posts

ts = df.set_index(pd.DatetimeIndex(df.created_at)).sort_index(ascending=True)['id_str']

# Making 1 hour timeseries resample
ts_1hour = ts.resample('10min').count()

# Timeseries for retweets
df_ts_rt = df[retweets][['created_at', 'id_str']]
ts_rt = df_ts_rt.set_index(pd.DatetimeIndex(df_ts_rt.created_at)).sort_index(ascending=True)['id_str']

ts_rt_1hour = ts_rt.resample('10min').count()

del(df_ts_rt)

# Timeseries for tweets
df_ts_tw = df[~retweets][['created_at', 'id_str']]
ts_tw = df_ts_tw.set_index(pd.DatetimeIndex(df_ts_tw.created_at)).sort_index(ascending=True)['id_str']

ts_tw_1hour = ts_tw.resample('10min').count()

del(df_ts_tw)

print('\nTimeseries (10min)')
print(ts_1hour, '\n')

# Calculating users tweeting and retweeting this many times
vol_rt = dict()
vol_tw = dict()

for i in range(1, 10):
    vol_tw[str(i)] = int(np.sum(top_tw == i))
    vol_rt[str(i)] = int(np.sum(top_rt == i))

vol_tw['10+'] = int(np.sum(top_tw >= 10))
vol_rt['10+'] = int(np.sum(top_rt >= 10))

print('Tweeters volumes')
print(vol_tw, '\n')

print('Retweeters volumes')
print(vol_rt, '\n')

#
#   Generating results for charts.js plotting script
#

res = {
    'cnt_rt': int(cnt_rt),
    'cnt_tw': int(cnt_tw),
    'records_total': int(df.id_str.size),
    'records_removed': int(df_bots_num),
    # Timeseries for both tweets and retweets
    'ts1_chart': {
        'dtm': ts_1hour.index.strftime('%Y-%m-%d %H:%M').tolist(),
        'val': ts_1hour.values.tolist()
    },
    # Timeseries for retweets and tweets
    'ts2_chart': {
        'dtm': ts_rt_1hour.index.strftime('%Y-%m-%d %H:%M').tolist(),
        'val_rt': ts_rt_1hour.values.tolist(),
        'val_tw': ts_tw_1hour.values.tolist()
    },
    # top retweeter including bots
    'top_rt_bots': {
        'users': ('@' + top_rt_bots[:10].index).tolist(),
        'tweets': top_rt_bots[:10].values.tolist()
    },
    # top tweeter including bots
    'top_tw_bots': {
        'users': ('@' + top_tw_bots[:10].index).tolist(),
        'tweets': top_tw_bots[:10].values.tolist()
    },
    # These users are bots from the list of top retweeters
    'top_rt_bot_list': ('@' + bots_rt.index).tolist(),
    # These users are bots from the list of top tweeters
    'top_tw_bot_list': ('@' + bots_tw.index).tolist(),
    # top 10 retweeters without bots (retweets)
    'top_10_rt': {
        'users': ('@' + top_rt_10.index).tolist(),
        'tweets': top_rt_10.values.tolist()
    },
    # top 10 tweeters without bots (tweets)
    'top_10_tw': {
        'users': ('@' + top_tw_10.index).tolist(),
        'tweets': top_tw_10.values.tolist()
    },
    # top 10 retweeters without bots (tweets)
    'top_10_rt_tw': {
        'users': ('@' + top_rt_tw.index).tolist(),
        'tweets': top_rt_tw.values.tolist()
    },
    # top 10 tweeters without bots (retweets)
    'top_10_tw_rt': {
        'users': ('@' + top_tw_rt.index).tolist(),
        'tweets': top_tw_rt.values.tolist()
    },
    # Unique users and rates
    'unique_users': int(unique_users),
    'unique_users_rate': unique_users_rate,
    'unique_users_rt': int(unique_users_rt),
    'unique_users_rt_rate': unique_users_rt_rate,
    'unique_users_tw': int(unique_users_tw),
    'unique_users_tw_rate': unique_users_tw_rate,
    # Top 10 tweets
    'top_retweets': {
        'text': top_retweets.index.tolist(),
        'count': top_retweets.values.tolist()
    },
    'geo_coded': {
        'enabled': int(geo_total),
        'disabled': int(df.id_str.size - geo_total)
    },
    'have_location': {
        'yes': int(have_location_total),
        'no': int(df.id_str.size - have_location_total)
    },
    'top_10_popular': {
        'users': top_followers.from_user.tolist(),
        'followers': top_followers.user_followers_count.tolist(),
        'friends': top_followers.user_friends_count.tolist(),
    },
    'hashtags': {
        'text': list(hashtags.keys()),
        'count': list(hashtags.values())
    },
    'mentions': {
        'text': list(mentions.keys()),
        'count': list(mentions.values())
    },
    'volumes': {
        'number': list(vol_tw.keys()),
        'tweeters': list(vol_tw.values()),
        'retweeters': list(vol_rt.values())
    }
}

#
#   Saving results in <charts_data.js>
#

fn_out = 'charts_data.js'

with open(fn_out, 'w', newline='', encoding='utf-8') as f:
    f.write('var ch_data = ')
    f.write(json.dumps(res, indent=4))

print(res)
