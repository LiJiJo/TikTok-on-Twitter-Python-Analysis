import tweepy
import pandas as pd
import re

consumer_key = "idNYoqv2yp3MPjokt87sG7CLs"
consumer_secret = "InNGLHlMJ8kLmu3wRCJyHAIfuY3iEzzdP2b9JAvoMHqviF30pw"
access_key = "1386001563508822016-noWXKHEYkjNP3xGs0TYkvQEhFKEWry"
access_secret = "XhwZfCvruXU6d8hWZn45HDAG0Adzqhvh1tWZKfeiyldYs"

# authentication object using these 4 keys
def twitter_setup():
    
    # Authentication and access using keys
    auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_key,access_secret)
    # api= tweepy.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    api= tweepy.API(auth,wait_on_rate_limit=True)
    # twitter api has rate limit
    # so these 2 rate limit parameters prints message and wait for rate limit to replenish if run out
    # 15min is the rate limit
    
    try:
        api.verify_credentials()
        print("Authentication OK")
    except:
            print("Error during authentication")
    return api

# Create an extractor object to hold the API by calling our twitter setup() function
extractor=twitter_setup()

def keyword_tweets(api,keyword,number_of_tweets):
    # usually when searched by keywords, original tweets and retweets are given
    # this will filter out retweets giving only the original tweets
    new_keyword=keyword+' -filter:retweets'
    
    tweets=[]
    # instead of user timeline, we use search function
    for status in tweepy.Cursor(api.search_tweets, q=new_keyword, 
                                lang="en", tweet_mode='extended', 
                                result_type='mixed').items(number_of_tweets):
        tweets.append(status)
    
    return tweets

# tiktok_alltweets=keyword_tweets(extractor,"TikTok -please -check until:2022-07-03 since:2022-07-02",2400)
data=pd.read_csv("TikTokUserTweets.csv")

# create a panda DataFrame by looping through each element and add it to the DataFrame
# data = pd.DataFrame(data=[tweet.full_text for tweet in tiktok_alltweets], 
#                     columns=['Tweets'])
# data['Tweets_ID'] = [tweet.id for tweet in tiktok_alltweets]
# data['Date'] = [tweet.created_at for tweet in tiktok_alltweets]
# data['Source'] = [tweet.source for tweet in tiktok_alltweets]
# data['Likes_no'] = [tweet.favorite_count for tweet in tiktok_alltweets]
# data['Retweets_no'] = [tweet.retweet_count for tweet in tiktok_alltweets]
# data['Hashtag'] = [tweet.entities['hashtags'] for tweet in tiktok_alltweets]
# data['Location'] = [tweet.user.location for tweet in tiktok_alltweets]
# data['Place'] = [tweet.place for tweet in tiktok_alltweets]
# data['UID'] = [tweet.user.id for tweet in tiktok_alltweets]
# data['Username'] = [tweet.user.screen_name for tweet in tiktok_alltweets]
# data['DisplayName'] = [tweet.user.name for tweet in tiktok_alltweets]
# data['Verified'] = [tweet.user.verified for tweet in tiktok_alltweets]
# data['Tweets_ID']=data['Tweets_ID'].astype(str)

# Note to self: use else if and dictionary next time
# for idx in data.index:
#     if data['Place'][idx]!=None and type(data['Place'][idx])==tweepy.models.Place:
#         data['Place'][idx]=data['Place'][idx].country
#     else:
#         # USA
#         regex=r'(?:^|\W)us(?:$|\W)|(?:^|\W)usa(?:$|\W)|(?:^|\W)united states(?:$|\W)|(?:^|\W)america(?:$|\W)|(?:^|\W)ny(?:$|\W)|(?:^|\W)new york(?:$|\W)|(?:^|\W)nyc(?:$|\W)|(?:^|\W)ohio(?:$|\W)|(?:^|\W)oh(?:$|\W)|(?:^|\W)nashville(?:$|\W)|(?:^|\W)tn(?:$|\W)|(?:^|\W)midwest(?:$|\W)|(?:^|\W)dc(?:$|\W)|(?:^|\W)md(?:$|\W)|(?:^|\W)toronto(?:$|\W)|(?:^|\W)nj(?:$|\W)|(?:^|\W)newark(?:$|\W)|(?:^|\W)hawaii(?:$|\W)|(?:^|\W)nc(?:$|\W)|(?:^|\W)wi(?:$|\W)|(?:^|\W)arizona(?:$|\W)|(?:^|\W)tn(?:$|\W)|(?:^|\W)ak(?:$|\W)|(?:^|\W)pa(?:$|\W)|(?:^|\W)ga(?:$|\W)|(?:^|\W)texas(?:$|\W)|(?:^|\W)tx(?:$|\W)|(?:^|\W)california(?:$|\W)|(?:^|\W)ak(?:$|\W)|(?:^|\W)alaska(?:$|\W)|(?:^|\W)lauderdale(?:$|\W)|(?:^|\W)illinois(?:$|\W)|(?:^|\W)carolina(?:$|\W)|(?:^|\W)in(?:$|\W)|(?:^|\W)indiana(?:$|\W)|(?:^|\W)manhattan(?:$|\W)|(?:^|\W)az(?:$|\W)|(?:^|\W)fl(?:$|\W)|(?:^|\W)florida(?:$|\W)|(?:^|\W)puerto rico(?:$|\W)|(?:^|\W)maryland(?:$|\W)|(?:^|\W)oklahoma(?:$|\W)|(?:^|\W)ms(?:$|\W)|(?:^|\W)mississippi(?:$|\W)|(?:^|\W)mi(?:$|\W)|(?:^|\W)michigan(?:$|\W)|(?:^|\W)idaho(?:$|\W)|(?:^|\W)boise(?:$|\W)|(?:^|\W)atlanta(?:$|\W)|(?:^|\W)oklahoma(?:$|\W)|(?:^|\W)sc(?:$|\W)|(?:^|\W)genesee(?:$|\W)|(?:^|\W)maryland(?:$|\W)|(?:^|\W)tennessee(?:$|\W)|(?:^|\W)ky(?:$|\W)|(?:^|\W)kentucky(?:$|\W)|(?:^|\W)brooklyn(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):        
#             data['Place'][idx]='United States'
#         # UK
#         regex=r'(?:^|\W)uk(?:$|\W)|(?:^|\W)united kingdom(?:$|\W)|(?:^|\W)england(?:$|\W)|(?:^|\W)dunstable(?:$|\W)|(?:^|\W)london(?:$|\W)|(?:^|\W)maidstone(?:$|\W)|(?:^|\W)liverpool(?:$|\W)|(?:^|\W)essex(?:$|\W)|(?:^|\W)wales(?:$|\W)|(?:^|\W)scotland(?:$|\W)|(?:^|\W)birmingham(?:$|\W)|(?:^|\W)yorkshire(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='United Kingdom'
#         # Ireland
#         regex=r'(?:^|\W)ireland(?:$|\W)|(?:^|\W)dublin(?:$|\W)|(?:^|\W)portstewart(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Ireland'
            
#         # Australia
#         regex=r'(?:^|\W)australia(?:$|\W)|(?:^|\W)new south wales(?:$|\W)|(?:^|\W)melbourne(?:$|\W)|(?:^|\W)wurundjeri(?:$|\W)|(?:^|\W)sydney(?:$|\W)|(?:^|\W)aus(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Australia'
            
#         # Malaysia
#         regex=r'(?:^|\W)malaysia(?:$|\W)|(?:^|\W)my(?:$|\W)|(?:^|\W)mas(?:$|\W)|(?:^|\W)petaling jaya(?:$|\W)|(?:^|\W)pj(?:$|\W)|(?:^|\W)negeri(?:$|\W)|(?:^|\W)setiap hari(?:$|\W)|(?:^|\W)selangor(?:$|\W)|(?:^|\W)johor(?:$|\W)|(?:^|\W)kl(?:$|\W)|(?:^|\W)kuala lumpur(?:$|\W)|(?:^|\W)cyberjaya(?:$|\W)|(?:^|\W)negeri(?:$|\W)|(?:^|\W)kelantan(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Malaysia'
            
#         # Singapore
#         regex=r'(?:^|\W)singapore(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Singapore'
            
#         # Philippines 
#         regex=r'(?:^|\W)philippines(?:$|\W)|(?:^|\W)ph(?:$|\W)|(?:^|\W)mandaluyong(?:$|\W)|(?:^|\W)diliman(?:$|\W)|(?:^|\W)quezon(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Philippines'
            
#         # ghana
#         regex=r'(?:^|\W)ghana(?:$|\W)|(?:^|\W)accra(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Ghana'
            
#         # nigeria
#         regex=r'(?:^|\W)nigeria(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Nigeria'
            
#         # africa
#         regex=r'(?:^|\W)africa(?:$|\W)|(?:^|\W)za(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Other African Countries'
            
#         # zimbabwe
#         regex=r'(?:^|\W)zimbabwe(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Zimbabwe'
            
#         # kenya
#         regex=r'(?:^|\W)kenya(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Kenya'
            
#         # algeria
#         regex=r'(?:^|\W)algeria(?:$|\W)|(?:^|\W)annaba(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Algeria'
            
#         # uganda
#         regex=r'(?:^|\W)uganda(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Uganda'
            
#         # kuwait
#         regex=r'(?:^|\W)kuwait(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Kuwait'
            
#         # Japan
#         regex=r'(?:^|\W)tokyo(?:$|\W)|(?:^|\W)japan(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Japan'
            
#         # Germany
#         regex=r'(?:^|\W)germany(?:$|\W)|(?:^|\W)schleswig-holstein(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Germany'
            
#         # South Korea
#         regex=r'(?:^|\W)south korea(?:$|\W)|(?:^|\W)daegu(?:$|\W)|(?:^|\W)seoul(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='South Korea'
            
#         # North Korea
#         regex=r'(?:^|\W)north korea(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='North Korea'
            
#         # Egypt
#         regex=r'(?:^|\W)egypt(?:$|\W)|(?:^|\W)alexandria(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Egypt'
            
#         # Saudi Arabia
#         regex=r'(?:^|\W)saudi arabia(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Saudi Arabia'
            
#         # France
#         regex=r'(?:^|\W)france(?:$|\W)|(?:^|\W)paris(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='France'
            
#         # sri lanka
#         regex=r'(?:^|\W)sri lanka(?:$|\W)|(?:^|\W)lk(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Sri Lanka'
            
#         # thailand
#         regex=r'(?:^|\W)thailand(?:$|\W)|(?:^|\W)bangkok(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Thailand'
            
#         # belgium
#         regex=r'(?:^|\W)belgium(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Belgium'
            
#         # hungary
#         regex=r'(?:^|\W)hungary(?:$|\W)|(?:^|\W)budapest(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Hungary'
            
#         # netherlands
#         regex=r'(?:^|\W)netherlands(?:$|\W)|(?:^|\W)den haag(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Netherlands'
            
#         # jamaica
#         regex=r'(?:^|\W)jamaica(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Jamaica'
            
#         # spain
#         regex=r'(?:^|\W)pain(?:$|\W)|(?:^|\W)spain(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Spain'
            
#         # new zealand
#         regex=r'(?:^|\W)new zealand(?:$|\W)|(?:^|\W)nz(?:$|\W)|(?:^|\W)auckland(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='New Zealand'
            
#         # canada
#         regex=r'(?:^|\W)canada(?:$|\W)|(?:^|\W)toronto(?:$|\W)|(?:^|\W)winnipeg(?:$|\W)|(?:^|\W)alberta(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Canada'
            
#         # india
#         regex=r'(?:^|\W)india(?:$|\W)|(?:^|\W)mumbai(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='India'
            
#         # Cyprus
#         regex=r'(?:^|\W)cyprus(?:$|\W)'
#         if re.search(regex,data['Location'][idx].lower()):
#             data['Place'][idx]='Cyprus'
# del regex
# del idx
# data.to_csv("TikTokUserTweets.csv")


# ==================================================================================================================
import numpy as np
# regular expressions: help to find patterns in string
import matplotlib.pyplot as plt
from wordcloud import WordCloud

data=data.drop_duplicates(subset=('Tweets'))

# Find earliest and latest tweets
print("Earliest tweet is at ", min(data['Date']), "Number of likes: ", data[data['Date']==min(data['Date'])].Tweets)
print("Latest tweet is at ", max(data['Date']), "Number of likes: ", data[data['Date']==max(data['Date'])].Tweets)

likes_max=np.max(data['Likes_no'])
rt_max=np.max(data['Retweets_no'])

most_likes=data[data['Likes_no']==likes_max].Tweets
print("The tweet with most likes is: ", most_likes, "Number of likes: ", likes_max)
most_likes_tweets=data.loc[data['Likes_no']==likes_max]
# help to find the tweets that follow the condition "data['Likes']==likes_max"
# get more columns than just the 1 in most_likes

most_rt=data[data['Retweets_no']==rt_max].Tweets
print("The tweet with most retweets is: ", most_rt, "Number of retweets: ", rt_max)
most_rt_tweets=data.loc[data['Retweets_no']==rt_max]

country_plot=pd.Series(data['Place']).value_counts().plot(kind="bar")
# ==================================================================================================================

from textblob import TextBlob
import string
import random
import networkx as nx
import nltk
nltk.download('stopwords')
nltk.download('punkt') #punkt tokenizer model
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist, bigrams
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import emoji
import seaborn as sns


# lemmatisation
# text normalisation technique
# returns the root form of the word

# the other normalisation technique is called STEMMING
# lemmatisation consider the context of the words and return the root form after
# stemming just cut off the last part of the word (eg: "s", " 's ", "-ing")
def lemmatize_sentence(token):
    # initiate wordnetlemmatizer()
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence=[]
    
    # each of the words in the doc will be assigned to a grammatical category
    # part of speech tagging
    # NN for noun, VB for verb, adjective, preposition, etc
    # lemmatizer can determine role of word 
        # and then correctly identify the most suitable root form of the word
    # return the root form of the word
    for word, tag in pos_tag(token):
        if tag.startswith('NN'):
            pos='n'
        elif tag.startswith('VB'):
            pos='v'
        else:
            pos='a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word,pos))
    return lemmatized_sentence

# remove noises from the tweets like links, mentions, numbers, punctuations, stopwords
# stopwords= uh, the, and
    # occurs frquently but dont carry meaning
# 1, remove http/https links
# 2, remove t.co and anything behind it
# 3, all @ mentions
# 4, all numbers
# 5, remove any non normal UT8 characters
# 6, remove emojis (just in case)
# 7, remove \r\n
# 8, remove non utf8+ascii characters (triple confirm)
# 9, turn double spaces to 1
# if condition to keep words with more than 3 char count 
     # if word is not punctuations 
     # if word is not in stopwords list
# functioin returns cleaned tokens- each of the words occured in the documents list
def remove_noise(tweet_tokens, stop_words):
    cleaned_tokens=[]
    for token in tweet_tokens:
        token = re.sub('http([!-~]+)?','',token)
        token = re.sub('//t.co/[A-Za-z0-9]+','',token)
        token = re.sub('(@[A-Za-z0-9_]+)','',token)
        token = re.sub('[0-9]','',token)
        token = re.sub('[^ -~]','',token)
        token = re.sub(emoji.get_emoji_regexp(), "", token)
        token = token.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
        token = re.sub('[^\x00-\x7f]','', token) 
        token = re.sub("\s\s+" , " ", token)
        if (len(token)>3) and (token not in string.punctuation) and (token.lower() not in stop_words):
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


# important to remove as many unnecessary words as possible
# requires self explaratory and self understanding of problems and context
stop_words=stopwords.words('english')
stop_words.extend(['video','account','social media', 'social medium','sosmed','people','also',
                   '/like','/comment','/subscribe','/tiktok','/youtube','twitter','instagram','tiktok',
                   'do like','do follow', 'please like','please follow','please',
                   'need', 'followers', 'likes', 'views', 'shares', 'subscribers', 'follow','follows'
                   'Instagram', 'twitter', 'shopee', 'youtube', 'facebook',
                   'you tube','make','n e e d','check it out', 'check', 'check out','checks'
                   'know','go','watch','videos','going','say','saying','said','says'])
    
# can begin cleaning data
# tokenise the tweets to smaller units
    # one of the most important process
    # split the sentences into smaller units (word,phrase,terms)
# Tokenise from nltk library to tokenise the tweets
tweets_token=data['Tweets'].apply(word_tokenize).tolist()

cleaned_tokens=[]
# lemmanise and remove noise function to help clean the data
# at the end returns the list of 'cleaned tokens'
for token in tweets_token:
    rm_noise =remove_noise(token, stop_words)
    lemma_tokens=lemmatize_sentence(rm_noise)
    cleaned_tokens.append(lemma_tokens)

tweet_list=[tweet for tweet in cleaned_tokens if tweet!='[]']
# list still messy, just need text value dont need id:

# when read the datatype changed to a line of raw strings
# so need to clean up as text
# need to find what is after the keyword text and before indices
# firstly remove all punctuations

# punctuation list created
punc= '''!()-[]{};:'"\,=<>./?@#$%^&*_~'''+'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'

# compress all rows into 1 string only in 'string' variable
twstring =''. join([str(item) for item in tweet_list])

# look thru the values, if the punctuations are found they are deleted
# use regular expression to retrieve hashtag we want
# in between text and indices
# god dang regex
for letter in twstring:
    if letter in punc:
        twstring = twstring.replace(letter,"")
        # tweets=re.findall(r'text (.*?) indices',twstring.lower())
        
# # word cloud wants to be passed with string value not list
# # converts the list to 1 raw string
# tweets_string=' '.join(tweets)

# create ai generated word cloud image
tweets_wordcloud=WordCloud(width=400,height=400,
                            background_color='white',
                            min_font_size=10,collocations=False).generate(twstring)
plt.imshow(tweets_wordcloud)
plt.axis("off")
plt.show()

###################### Classification ##############################

text_blob=[]
for tweet in data['Tweets'].tolist():
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity==0:
        sentiment="Neutral"
    elif analysis.sentiment.polarity>0:
        sentiment="Positive"
    elif analysis.sentiment.polarity<0:
        sentiment="Negative"
    text_blob.append(sentiment)
    
data["Sentiment"]=text_blob

data.groupby(by='Sentiment').mean()
data.groupby(by='Sentiment')['Sentiment'].count()


tps=data[['Tweets','Place','Sentiment']]
tps=tps[tps.Sentiment != 'Neutral']
############################Check Sentiment by Country##############################
tps.groupby(by=['Place'])['Sentiment'].size()
groupedforcomparison=tps.groupby(by=['Place'])['Sentiment'].size()

othercountrylist=[]
for idx, place in enumerate(groupedforcomparison):
    if place<=15:
        othercountrylist.append(groupedforcomparison.index[idx])

len(othercountrylist)

tps.loc[tps['Place'].isin(othercountrylist),'Place']='Others'
# tps.groupby(by=['Place'])['Sentiment'].size().index[0]
tps.groupby(by=['Place'])['Sentiment'].size()
# tps.groupby(by=['Place','Sentiment']).size()

# pbs=tps.groupby(by=['Place','Sentiment']).size().reset_index()
# pbs=pbs.rename(columns={0:"Count"})
pbs=tps.pivot_table(index='Place',columns='Sentiment',values='Sentiment', aggfunc='size')
ax = pbs.plot(kind="bar",color=["#ca472f","#0b84a5"], grid=True,rot=0, title="Country's Sentiment \non TikTok")
plt.xticks(rotation=30, horizontalalignment="center")
ax.set_xlabel("Country")
ax.set_ylabel("count")
plt.rcParams["figure.figsize"] = [10, 6]
plt.show()

pbs=tps.pivot_table(index='Place',columns='Sentiment',values='Sentiment', aggfunc='size')
pbs['Negative']=pbs['Negative'].fillna(0)
pbs['Positive']=pbs['Positive'].fillna(0)
pbs['Negative']=pbs['Negative']/(pbs['Negative']+pbs['Positive'])*100
pbs['Positive']=100-pbs['Negative']
ax = pbs.plot(kind="bar",color=["#ca472f","#0b84a5"], grid=True,rot=0, title="Country's Sentiment by Percentage \non TikTok")
plt.xticks(rotation=30, horizontalalignment="center")
ax.set_xlabel("Country")
ax.set_ylabel("count")
plt.rcParams["figure.figsize"] = [10, 6]
plt.show()

########################################################################################################
tweets_token=data.loc[data['Sentiment']=='Positive']['Tweets'].apply(word_tokenize).tolist()

cleaned_tokens=[]
for token in tweets_token:
    rm_noise =remove_noise(token, stop_words)
    lemma_tokens=lemmatize_sentence(rm_noise)
    cleaned_tokens.append(lemma_tokens)

tweet_list=[tweet for tweet in cleaned_tokens if tweet!='[]']

# punctuation list created
punc= '''!()-[]{};:'"\,=<>./?@#$%^&*_~'''+'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'

# compress all rows into 1 string only in 'string' variable
twstring =''. join([str(item) for item in tweet_list])

for letter in twstring:
    if letter in punc:
        twstring = twstring.replace(letter,"")
        # tweets=re.findall(r'text (.*?) indices',twstring.lower())

tweets_wordcloud=WordCloud(width=400,height=400,
                            background_color='white',
                            min_font_size=10,collocations=False).generate(twstring)
plt.imshow(tweets_wordcloud)
plt.axis("off")
plt.show()

tweets_token=data.loc[data['Sentiment']=='Negative']['Tweets'].apply(word_tokenize).tolist()

cleaned_tokens=[]
for token in tweets_token:
    rm_noise =remove_noise(token, stop_words)
    lemma_tokens=lemmatize_sentence(rm_noise)
    cleaned_tokens.append(lemma_tokens)

tweet_list=[tweet for tweet in cleaned_tokens if tweet!='[]']

# punctuation list created
punc= '''!()-[]{};:'"\,=<>./?@#$%^&*_~'''+'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'

# compress all rows into 1 string only in 'string' variable
twstring =''. join([str(item) for item in tweet_list])

for letter in twstring:
    if letter in punc:
        twstring = twstring.replace(letter,"")
        # tweets=re.findall(r'text (.*?) indices',twstring.lower())

tweets_wordcloud=WordCloud(width=400,height=400,
                            background_color='white',
                            min_font_size=10,collocations=False).generate(twstring)
plt.imshow(tweets_wordcloud)
plt.axis("off")
plt.show()
# ==================================================================================================================        

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models
  

# Gensim is specialised for topic modelling
# pyLDAvis is for visualisation for the topics



tweets_token=data['Tweets'].apply(word_tokenize).tolist()
cleaned_tokens=[]
for token in tweets_token:
    lemma_tokens=lemmatize_sentence(token)
    rm_noise =remove_noise(lemma_tokens, stop_words)
    cleaned_tokens.append(rm_noise)

id2word=corpora.Dictionary(cleaned_tokens)
print(id2word.token2id)

id2word.filter_extremes(no_below=30,no_above=30)
print(id2word.token2id)

corpus=[id2word.doc2bow(text) for text in cleaned_tokens]


ldamodel=gensim.models.ldamodel.LdaModel(corpus=corpus,
                                         id2word=id2word,
                                         passes=50,
                                         iterations=50,
                                         num_topics=7,
                                         random_state=1)


ldamodel.print_topics(num_words=7)
coherence_model_lda=CoherenceModel(model=ldamodel,texts=cleaned_tokens,dictionary=id2word,coherence='u_mass')
# coherence_model_lda=CoherenceModel(model=ldamodel,texts=cleaned_tokens,dictionary=id2word,coherence='u_mass')
coherence_lda=coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
print('\nCoherence Score: ', coherence_model_lda.get_coherence())
# score:
# 1    5.7
# 2    8.7
# 3    9.7
# 4    10.6
# 5    9.7
# 6    9.2
# 7    8.44<------------
# 8    9.3
# 9    8.7
# 10    9.2
# 11    8.25<------------
# 12    9
# 13    8.6
# 14    8.8

# Acceptable score
# Based on Internet, the lower the score the better
# 11 has too many overlaps

# ====================================
# VISUALISATION TIME
# LDA vis library

# interactive html widget
# show topic distribution and top words associated to each topic
# prepare visualisation
lda_display=pyLDAvis.gensim_models.prepare(ldamodel, corpus, id2word)
# save visualisation into html format and open thru browser
pyLDAvis.save_html(lda_display,'lda.html')

# shows how close the topics are
# size matters              omglol
# could reduce the number of k so any intersecting topics can be combined

# Model each of the tweets using the LDA model and see how this model will group the tweets based on the topics given

# now loop for all tweets
tp_list=[]
for i in range(len(ldamodel[corpus])):
    tp=ldamodel[corpus][i]
    tp=sorted(tp, key=lambda tp: tp[1],reverse=True)
    tp_list.append(tp[0][0])

# dataframe of only the tweets and their topics
# the topics will be renamed according to the interpretation
topicwithtweet=pd.DataFrame(data['Tweets'])
topicwithtweet['Topic']=tp_list
topicwithtweet['Topic']=topicwithtweet['Topic'].replace([0,1,2,3,4,5,6],
                                        ['Studying content on TikTok and how to get popular', 'User’s intention and possibly other people offering or requesting help or work on the platform',
                                          'User’s engagement and opinions', 'Viral TikTok content, mainly about a girl doing challenge that involves music',
                                          'Strategizing TikTok posts with inclusion of feedback','Real world and personal topics.',
                                          'Regarding the data access (most probably by China)'])

for topic in [0,1,2,3,4,5,6]:
    # topicwithtweet.loc[topicwithtweet['Topic']==topic]
    tweets_token=topicwithtweet.loc[topicwithtweet['Topic']==topic]['Tweets'].apply(word_tokenize).tolist()
    
    cleaned_tokens=[]
    for token in tweets_token:
        rm_noise =remove_noise(token, stop_words)
        lemma_tokens=lemmatize_sentence(rm_noise)
        cleaned_tokens.append(lemma_tokens)
    
    tweet_list=[tweet for tweet in cleaned_tokens if tweet!='[]']
   
    # punctuation list created
    punc= '''!()-[]{};:'"\,=<>./?@#$%^&*_~'''+'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    
    # compress all rows into 1 string only in 'string' variable
    twstring =''. join([str(item) for item in tweet_list])
    
    # look thru the values, if the punctuations are found they are deleted
    # use regular expression to retrieve hashtag we want
    # in between text and indices
    # god dang regex
    for letter in twstring:
        if letter in punc:
            twstring = twstring.replace(letter,"")
            # tweets=re.findall(r'text (.*?) indices',twstring.lower())
            
    # # word cloud wants to be passed with string value not list
    # # converts the list to 1 raw string
    # tweets_string=' '.join(tweets)
    
    # create ai generated word cloud image
    tweets_wordcloud=WordCloud(width=400,height=400,
                                background_color='white',
                                min_font_size=10,collocations=False).generate(twstring)
    plt.imshow(tweets_wordcloud)
    plt.axis("off")
    plt.show()

topic7=topicwithtweet.loc[topicwithtweet['Topic']==6]['Tweets']

text_blob=[]
for tweet in topic7.tolist():
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity==0:
        sentiment="Neutral"
    elif analysis.sentiment.polarity>0:
        sentiment="Positive"
    elif analysis.sentiment.polarity<0:
        sentiment="Negative"
    text_blob.append(sentiment)

topic7=pd.DataFrame(topic7)
topic7["Sentiment"]=text_blob

topic7.groupby(by='Sentiment').mean()
topic7.groupby(by='Sentiment')['Sentiment'].count()


tps=topic7[['Tweets','Place','Sentiment']]
tps=tps[tps.Sentiment != 'Neutral']
tps.groupby(by=['Place'])['Sentiment'].size()
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
from datetime import datetime

def get_user_tweets(api, username):
    # create empty list to temporarily hold data for scraping
    tweets=[]
    # use for loop to keep feeding every tweets scraped using user timeline and cursor function to the empty list
    # user timeline help to get the tweets
    # cursor function help to look through the page
    # need to look through the page manually without cursor function
    
    # can also use tweepy.Cursor(api.user_timeline,screen_name=username).pages(): to scrape by page
    for status in tweepy.Cursor(api.user_timeline,screen_name=username).items():
        tweets.append(status)
    return tweets

# tiktoktweets=get_user_tweets(extractor, 'tiktok_us')
# print("Number of tweets extracted: ", len(tiktoktweets))
# tiktoktweets
# data = pd.DataFrame(data=[tweet.text for tweet in tiktoktweets], 
#                     columns=['Tweets'])
# data['Tweets_ID'] = [tweet.id for tweet in tiktoktweets]
# data['Date'] = [tweet.created_at for tweet in tiktoktweets]
# data['Source'] = [tweet.source for tweet in tiktoktweets]
# data['Likes_no'] = [tweet.favorite_count for tweet in tiktoktweets]
# data['Retweets_no'] = [tweet.retweet_count for tweet in tiktoktweets]
# # data['Hashtag'] = [tweet.entities['hashtags'] for tweet in tiktoktweets]
# # data['Location'] = [tweet.user.location for tweet in tiktoktweets]
# # data['Place'] = [tweet.place for tweet in tiktoktweets]
# # data['UID'] = [tweet.user.id for tweet in tiktoktweets]
# # data['Username'] = [tweet.user.screen_name for tweet in tiktoktweets]
# # data['DisplayName'] = [tweet.user.name for tweet in tiktoktweets]
# # data['Verified'] = [tweet.user.verified for tweet in tiktoktweets]

data=pd.read_csv("TikTokUserTweets.csv")
# data.to_csv("TikTokTweets.csv")
print("Earliest tweet is at ", min(data['Date']))
print("Latest tweet is at ", max(data['Date']))

timelineTikTokProf=data[['Date','Likes_no','Retweets_no']]
plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Likes_no'], label='Likes')
plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Retweets_no'], label='Retweets')
plt.legend(loc="upper right")
plt.title("TikTok_US Engagement Over 3 Years")
plt.figure(figsize=(20,12)) 
plt.show()

timelineTikTokProf=data[data['Date'].dt.date<datetime.strptime('2021','%Y').date()][['Date','Likes_no','Retweets_no']]
plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Likes_no'], label='Likes')
plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Retweets_no'], label='Retweets')
plt.legend(loc="upper right")
plt.title("TikTok_US Engagement in 2020")
plt.figure(figsize=(20,12)) 
plt.show()

timelineTikTokProf=data[np.logical_and(data['Date'].dt.date<datetime.strptime('2022','%Y').date() ,
                                       data['Date'].dt.date>=datetime.strptime('2021','%Y').date())][['Date','Likes_no','Retweets_no']]
plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Likes_no'], label='Likes')
plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Retweets_no'], label='Retweets')
plt.legend(loc="upper left")
plt.title("TikTok_US Engagement in 2021")
plt.figure(figsize=(20,12)) 
plt.show()

timelineTikTokProf=data[np.logical_and(data['Date'].dt.date<datetime.strptime('2023','%Y').date() ,
                                       data['Date'].dt.date>=datetime.strptime('2022','%Y').date())][['Date','Likes_no','Retweets_no']]
plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Likes_no'], label='Likes')
plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Retweets_no'], label='Retweets')
plt.legend(loc="upper right")
plt.title("TikTok_US Engagement in 2022")
plt.figure(figsize=(20,12)) 
plt.show()


##################################################################### Likes #####################################################################
###### Year 2020##########

timelineTikTokProf=data[data['Date'].dt.date<datetime.strptime('2021','%Y').date()][['Date','Likes_no']]
plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Likes_no'])
plt.figure(figsize=(20,12)) 
plt.show()

most_likes_tweets= pd.DataFrame(columns=['Tweets','Tweets_ID','Date','Source','Likes_no','Retweets_no'])
for i in range(1):
    
    likes_max=np.max(timelineTikTokProf['Likes_no'])
    print("The ", i+1," tweet with the highest like count in this year is:\n ",likes_max)
    most_likes_tweets.append(data[data['Likes_no']==likes_max][['Tweets','Date','Likes_no']])
    
    timelineTikTokProf=timelineTikTokProf[timelineTikTokProf['Likes_no']!=likes_max]
# most_likes_tweets=pd.concat(most_likes_tweets)

plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Likes_no'])
plt.figure(figsize=(20,12)) 
plt.show()

###### Year 2021############
timelineTikTokProf=data[np.logical_and(data['Date'].dt.date<datetime.strptime('2022','%Y').date() ,
                                       data['Date'].dt.date>=datetime.strptime('2021','%Y').date())][['Date','Likes_no']]
plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Likes_no'])
# plt.rcParams["figure.figsize"] = [20, 12]
plt.figure(figsize=(20,12)) 
plt.show()


# most_likes_tweets=[]
for i in range(3):
    
    likes_max=np.max(timelineTikTokProf['Likes_no'])
    print("The ", i+1," tweet with the highest like count in this year is:\n ",likes_max)
    most_likes_tweets.append(data[data['Likes_no']==likes_max][['Tweets','Date','Likes_no']])
    
    timelineTikTokProf=timelineTikTokProf[timelineTikTokProf['Likes_no']!=likes_max]
# most_likes_tweets=pd.concat(most_likes_tweets)

plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Likes_no'])
plt.figure(figsize=(20,12)) 
plt.show()

############ Year 2022##############
timelineTikTokProf=data[np.logical_and(data['Date'].dt.date<datetime.strptime('2023','%Y').date() ,
                                       data['Date'].dt.date>=datetime.strptime('2022','%Y').date())][['Date','Likes_no']]
plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Likes_no'])
# plt.rcParams["figure.figsize"] = [20, 12]
plt.figure(figsize=(20,12)) 
plt.show()

# most_likes_tweets=[]
for i in range(3):
    
    likes_max=np.max(timelineTikTokProf['Likes_no'])
    print("The ", i+1," tweet with the highest like count in this year is:\n ",likes_max)
    most_likes_tweets.append(data[data['Likes_no']==likes_max][['Tweets','Date','Likes_no']])
    
    timelineTikTokProf=timelineTikTokProf[timelineTikTokProf['Likes_no']!=likes_max]
# most_likes_tweets=pd.concat(most_likes_tweets)

plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Likes_no'])
plt.figure(figsize=(20,12)) 
plt.show()

##################################################################### Retweets #####################################################################
############## Year 2020############
timelineTikTokProf=data[data['Date'].dt.date<datetime.strptime('2021','%Y').date()][['Date','Retweets_no']]
plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Retweets_no'])
plt.figure(figsize=(20,12)) 
plt.show()

most_rt_tweets=pd.DataFrame(columns=['Tweets','Tweets_ID','Date','Source','Likes_no','Retweets_no'])
for i in range(1):
    
    rt_max=np.max(timelineTikTokProf['Retweets_no'])
    print("The ", i+1," tweet with the highest retweets count in this year is:\n ",rt_max)
    most_rt_tweets.append(data[data['Retweets_no']==rt_max][['Tweets','Date','Retweets_no']])
    
    timelineTikTokProf=timelineTikTokProf[timelineTikTokProf['Retweets_no']!=rt_max]
# most_rt_tweets=pd.concat(most_rt_tweets)

plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Retweets_no'])
plt.figure(figsize=(20,12)) 
plt.show()


############ Year 2021###############
timelineTikTokProf=data[np.logical_and(data['Date'].dt.date<datetime.strptime('2022','%Y').date() ,
                                       data['Date'].dt.date>=datetime.strptime('2021','%Y').date())][['Date','Retweets_no']]
plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Retweets_no'])
plt.figure(figsize=(20,12)) 
plt.show()

# most_rt_tweets=[]
for i in range(3):
    
    rt_max=np.max(timelineTikTokProf['Retweets_no'])
    print("The ", i+1," tweet with the highest retweets count in this year is:\n ",rt_max)
    most_rt_tweets.append(data[data['Retweets_no']==rt_max][['Tweets','Date','Retweets_no']])
    
    timelineTikTokProf=timelineTikTokProf[timelineTikTokProf['Retweets_no']!=rt_max]
# most_rt_tweets=pd.concat(most_rt_tweets)

plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Retweets_no'])
plt.figure(figsize=(20,12)) 
plt.show()

################ Year 2022####################
timelineTikTokProf=data[np.logical_and(data['Date'].dt.date<datetime.strptime('2023','%Y').date() ,
                                       data['Date'].dt.date>=datetime.strptime('2022','%Y').date())][['Date','Retweets_no']]
plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Retweets_no'])
plt.figure(figsize=(20,12)) 
plt.show()

# most_rt_tweets=[]
for i in range(3):
    
    rt_max=np.max(timelineTikTokProf['Retweets_no'])
    print("The ", i+1," tweet with the highest retweets count in this year is:\n ",rt_max)
    most_rt_tweets.append(data[data['Retweets_no']==rt_max][['Tweets','Date','Retweets_no']])
    
    timelineTikTokProf=timelineTikTokProf[timelineTikTokProf['Retweets_no']!=rt_max]
# most_rt_tweets=pd.concat(most_rt_tweets)

plt.plot(timelineTikTokProf['Date'],timelineTikTokProf['Retweets_no'])
plt.figure(figsize=(20,12)) 
plt.show()


#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################


def keyword_tweetsandreplies(api,keyword,number_of_tweets):
    # usually when searched by keywords, original tweets and retweets are given
    # this will filter out retweets giving only the original tweets
    new_keyword=keyword+' -filter:retweets'
    
    tweets=[]
    # instead of user timeline, we use search function
    for status in tweepy.Cursor(api.search_tweets, q=new_keyword, 
                                lang="en", tweet_mode='extended', 
                                result_type='mixed').items(number_of_tweets):
        tweets.append(status)
        # replykey='(to:'+status.user.screen_name+') since:'+status.created_at.strftime("%Y-%m-%d")
        # for stuff in tweepy.Cursor(api.search_tweets, q=replykey, 
        #                             lang="en", tweet_mode='extended').items(number_of_tweets):
        #     if stuff.in_reply_to_status_id_str == status.id:
        #         tweets.append(stuff)
    return tweets


tiktok_newstweets=keyword_tweetsandreplies(extractor,"https://www.businessinsider.com/tiktok-confirms-us-user-data-accessed-in-china-bytedance-2022-7",2400)

# create a panda DataFrame by looping through each element and add it to the DataFrame
data = pd.DataFrame(data=[tweet.full_text for tweet in tiktok_newstweets], 
                    columns=['Tweets'])
data['Tweets_ID'] = [tweet.id for tweet in tiktok_newstweets]
data['Date'] = [tweet.created_at for tweet in tiktok_newstweets]
data['Source'] = [tweet.source for tweet in tiktok_newstweets]
data['Likes_no'] = [tweet.favorite_count for tweet in tiktok_newstweets]
data['Retweets_no'] = [tweet.retweet_count for tweet in tiktok_newstweets]

data=data.drop_duplicates(subset=('Tweets'))

data.to_csv("TikTokNewsTweets.csv")

def remove_noise(tweet_tokens, stop_words):
    cleaned_tokens=[]
    for token in tweet_tokens:
        token = re.sub('http([!-~]+)?','',token)
        token = re.sub('//t.co/[A-Za-z0-9]+','',token)
        token = re.sub('(@[A-Za-z0-9_]+)','',token)
        token = re.sub('[0-9]','',token)
        token = re.sub('[^ -~]','',token)
        token = re.sub(emoji.get_emoji_regexp(), "", token)
        token = token.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
        token = re.sub('[^\x00-\x7f]','', token) 
        token = re.sub("\s\s+" , " ", token)
        if (len(token)>3) and (token not in string.punctuation) and (token.lower() not in stop_words):
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


# important to remove as many unnecessary words as possible
# requires self explaratory and self understanding of problems and context
stop_words=stopwords.words('english')
tweets_token=data['Tweets'].apply(word_tokenize).tolist()

cleaned_tokens=[]
# lemmanise and remove noise function to help clean the data
# at the end returns the list of 'cleaned tokens'
for token in tweets_token:
    rm_noise =remove_noise(token, stop_words)
    # lemma_tokens=lemmatize_sentence(rm_noise)
    # cleaned_tokens.append(lemma_tokens)
    cleaned_tokens.append(rm_noise)

tweet_list=[]
for tokens in cleaned_tokens:
    toke=' '. join([str(token) for token in tokens])
    tweet_list.append(toke)

data['CleanTweet']=tweet_list

# data=data.drop_duplicates(subset=('CleanTweet'))
twstring=''. join([str(item) for item in data['CleanTweet']])
twstring = re.sub('(?:^|\W)data(?:$|\W)|(?:^|\W)user(?:$|\W)|(?:^|\W)cleaned(?:$|\W)|(?:^|\W)confirms(?:$|\W)|(?:^|\W)confirm(?:$|\W)','',twstring)
# twstring =''. join([str(item) for item in tweet_list])
tweets_wordcloud=WordCloud(width=400,height=400,
                            background_color='white',
                            min_font_size=10,collocations=False).generate(twstring)
plt.imshow(tweets_wordcloud)
plt.axis("off")
plt.show()

text_blob=[]
for tweet in data['Tweets'].tolist():
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity==0:
        sentiment="Neutral"
    elif analysis.sentiment.polarity>0:
        sentiment="Positive"
    elif analysis.sentiment.polarity<0:
        sentiment="Negative"
    text_blob.append(sentiment)
    
data["Sentiment"]=text_blob

data.groupby(by='Sentiment').mean()
data.groupby(by='Sentiment')['Sentiment'].count()









