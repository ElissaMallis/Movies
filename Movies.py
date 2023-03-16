import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
pd.set_option('display.max_columns',15)
pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None
np.seterr(divide = 'ignore')

# load data from csv ----------------------------

genre = pd.read_csv("genre_list_df.csv")

# I identified id = 36 was missing, looked up the dataset and found that it was the history genre
# I couldn't find the meaning of 00 and no other id was 0 so I made all ambiguous ids to 0
genre = genre.append([{'id':36,'name':'History'}, {'id':0,'name':'Unknown'}], ignore_index=True)

# rename the id column to match movie dataframe for smoother joining
genre.rename(columns={'id':'genre_id'}, inplace=True)

# load movie data, csv didn't load with default UTF8 encoding
movies = pd.read_csv("movie_list_df.csv", encoding="latin1", parse_dates=['release_date'])

# clean data ----------------------------------------

# change vote count to log scale to use in a weighted average (explained later)
movies['log_vote_count'] = np.where(movies['vote_count'] == 0 , 0, np.log(movies['vote_count']))

# remove extraneous characters from genre ids
cleaned_movies = movies
cleaned_movies['genre_ids'] = cleaned_movies['genre_ids'].str.replace(':', ',')
cleaned_movies['genre_ids'] = cleaned_movies['genre_ids'].str.replace(r'[c() ]|(integer)', '')

# for genre popularity, we want to explode out the genres since one movie can fall into multiple genres
explode_movies = cleaned_movies.assign(genre_id=cleaned_movies['genre_ids'].str.split(',')).explode('genre_id')
explode_movies['genre_id'] = pd.to_numeric(explode_movies['genre_id'])

# label movie genre ids with genre names
labeled_movies = pd.merge(explode_movies, genre, on='genre_id', how='left')


"1. Considering only movies released during December 2017, what was the most popular movie genre?"

# filter movies to only december, 2017
dec_movie = labeled_movies[
    ((labeled_movies['release_date'] >= '2017-12-01')
     & (labeled_movies['release_date'] <= '2017-12-31'))
]

"""a. How did you define a metric for genre popularity and why is it meaningful?"""

# define new metric for genre popularity------------------------
# I decided to calculate a new weighted vote average based on a bayesian formula
# This new metric takes into account vote average and vote count
# The lower the vote count, the less we trust the vote average and it moves toward the overall average for all movies
# low rated movies with high vote counts will be rated lower
# To be popular, vote counts and vote average need to be high
# But it also punishes poorly rated movies with a lot of votes

print('\nDescriptive statistics of movie data')
print(movies.describe())

# Based on the descriptive statistics, I converted vote count to log scale because it has a long tail distribution
# 75% of movies have a vote count of less than or equal to 92, while the max count is 25,222
# This changes the scale so that the distance between values is more linear in the weight value and not skewed by the few movies with overwhelming vote counts
max_log_vote_count = max(movies['log_vote_count'])
dec_movie["weight"] = dec_movie['log_vote_count'] / max_log_vote_count

# Calculate my popularity scoring per movie
all_movies_vote_average = movies['vote_average'].mean()
dec_movie["weighted_vote_ave"] = dec_movie["weight"] * dec_movie["vote_average"] + (1 - dec_movie["weight"]) * all_movies_vote_average

# I decided to average the new weighted vote rather than add the weighted vote-ave as it would just bias towards the genre with the most movies
genre_popularity = dec_movie.groupby('name').agg( new_popularity_ave=('weighted_vote_ave', 'mean'), popularity_ave = ('popularity','mean'))
print('\nNew popularity rankings: ')
print(genre_popularity.sort_values(by=['new_popularity_ave'], ascending=False))
# Most popular genre is War

"""b. What other metrics did you consider? Please provide a visualization to support your finding."""

# Figure 1: histogram of vote counts
# Histogram shows that vote counts are heavily skewed to a small volume of votes,
# and very few movies have extremely large vote counts
fig, ax = plt.subplots(figsize=(6, 6))
movies["vote_count"].plot(
    title = "Distribution of Movie Vote Counts",
    kind = "hist",
    ax=ax,
    bins = [0,5,10,15,20,30,50,100,500,1000,2000,5000,8000,20000,26000],
)
ax.set_ylabel("Movie Frequency")
ax.set_xlabel("Vote Count")
# # plt.show()
plt.close(fig=None)

# checking to see if my popularity correlates with TMBD popularity, I am not using popularity per the instructions but I
# would like to see if my popularity metric generally tracks with the one provided. They correlate at .51 so there is some
# moderate relationship. I would never be able to recreate the popularity as I do not have access to all the variables that went into that measure

# figure 2: Plot of Popularity Vs. New Metric
plt.plot(dec_movie["weighted_vote_ave"], dec_movie["popularity"], 'ro')
ax.set_title('Plot of Popularity Vs. New Metric, Dec 2017')
ax.set_ylabel("Popularity")
ax.set_xlabel("Weighted Vote Average")
# # plt.show()
plt.close(fig=None)

print(' ')
print('Correlation between my metric for genre popularity and the given popularity')
print(round(dec_movie[['weighted_vote_ave','popularity']].corr(),2))

# Figure 3: Release Date vs Vote Count
# I considered using release date since a movie with an early release date could have an advantage over a newer movie
# because more people had time to find, see and vote on that movie
# However I found release date to have no impact on the vote count, making it unnecesary to include in my popularity metric
plt.plot(dec_movie["release_date"], dec_movie["vote_count"], 'ro')
ax.set_title('Movie Release Date vs Vote Count, Dec 2017')
ax.set_ylabel("Vote Count")
ax.set_xlabel("Release Date")
# # plt.show()
plt.close(fig=None)

# -------------------------------------------------------------------------------------------
"""2. Use the entire dataset (i.e., all movies released between 2015-01-01 and 2018-12-31) to answer the following questions."""

# build new explanatory variables --------------------------------

# create days since release date, instructions do not specify if that should be present date or last release date of data set
# days since last release date within the data set makes the most sense, given this data is a few years old
regression_movies = cleaned_movies
last_date = max(cleaned_movies['release_date'])
regression_movies["days_since_release"] = (last_date - cleaned_movies['release_date']).dt.days

# create June dummy variable
regression_movies['month'] = pd.DatetimeIndex(regression_movies['release_date']).month
regression_movies['june_release'] = np.where(regression_movies["month"] == 6 , 1 , 0)

# create linear regression ------------------------------------

x_independent_variables = regression_movies[['vote_average', 'vote_count', 'days_since_release', 'june_release']]
y_dependent_variable = regression_movies[['popularity']]

regr = linear_model.LinearRegression()
regr.fit(x_independent_variables, y_dependent_variable)

predicted_popularity = regr.predict(x_independent_variables)
residuals = predicted_popularity - y_dependent_variable



"Questions"
"""a. Fit a linear regression model using the popularity measure that was provided in your original dataset as your 
    outcome (i.e., dependent) variable. Include the following explanatory (i.e., independent) variables: vote average, 
    vote count, time since primary release date, and a dummy variable to indicate whether a movie was released in June. 
    Provide your estimation results."""
# linear regression model results
# popularity = 0.472(vote average) + 0.007(vote count) - 0.002(days since release) + 0.191(june release) + 3.477
X2 = sm.add_constant(x_independent_variables)
lm = sm.OLS(y_dependent_variable, X2)
print('\n')
print(lm.fit().summary())

"""b. Does your fitted model satisfy classical assumptions of the linear regression model? Provide supporting evidence 
    and/or a visualization."""
# No, the fitted model does not satisfy the classical assumptions

"Assumption: Linear Relationship between the Target and the Feature"
# This assumption is violated, coordinates do not follow a diagonal line

# Figure 4: plot predicted values vs actual
plt.scatter(y_dependent_variable, predicted_popularity)
ax.set_title('Predicted Popularity Vs Popularity')
ax.set_ylabel("Predicted Popularity")
ax.set_xlabel("Popularity")
ax.set_ylim(0,300)
ax.set_xlim(0,300)
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
# plt.show()
plt.close(fig=None)

'Assumption: The error terms are normally distributed'
# Figure 5: histogram of residuals from the model
# The distribution is skewed
plt.hist(residuals, bins=1000)
ax.set_title('Distribution of Residuals')
ax.set_ylabel("Frequency")
ax.set_xlabel("Residuals")
# # it looks a little bit skewed
# plt.show()
plt.close(fig=None)

'Assumption: No Autocorrelation of the Error Terms'
# Durbin-Watson: 0.748
# The closer the test statistic is to 0, the more evidence of positive serial correlation.
# A score of 2 would indicate no autocorrelation
# assumption is violated

'Assumption: Homoscedasticity of Error Terms'
# Residuals violate homoscedasticity - residuals are not randomly distributed
# This plot should look random and center around zero but it doesn't, residuals should have relative constant variance.
# Figure 6: residual plot
plt.scatter(predicted_popularity, residuals, alpha=.5)
ax.set_title('Homoscedasticity')
ax.set_ylabel("Residuals")
ax.set_xlabel("Predicted Popularity")
ax.plot(ax.get_xlim(), [0,0], ls="--", c=".3")
# plt.show()
plt.close(fig=None)

'Assumption: No Collinearity Among Predictors'
# checking for collinearity, build a pair plot and a correlation table
# Assumption is satisfied, the correlation between predictors is very close to zero
print('\n\nCheck correlation between independent variables')
print(round(x_independent_variables.corr(), 2))

# figure 7: pair plot of independent variables
sns.pairplot(x_independent_variables)
# plt.show()
plt.close(fig=None)


"""c. What is the marginal impact of vote average (i.e., quality) on popularity? 
    Is the effect statistically significant?"""
# vote_average has a coefficient of 0.4716
# The marginal impact is a unit increase in vote average would result in a .47 increase in popularity
# It is significant with p-value less than alpha 0.05

"""d. All other things being equal, how does the popularity of movies released in June compare to movies released 
    during the rest of the year?"""
# The June release coefficient (0.19) was not statistically significant (p-value = .638). There is no difference in popularity between movies released
# in June vs the rest of the year.

"""e. What is the impact of how long ago a movie was released on its popularity?"""
# time since movie release date has a marginal impact on popularity, the coefficient was -0.0021, the relationship is inverse
# I would have assumed that older movies had an advantage on popularity because they had more time for people to find them and rate them
# However, the reverse seems to be true, for each additional day since the release date, popularity decreases by .0021
# In a way this makes sense, people are probably more interested in new movies rather than older movies
# In other words, new releases tend to be more popular

"""e. Does this variable have a linear, quadratic, or no impact on popularity? 
    Provide supporting evidence and/or a visualization."""
# I created a new model with the days since the release date squared and the days squared is not significant,
# so days since release date has a linear impact on popularity

# create new variable and assign independent variables to x
regression_movies['days_squared'] = regression_movies['days_since_release'] * regression_movies['days_since_release']
x = regression_movies[['vote_average', 'vote_count', 'days_since_release', 'june_release', 'days_squared']]
# create new regression model with days squared
X2 = sm.add_constant(x)
lm2 = sm.OLS(y_dependent_variable, X2)
print('\nSummary of Second Model for Question 2.e')
print(lm2.fit().summary())
