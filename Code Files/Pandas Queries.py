import pandas as pd
import matplotlib.pyplot as plt
import random



##########

#Note that the plot for player_rating_history may sometimes not display. If this happens, just run the program again

##########



#Returns all the games played by that certain player, and displays their rating by date and by the given time control
#Leave player_name blank if you want a random player to be selected
def player_rating_history(pd_df, player_name, time_control):


    while player_name is None:
        random_player_number = random.randint(1, pd_df.shape[0])

        player_name = pd_df.iloc[random_player_number, 1]

        player_and_time_control = pd_df.loc[(pd_df.iloc[:, 1] == player_name) & 
                                       (pd_df.iloc[:, 4] == time_control)]
        
        #Repicks a different player to sample if they have no or very few games at that time control
        if player_and_time_control.shape[0]<5:
            player_name = None


    #Stores the player's white rating at the appropriate dates
    white_rating_df = pd_df.loc[(pd_df.iloc[:, 1] == player_name) & 
                                       (pd_df.iloc[:, 4] == time_control)].iloc[:, [0, 8]]
    
    
    #Rename columns to make them identical across both black and white dataframes
    white_rating_df.columns = ['Date', 'Rating']

    #Stores the player's black rating at the appropriate dates
    black_rating_df = pd_df.loc[(pd_df.iloc[:, 2] == player_name)
                                       & (pd_df.iloc[:, 4] == time_control)].iloc[:, [0, 9]]
    

    #Rename columns to make them identical across both black and white dataframes
    black_rating_df.columns = ['Date', 'Rating']


    #Stack the ratings and the dates together
    stacked_df = pd.concat([white_rating_df, black_rating_df], axis=0)


    #Sort the df by date to get a continuous series of ratings
    stacked_df = stacked_df.sort_values(by='Date', ascending=True)


    #Groups the data by Date. Necessary to remove multiple ratings at the same date which would distort the graph
    stacked_df = stacked_df.groupby('Date').max()


    stacked_df['Rating'].plot()


    plt.title(f"Rating History of {player_name} for the {time_control} Time Control")
    plt.xlabel("Date")
    plt.ylabel("Rating")
    plt.grid(True)
    plt.show()
    




#set time_control to be None if you want to include all time controls
def win_rate(pd_df, player_name, time_control, group_frequency):

    while player_name is None:
        random_player_number = random.randint(1, pd_df.shape[0])

        player_name = pd_df.iloc[random_player_number, 1]

        player_and_time_control = pd_df.loc[(pd_df.iloc[:, 1] == player_name) & 
                                       (pd_df.iloc[:, 4] == time_control)]
        
        #Repicks a different player to sample if they have no or very few games at that time control
        if player_and_time_control.shape[0]<5:
            player_name = None



    #Runs if no time control is selected
    if time_control is None:
        wins_only = pd_df[(pd_df['Result'] == 1) & ((pd_df['White Name']  == player_name) | (pd_df['Black Name']  == player_name))]
        all_results = pd_df[(pd_df['White Name']  == player_name) | (pd_df['Black Name']  == player_name)]

    #Runs if we want to subset for a particular time control
    else:
        wins_only = pd_df[(pd_df['Result'] == 1) 
                          & ((pd_df['White Name']  == player_name) | (pd_df['Black Name']  == player_name))
                          & (pd_df['Time Control'] == time_control) ]
        all_results = pd_df[((pd_df['White Name']  == player_name) | (pd_df['Black Name']  == player_name)) 
                            & (pd_df['Time Control'] == time_control)]


    #Groups results by day
    wins_only_count = wins_only.groupby(pd.Grouper(key='Date', freq=group_frequency))['Result'].count()
    all_results_count = all_results.groupby(pd.Grouper(key='Date', freq=group_frequency))['Result'].count()


    #Removing rows with zero to avoid ZeroDivisionError
    wins_only_count = wins_only_count[wins_only_count!=0]
    all_results_count = all_results_count[all_results_count!=0]


    win_proportion = wins_only_count/all_results_count

    print(f"Player {player_name} has these total number of wins:  ")
    print(wins_only_count)
    print(f"for the {time_control} time control, and a win rate of ")
    print(win_proportion)
     
    





#Displays a graph of the number of games played every day throughout the month
def num_games_by_date_plot(pd_df):
    num_games_by_date = pd_df['Date'].value_counts()
    num_games_by_date = num_games_by_date.sort_index()

    plt.plot(num_games_by_date.index, num_games_by_date.values)

    plt.xlabel('Date')
    plt.ylabel('Number of Games Played')
    plt.title("Number of Games Played by Date")
    plt.show()



#With this function, we can see that shorter time controls have a greater proportion of time forfeits
def loss_type_by_popular_time_controls(pd_df, num_time_controls_to_display):
    popular_time_controls = pd_df['Time Control'].value_counts().head(num_time_controls_to_display)

    #Subsets only the popular time controls
    popular_df = pd_df[pd_df['Time Control'].isin(popular_time_controls.index)]

    print(popular_df.groupby('Time Control')['Loss Type'].value_counts())





str_df = pd.read_csv("C:/Users/ryne0/Downloads/Python Files/Lichess 2013-01_str_list.csv", header=None)
int_df = pd.read_csv("C:/Users/ryne0/Downloads/Python Files/Lichess 2013-01_int_array.csv", header=None)


game_info_df = pd.concat([str_df, int_df], axis=1)
game_info_df.columns = ['Date', 'White Name', 'Black Name', 'Game Type', 'Time Control', 'Opening', 'Loss Type',
                        'Result', 'White Rating', 'Black Rating', 'Delta Rating', '40th FEN Eval', 
                        'Sum Eval', 'Total Half Moves']
game_info_df['Date'] = pd.to_datetime(game_info_df['Date'])







#Range of acceptable material for a side to be up by. In almost all cases, a game is decided if the material difference
#is outside of this range, unless the ratings of both players are very low. 
fen_range = range(-2,3)
fortieth_fen_pivot_table = game_info_df.pivot_table(values = 'Delta Rating', index = '40th FEN Eval', aggfunc='mean')



#Table 1:

#Prints 40th FEN Eval values ranging from -2 to 2, inclusive.
print(fortieth_fen_pivot_table[fortieth_fen_pivot_table.index.isin(fen_range)])




#Getting the 10 most popular openings
ten_most_popular_openings = game_info_df['Opening'].value_counts().head(10).index


#Filtering the dataframe to only include games that were played as one of the 10 most popular openings
popular_opening_df = game_info_df[game_info_df['Opening'].isin(ten_most_popular_openings)]

#Getting the number of half moves per opening
popular_opening_pivot_table = popular_opening_df.pivot_table(values = 'Total Half Moves', 
                                                       index = 'Opening', aggfunc='mean')


#Table 2:

#Displaying the total number of half moves by opening
print(popular_opening_pivot_table)




highest_rated_player_by_opening = popular_opening_df.pivot_table(values = ['White Rating', 'Black Rating'], 
                                                           index = 'Opening', aggfunc='max')


#Table 3:

#Displaying the maximum player ratings by opening
print(highest_rated_player_by_opening)






#Running the functions made previously here:


#Figure 1:

#Gets the rating history for a random player for the "60+0" time control
player_rating_history(game_info_df, None, "60+0")



#Figure 2:

#Creates a time series graph of the number of games played every day across the website
num_games_by_date_plot(game_info_df)


#Table 4:

#Gets the win rate for a random player at the "60+0" time control and groups the results by week.
win_rate(game_info_df, None, "60+0", "W")



#Table 5:

#Prints the frequency of the type of losses by the 10 most popular different time controls
loss_type_by_popular_time_controls(game_info_df, 10)