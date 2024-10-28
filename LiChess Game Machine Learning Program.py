import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC




#Call this program after running the LiChess Multiprocessing file for cleaning and preparing a file of chess games

#It can be seen that the errors for the model results are high, which is due to multiple confounding variables, some
#of which are the short time control, cheating and players caring less about the game outcome because it is online.


#Future improvements would be to generate new predictor variables to improve our models, and to use a reliable dataset
#of in-person games, which would eliminate the confounding variables of cheating and players' lack of care for the game's
#outcome.











#Need this function because running the number of black bins may give an error for when our dataset is quite small.

#The except chunk won't be called when the dataset is quite large, because we will always be able to correctly bin the
#data, provided the number of bins is not extremely large
def separate_black_elos(elo_pd_series, num_bins):
    try:
        binned_data_black, black_elo_boundaries = pd.qcut(elo_pd_series, q=num_bins, retbins=True)
        return binned_data_black, black_elo_boundaries
    except ValueError:

        if num_bins > 1:  # Prevent infinite recursion
            print("ValueError arose, trying again with "+str(num_bins-1)+" number of bins")
            return separate_black_elos(elo_pd_series, num_bins-1)
        else:
            raise ValueError("Cannot bin the data into fewer bins.")
        



#Separates our array into different grids based on the white and black players' ratings
#Also returns the boundaries used to separate our data into grids
def array_separation(pd_df, num_bins_white, num_bins_black):

    # Bin the white player ratings and get the boundaries
    white_rating_pd_series = pd_df['White Rating']
    binned_data, white_rating_boundaries = pd.qcut(white_rating_pd_series, q=num_bins_white, retbins=True)


    #Initializing the white sections and black subsections within those white sections
    pd_df['White Section'] = binned_data.cat.codes
    pd_df['Black Subsection'] = 0


    black_rating_boundaries_list = []

    # For each white section bin, separate black ratings into their respective bins
    for bin_number in range(num_bins_white):
        # Filter the rows for the current white section bin
        white_bin_rows = pd_df[pd_df['White Section'] == bin_number]


        # Extract the black ratings for the current white section
        black_elo_pd_series = white_bin_rows['Black Rating']

        # Bin the black players within this white section
        binned_data_black, black_rating_boundaries = separate_black_elos(black_elo_pd_series, num_bins_black)


        # Store black rating boundaries for each white section
        black_rating_boundaries_list.append(list(black_rating_boundaries))
        pd_df.loc[white_bin_rows.index, 'Black Subsection'] = binned_data_black.cat.codes

    return pd_df, white_rating_boundaries, black_rating_boundaries_list





#Rename the variables in this function
def plot_points_only(pd_df):
    white_ratings = pd_df['White Rating'].to_numpy()  
    black_ratings = pd_df['Black Rating'].to_numpy()

    # Create a scatter plot
    plt.scatter(white_ratings, black_ratings)
    
    # Add labels and title
    plt.xlabel('White Elos')
    plt.ylabel('Black Elos')
    plt.title('Plot of the White and Black Elos')
    
    # Show the plot
    plt.show()




#Rename the variables in this function
def plot_white_boundaries_only(pd_df, white_elo_boundaries):
    # Extract the 'White Ratings' and 'Black Ratings' columns from the DataFrame
    white_ratings = pd_df['White Rating'].to_numpy()  
    black_ratings = pd_df['Black Rating'].to_numpy()  
    
    # Create a list of colors for each point
    colors = []
    
    # Get the colormap
    cmap = plt.get_cmap('tab10')  # Use the new Matplotlib API
    
    # Iterate through the white ratings (x values) and assign colors based on white_elo_boundaries
    for i in range(len(white_ratings)):
        color_assigned = 'gray'  # default color if no boundary matches
        # Iterate through the boundaries using two consecutive elements to form the range
        for j in range(len(white_elo_boundaries) - 1):
            boundary_min = white_elo_boundaries[j]
            boundary_max = white_elo_boundaries[j + 1]
            if boundary_min <= white_ratings[i] <= boundary_max:
                # Assign a different color for each boundary
                color_assigned = cmap(j % 10)  # mod to cycle through colors
                break
        colors.append(color_assigned)
    
    # Create a scatter plot with the assigned colors
    plt.scatter(white_ratings, black_ratings, c=colors)
    
    # Add labels and title
    plt.xlabel('White Ratings')
    plt.ylabel('Black Ratings')
    plt.title('Points Colored Based on White Elo Boundaries')
    
    # Show the plot
    plt.show()



#Rename the variables in this function
#Work on this later. it is not quite finished yet. the whole plot will look like one big chunk of 4 colors when outputted
def plot_elo_boundaries(pd_df, white_elo_boundaries, black_elo_boundaries):

    
    white_ratings = pd_df['White Rating'].to_numpy()  
    black_ratings = pd_df['Black Rating'].to_numpy()  
    
    # Create a list of colors for each point
    colors = []
    
    # Get the colormap (can adjust if needed, but 'tab20' is used here for simplicity)
    cmap = plt.colormaps['tab20']  # Use a colormap with enough color variation
    
    # Iterate through the x values and assign colors based on white_elo_boundaries and black_elo_boundaries
    for i in range(len(white_ratings)):
        color_assigned = 'gray'  # default color if no boundary matches

        # Iterate through the white boundaries first
        for j in range(len(white_elo_boundaries) - 1):
            white_min = white_elo_boundaries[j]
            white_max = white_elo_boundaries[j + 1]
            
            if white_min <= white_ratings[i] <= white_max:

                # Now use the corresponding black boundaries for this white boundary
                black_bounds = black_elo_boundaries[j]

                # Assign a unique color per white + black sub-boundary combination
                for k in range(len(black_bounds) - 1):
                    black_min = black_bounds[k]
                    black_max = black_bounds[k + 1]
                    
                    if black_min <= black_ratings[i] <= black_max:
                        # Combine the white boundary index (j) and black boundary index (k) to create a unique color index
                        unique_color_index = j * len(black_bounds) + k
                        color_assigned = cmap(unique_color_index % 20)  # Use the unique index to pick the color
                        break
                break
        colors.append(color_assigned)
    
    # Create a scatter plot with the assigned colors
    plt.scatter(white_ratings, black_ratings, c=colors)
    
    # Add labels and title
    plt.xlabel('White Ratings')
    plt.ylabel('Black Ratings')
    plt.title('Points Colored Based on White and Black Elo Boundaries')
    
    # Show the plot
    plt.show()




def get_best_svm_model(pd_df, cost_values, chosen_columns, include_draws):

    model_list = []
    errors = []
    cost_used = []  # List to store cost values

    # Define results based on whether to include draws or not
    results = pd.Categorical(pd_df['Result'], categories=[-1, 0, 1] if include_draws else [-1, 1], ordered=True)

    # Loop over the different C values for SVM
    for c in cost_values:
        # Initialize the SVM model with varying C parameter
        model = SVC(C=c, random_state=1000, probability=True)
        
        # Perform cross-validation and calculate the mean error using negative log loss
        error = -np.mean(cross_val_score(model, pd_df.loc[:, chosen_columns], results, cv=3, scoring='neg_log_loss'))
        
        model_list.append(model)
        errors.append(error)
        cost_used.append(c)  # Store the corresponding cost value

    # Select the model with the minimum error
    best_model_index = np.argmin(errors)
    best_model = model_list[best_model_index]
    best_cost = cost_used[best_model_index]  # Retrieve the best cost value

    # Train the best model on the full dataset
    best_model.fit(pd_df.loc[:, chosen_columns], results)

    # Retrieve the error of the best model
    best_error = errors[best_model_index]

    
    return best_model, best_cost, best_error




#Have to make a try statement for when there are no draws within a dataset, but we want to include draws
def get_best_rf_model(pd_df, trees, chosen_columns, include_draws):
    
    results = pd.Categorical(pd_df['Result'], categories=[-1, 0, 1] if include_draws else [-1, 1], ordered=True)

    
    model_list = []
    errors = []

    # Loop over the different tree counts
    for tree_count in trees:
        # Initialize the RandomForest model
        rf = RandomForestClassifier(n_estimators=tree_count, random_state=1000, class_weight='balanced')

        # Perform cross-validation and calculate error (using negative log loss for error)
        error = -np.mean(cross_val_score(rf, pd_df.loc[:, chosen_columns], results, cv=3, scoring='neg_log_loss'))
        
        # Append model and errors to lists
        model_list.append(rf)
        errors.append(error)


    # Select the model with the minimum error
    best_model_index = np.argmin(errors)
    best_model = model_list[best_model_index]


    # Train the best model on the full dataset
    best_model.fit(pd_df.loc[:, chosen_columns], results)


    # Retrieve the first tree used in the best random forest model
    best_tree = best_model.n_estimators

    # Retrieve the error of the best model
    best_error = errors[best_model_index]
    
    return best_model, best_tree, best_error



#Returns black win, draw if included, and white win prediction accuracy 

#Will calculate the percentage accuracy for a model's predictions
def percent_accuracy(predictions, pd_df, include_draws):
    
    if include_draws:
        available_outcomes = [-1, 0, 1]
    else:
        available_outcomes = [-1, 1]

    # Entries will be win, draw, and loss percentage prediction accuracy
    # If only win and loss outcome, then entries will be win and loss percentage prediction accuracy
    percentage_accuracy_list = []

    for outcome in available_outcomes:
        # Find indices where the predicted outcome matches the current outcome
        prediction_indices = np.where(predictions == outcome)[0]
        total_outcome_predictions = len(prediction_indices)


        # If there are no predictions for this outcome, append 0 to accuracy list
        if total_outcome_predictions == 0:
            percentage_accuracy_list.append(0)

        else:
            # Extract the actual results from pd_df based on prediction indices
            actual_results = pd_df.iloc[prediction_indices]['Result'].values


            total_correct_outcome_predictions = np.sum(actual_results == outcome)

            # Calculate the percentage accuracy for the current outcome
            accuracy = (total_correct_outcome_predictions / total_outcome_predictions) * 100
            percentage_accuracy_list.append(accuracy)

    return percentage_accuracy_list





#Could make a while loop using the below function to keep repicking training and testing data until we
#are able to predict for all outcomes

#Checks that there is at least one entry of each outcome within both the training and testing data. 
#needed when our dataset is small, or when we use many bins
def verify_valid_training_testing_data(pd_df, include_draw):
    
    #One for training one for testing
    for train_or_test in range(2):

        subgrid_array = pd_df[(pd_df['TrainTest Indicators'] == train_or_test)]['Result'].to_numpy()


        #Runs true if we are to include draws in the predictions
        if include_draw:

            #Checking that there is at least one win, one draw and one loss in the dataset
            for result_int in list((-1,0,1)):

                #Runs true if there isn't at least one instance of the desired outcome in the data
                if (np.where(subgrid_array==result_int)[0].size==0):

                    #means that the training_testing data for that subgrid is not valid
                    return False
        

        #int_array will be called after we removed draws at the beginning of the run_models function
        else:
            #Iterating through the possible outcomes for the dataset with no draws
            for result_int in list((-1,1)):

                #Runs true if there isn't at least one instance of the desired outcome in the data
                if (np.where(subgrid_array==result_int)[0].size==0):

                    #means that the training_testing data for that subgrid is not valid
                    return False

    #All the training and testing data have at least one of each outcome, so our dataset is valid
    return True





#num_trees and cost should be lists
#include_draws and use_rf_model should be booleans
#chosen_columns are the columns we want to run for prediction purposes

#best_parameter_value in the function can either be the best num_trees or best cost found
def run_models(pd_df, include_draws, use_rf_model, num_trees, cost, chosen_columns):
    pd_df = pd_df.copy()


    # Holds the total number of white and black grid values subtracted by 1.
    # These variables are necessary to correctly analyze our data.
    num_white_grid_values = len(np.unique(pd_df['White Section']))
    num_black_grid_values = len(np.unique(pd_df['Black Subsection']))
    num_rows_model_result_summary = num_white_grid_values * num_black_grid_values

    # Remove draws from the data if not including draws
    if not include_draws:
        pd_df = pd_df[pd_df['Result'] != 0]

    # Initialize the model_result_summary array
    if include_draws:
        model_result_summary = np.zeros(shape=(num_rows_model_result_summary, 7), dtype=np.float16)
    else:
        model_result_summary = np.zeros(shape=(num_rows_model_result_summary, 6), dtype=np.float16)

    # Generate random indicators for training (70%) and testing (30%)
    random_numbers = random.choices([0, 1], weights=[0.7, 0.3], k=pd_df.shape[0])
    pd_df.loc[:,'TrainTest Indicators'] = np.array(random_numbers)

    current_iteration = 0

    # Iterate through all the white and black grid values for analysis
    for white_grid_value in range(num_white_grid_values):
        for black_grid_value in range(num_black_grid_values):

            # Subset the data for the current white and black grid value
            pd_df_subset = pd_df[(pd_df['White Section'] == white_grid_value) & 
                                 (pd_df['Black Subsection'] == black_grid_value)]

            # Verify if the subset data is valid for training and testing
            if verify_valid_training_testing_data(pd_df_subset, include_draws):

                # Split the subset into training and testing sets
                pd_df_subset_train = pd_df_subset[pd_df_subset['TrainTest Indicators'] == 0]
                pd_df_subset_test = pd_df_subset[pd_df_subset['TrainTest Indicators'] == 1]

                # Run the chosen model (random forest or SVM)
                if use_rf_model:
                    best_model, best_parameter_value, best_error = get_best_rf_model(
                        pd_df_subset_train, num_trees, chosen_columns, include_draws)
                    
                elif use_rf_model==False:
                    best_model, best_parameter_value, best_error = get_best_svm_model(
                        pd_df_subset_train, cost, chosen_columns, include_draws)

                # Make predictions on the test set
                prediction = best_model.predict(pd_df_subset_test[chosen_columns])
                accuracy = percent_accuracy(prediction, pd_df, include_draws)


            else:
                # If the subset data is not valid, set default values
                if include_draws:
                    accuracy = [0, 0, 0]
                else:
                    accuracy = [0, 0]
                best_parameter_value = 0
                best_error = 0


            # Store the results in the summary array
            if include_draws:
                model_result_summary[current_iteration, :] = np.array([
                    accuracy[0], accuracy[1], accuracy[2], best_parameter_value,
                    best_error, white_grid_value, black_grid_value])
            else:
                model_result_summary[current_iteration, :] = np.array([
                    accuracy[0], accuracy[1], best_parameter_value, best_error,
                    white_grid_value, black_grid_value])


            current_iteration += 1

    return model_result_summary

    




#use for querying pandas table
#randomly selects a player to do machine learning on for their games
def random_player_games(pd_string_df):
    num_rows = pd_string_df.shape[0]

    rand_color_col = random.randint(1, 2)


    #selects a random player to subset the pandas array
    random_player_name_index = random.randint(0, num_rows-1)


    random_player_name = pd_string_df.loc[random_player_name_index, rand_color_col]


    return pd_string_df.loc[pd_string_df.iloc[:,rand_color_col] == random_player_name, :]



#Plot the grid of the model errors
def plot_model_errors_with_draws(model_results_df, is_rf_model, include_draws, num_black_subgrids):
    # Determine the plot title based on the model type and inclusion of draws
    if is_rf_model:
        plot_title_name = "Results for the Random Forest Model with Draws" if include_draws else "Results for the Random Forest Model without Draws"
    else:
        plot_title_name = "Results for the SVM Model with Draws" if include_draws else "Results for the SVM Model without Draws"

    # Assign column names based on whether draws are included
    if include_draws:
        model_results_df.columns = ['Black Win Accuracy', 'Draw Accuracy', 'White Win Accuracy', 
                                    'Parameter Used', 'Error', 'White Grid', 'Black Subgrids']
    else:
        model_results_df.columns = ['Black Win Accuracy', 'White Win Accuracy', 
                                    'Parameter Used', 'Error', 'White Grid', 'Black Subgrids']

    # Group by 'White Grid' (5th column)
    grouped = model_results_df.groupby('White Grid')

    # Plotting the 'Error' values for each group
    for white_grid_number, group_data in grouped:
        

        # Check if the group has exactly the expected number of subgrids (10)
        if len(group_data) == num_black_subgrids:
            black_subgrid_values = np.arange(1, num_black_subgrids + 1)  # X-axis from 1 to 10
            group_data_sorted = group_data.sort_values(by='Black Subgrids')  # Ensure the subgrids are in order


            #Converting white grid number from float to int for polishing the legend
            white_grid_number = int(white_grid_number)


            # Plot the 'Error' values against black_subgrid_values
            plt.plot(black_subgrid_values, group_data_sorted['Error'].values, label=f'White Grid {white_grid_number+1}')
        else:
            print(f"Skipping group {white_grid_number} because it has {len(group_data)} black subgrids (expected {num_black_subgrids}).")

    # Add labels, title, and legend
    plt.xlabel('Black Subgrids (1-10)')
    plt.ylabel('Error Values')
    plt.title(plot_title_name)
    plt.legend(title="White Grids")
    plt.show()

            


#Prints sample rows from the results dataframe.
def sample_results(model_results_df, include_draws, is_rf_model, num_rows_to_sample):

    if include_draws:
        model_results_df.columns = ['Black Win Accuracy', 'Draw Accuracy', 'White Win Accuracy', 
                                    'Cost Used', 'Error', 'White Grid', 'Black Subgrids']
    else:
        model_results_df.columns = ['Black Win Accuracy', 'White Win Accuracy', 
                                    'Cost Used', 'Error', 'White Grid', 'Black Subgrids']
    

    if is_rf_model:
        model_results_df = model_results_df.rename(columns={'Cost Used': 'Tree Used'})


    random_numbers = [random.randint(0, model_results_df.shape[0]-1) for _ in range(num_rows_to_sample)]
    model_df_subset = model_results_df.iloc[random_numbers]

    model_df_subset.index.name = "Row"
    model_df_subset = model_df_subset.sort_index()
    print(model_df_subset.iloc[:,:-2])







str_df = pd.read_csv("C:\\Users\\ryne0\\Downloads\\Python Files\\Lichess 2013-01_str_list.csv", dtype=str)
int_df = pd.read_csv("C:\\Users\\ryne0\\Downloads\\Python Files\\Lichess 2013-01_int_array.csv", delimiter=',')
int_df = int_df.astype('int64')


game_info_df = pd.concat([str_df, int_df], axis=1)
game_info_df.columns = ['Date', 'White Name', 'Black Name', 'Game Type', 'Time Control', 'Opening', 'Loss Type',
                        'Result', 'White Rating', 'Black Rating', 'Delta Rating', '40th FEN Eval', 
                        'Sum Eval', 'Total Half Moves']
game_info_df['Date'] = pd.to_datetime(game_info_df['Date'])



#Removing the instances where one or both players has a rating of 0
#Necessary for predicting correctly, and to prevent distortion of plots
game_info_df = game_info_df[(game_info_df['White Rating']!=0) & (game_info_df['Black Rating']!=0)]



#Setting up the original model, where we did not account for any confounders other than removing the 0 ratings. 
game_info_df, white_elo_boundaries, black_elo_boundaries_list = array_separation(game_info_df, 10, 10)


plot_points_only(game_info_df)
plot_white_boundaries_only(game_info_df, white_elo_boundaries)
plot_elo_boundaries(game_info_df, white_elo_boundaries, black_elo_boundaries_list)



include_draws2 = False; use_rf_model2 = True; num_trees2 = list(range(100, 151)); cost2 = list((1, 10, 50, 100, 150)) 
chosen_columns2 = list(('Delta Rating', '40th FEN Eval', 'Sum Eval', 'Total Half Moves'))



#rf_model_summary_array = run_models(game_info_df, include_draws2, use_rf_model2, num_trees2, cost2, chosen_columns2)



#svm_model_summary_array = run_models(game_info_df, include_draws2, False, num_trees2, cost2, chosen_columns2)

#np.savetxt("rf_model_summary_array.csv", rf_model_summary_array, delimiter=",", fmt='%.4f')
#np.savetxt("svm_model_summary_array.csv", svm_model_summary_array, delimiter=",", fmt='%.4f')





svm_results = pd.read_csv("C:\\Users\\ryne0\\Downloads\\svm_model_summary_array.csv", dtype=float, header = None)

plot_model_errors_with_draws(svm_results, False, False, 10)
sample_results(svm_results, False, False, 10)



