import chess.pgn
import numpy as np
from multiprocessing import Pool
import time


#This program generates the datasets which are used in Pandas Queries and the Lichess Machine Learning Programs.
#It takes under 6 minutes for this program to examine 120,000 chess games and all of its positions, each of which
#is stored in a string notation called FEN. 

#You can download the file and run this program yourself by going to: https://database.lichess.org/ and downloading 
#the 2013-January games file in .pgn notation. After doing so, replace the file path to access the file and run the program.



def evalFen(indFen):
    #Character to value mapping for chess pieces
    piece_values = np.zeros(256, dtype=np.int16)


    #remapping the ascii values of the characters to chess piece values within the piece_values array.
    #Only giving these characters their respective integer values allows us to filter out unnecessary characters in the FEN. 
    piece_values[ord('p')] = 1
    piece_values[ord('b')] = 3
    piece_values[ord('n')] = 3
    piece_values[ord('r')] = 5
    piece_values[ord('q')] = 9
    piece_values[ord('k')] = 200
    piece_values[ord('P')] = 1
    piece_values[ord('B')] = 3
    piece_values[ord('N')] = 3
    piece_values[ord('R')] = 5
    piece_values[ord('Q')] = 9
    piece_values[ord('K')] = 200


    #Converts the characters of the chess position into their respective ascii values
    fen_bytes = np.frombuffer(indFen.split()[0].encode(), dtype=np.uint8)


    #Summing all of the piece values up in the position.
    #First retrieves all relevant pieces from the position, then sums their values.
    #The indices of piece_values is the value of the given piece.
    white_points = np.sum(piece_values[fen_bytes[(fen_bytes >= ord('A')) & (fen_bytes <= ord('Z'))]])
    black_points = np.sum(piece_values[fen_bytes[(fen_bytes >= ord('a')) & (fen_bytes <= ord('z'))]])


    points_difference = white_points - black_points
    return points_difference



def result_str_to_num(result_str):
    result_map = {"0-1": -1, "1-0": 1, "1/2-1/2": 0}
    
    #Returns as 2 if not found
    return result_map.get(result_str, 2)  
    

def str_elo_to_int(elo_str):
    try:
        elo_int = int(elo_str)

    except ValueError:
        # Handle the case where conversion fails
        elo_int = 0  # or set it to a default value, e.g., 0
    
    return elo_int


#Gets start positions for groups of chunk_size games in the PGN file."
def get_game_start_positions(pgn_file, chunk_size):
    
    start_positions = []
    with open(pgn_file) as f:
        while True:
            #Saves the current file position
            pos = f.tell()  
            game = chess.pgn.read_game(f)
            if game is None:
                break
            start_positions.append(pos)

            if len(start_positions) >= chunk_size:
                yield start_positions
                start_positions = []

        if start_positions:  # Yield remaining games
            yield start_positions



def lichess_pgn_reader(pgn_file, start_positions, include_short_games, output_file_name):
    """Reads PGN games, processes them, and writes results directly to a file in chunks."""
    
    with open(pgn_file) as f:
        with open(output_file_name + "_str_list.csv", "a") as str_file, open(output_file_name + "_int_array.csv", "a") as int_file:
            for pos in start_positions:
                f.seek(pos)  # Move to the start of the game
                game = chess.pgn.read_game(f)
                if game is None:
                    continue

                current_game = game

                #Saves the sum of the material count throughout the game
                sum_eval_value = 0
                fortieth_fen_eval = None

                #Saves the number of moves made by both sides 
                total_half_moves = 0

                #Processes all the moves of the current chess game
                while current_game.variations:
                    next_node = current_game.variations[0]
                    current_game = next_node


                    try:
                        #Extracts one FEN string from the current game
                        ind_fen = str(current_game.board().fen())

                        #Evaluates the material count of the position
                        fen_eval = evalFen(ind_fen)
                        sum_eval_value += fen_eval


                        #Runs true if the game has reached its 40th move
                        if total_half_moves == 39:
                            fortieth_fen_eval = fen_eval

                    
                    #Skips a game if it can't be read due to an error
                    except ValueError as e:
                        #Skips the game
                        print(f"Skipping game due to error: {e}")
                        break

                    total_half_moves += 1


                if total_half_moves >= 40 or include_short_games:
                    #Write integer values to the int_file
                    int_file.write(f"{result_str_to_num(game.headers.get('Result'))},"
                                   f"{str_elo_to_int(game.headers.get('WhiteElo'))},"
                                   f"{str_elo_to_int(game.headers.get('BlackElo'))},"
                                   f"{str_elo_to_int(game.headers.get('WhiteElo')) - str_elo_to_int(game.headers.get('BlackElo'))},"
                                   f"{fortieth_fen_eval},{sum_eval_value},{total_half_moves}\n")

                    # Write string values to the str_file
                    str_file.write(f"{game.headers.get('UTCDate')},{game.headers.get('White')},"
                                   f"{game.headers.get('Black')},{game.headers.get('Variant')},"
                                   f"{game.headers.get('TimeControl')},{game.headers.get('ECO')},"
                                   f"{game.headers.get('Termination')}\n")



def process_pgn_in_parallel(pgn_file, output_file_name, include_short_games, 
                            chunk_size, num_processes):
    # Get game start positions in chunks
    chunk_positions = list(get_game_start_positions(pgn_file, chunk_size))

    # Create a multiprocessing pool
    with Pool(processes=num_processes) as pool:
        # Use pool.starmap to process each chunk in parallel
        pool.starmap(lichess_pgn_reader, [(pgn_file, chunk, include_short_games, output_file_name) for chunk in chunk_positions])

    print(f"Finished processing PGN file: {pgn_file}")




#Included to clean all these files at once
def generate_pgn_file_paths(start_year, end_year):
    pgn_files = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 2):
            month_str = f"{month:02d}"
            pgn_file = f"C:\\Users\\ryne0\\Downloads\\Raw Lichess Games\\lichess_db_standard_rated_{year}-{month_str}.pgn"
            pgn_files.append(pgn_file)
    return pgn_files



if __name__ == "__main__":
    start_time = time.time()

    # Generate PGN file paths for the range 2013-01 to 2014-12
    pgn_files = generate_pgn_file_paths(2013, 2013)
    
    print(pgn_files)

    for pgn_file in pgn_files:
        print(f"Processing {pgn_file}")
        process_pgn_in_parallel(pgn_file, f"Lichess {pgn_file[-11:-4]}", 
                                include_short_games=False, chunk_size=1000, num_processes=10)

    end_time = time.time()
    runtime_seconds = end_time - start_time
    runtime_minutes = runtime_seconds / 60
    print(f"The function took {runtime_minutes:.4f} minutes to run.")



#Processed 10205437 in 472.4005 minutes, which is 21603.36 games a minute, on average