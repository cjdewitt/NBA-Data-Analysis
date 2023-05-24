from flask import Flask, redirect, render_template, request, url_for, send_file 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from flask import Response 
import io 
from io import BytesIO
import sqlite3 as sl
from sklearn.linear_model import LinearRegression
import base64
app = Flask(__name__)

print('***')
print('Processing data...')
print('***')

# Connects to database
conn = sl.connect('nba.db')
curs = conn.cursor()
curs.execute('DROP TABLE IF EXISTS nba') # Drops the table if it already exists
# Returns a list of all the players in the database who have played at least one season
curs.execute('CREATE TABLE IF NOT EXISTS nba (player TEXT, year INTEGER, pts INTEGER, fg_pct REAL)')

# Loads data into database
df = pd.read_csv('nba.csv')
df = df.fillna(df.mean())

curs.executemany('INSERT INTO nba(player, year, pts, fg_pct) VALUES (?, ?, ?, ?)', 
                     [(df.loc[i, 'Player'], df.loc[i, 'Year'], df.loc[i, 'PTS'], df.loc[i, 'FG%']) for i in range(len(df))])
conn.commit()



# Creates a dictionary to hold the NBA data
player_data = {}
# Creates a dictionary to hold the NBA models
player_model = {}
# Creates a linear regression model for the player's points
model_pts = LinearRegression() 
# Creates a linear regression model for the player's FG%
model_fg_ptg = LinearRegression() 


# Loops through each row of the dataset
for idx, row in df.iterrows():

    # Establishes desired data to be stored in the dictionary
    name = row['Player']
    year = row['Year']
    pts = row['PTS']
    fg_pct = row['FG%']


    # If the player is not in the dictionary, add them
    if name not in player_data:
        player_data[name] = { 'PTS': {year: pts}, 'FG%': {year: fg_pct} } # Creates a dictionary for the player's points and FG% for the given year
        player_model[name] = {'PTS': LinearRegression(), 'FG%': LinearRegression()} # Creates a dictionary for the player's points and FG% models   

        # Tains the models for the player's points and FG% for given year
        X = [row['FG%'], row['3P%'], row['FT%'], row['G']]
        player_model[name]['PTS'].fit([X], [pts]) # Fits the model to the data
        player_model[name]['FG%'].fit([X], [fg_pct]) # Fits the model to the data


    # If the player is in the dictionary, update their data
    else:
        player_data[name]['PTS'][year] = pts # Adds the player's points for the given year to the dictionary
        player_data[name]['FG%'][year] = fg_pct # Adds the player's FG% for the given year to the dictionary



@app.route('/')
def home():
    # A dictionary that maps each graph type to its corresponding name in the static directory
    types = { '(1)Total Season Points Over Time': 'histogram1', '(2)Projected Season Points': 'histogram2', '(3)Field Goal Percentage Over Time': 'bar chart1', '(4)Projected Field Goal Percentage': 'bar chart2'}

    return render_template('home.html', players=db_get_players(), graph=types, message="Analyze any NBA player since 1993.") # Renders the home.html template and passes in the options dictionary )


@app.route('/analyze_player', methods=['POST'])
def analyze_player():
    
    # Extracts the selected player and graph type from the form submission
    selected_player = request.form['player']
    selected_graph = request.form['graph']
    
    # Calls the create_graph function with the appropriate parameters based on the selected graph type
    if selected_graph == '(1)Total Season Points Over Time':
        fig = create_graph(selected_player, 'histogram1')
    elif selected_graph == '(2)Projected Season Points':
        fig = create_graph(selected_player, 'histogram2')
    elif selected_graph == '(3)Field Goal Percentage Over Time':
        fig = create_graph(selected_player, 'bar chart1')
    elif selected_graph == '(4)Projected Field Goal Percentage':
        fig = create_graph(selected_player, 'bar chart2')

    # Converts the figure to a base64 encoded string and passes it to the analyze_player.html template
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    fig_data = base64.b64encode(img.getvalue()).decode()


    return render_template('analyze_player.html', fig=fig_data, selected_player=selected_player) # Renders the analyze_player.html template and passes in the selected player and visual )


def db_get_players():
    results = []
    for player in player_data.keys():
        if player not in results:
            results.append(player)
    return results


# This function takes a player and a dictionary of data containing their points and 
# field goal percentage and predicts what their points and FG% will be in a hypothetical year
def predict(player, data):
    pts_prediction = player_model[player]['PTS'].predict([data])
    fg_prediction = player_model[player]['FG%'].predict([data])

    return pts_prediction, fg_prediction


# This route is used to generate and return an image of a player's graph in the specified type
@app.route("/fig/<data_request>/<locale>")
def fig(player, type):
    fig = create_graph(player, type)
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype="image/png")


# This function takes a player's name and a graph type and creates a graph of the 
# specified type for the player's points or field goal percentage over time
def create_graph(player, type):
    # Get the player's points and field goal percentage data
    pts_data = player_data[player]['PTS']
    fg_data = player_data[player]['FG%']

    # Check which graph type was selected
    if type == 'histogram1':
        # Create a histogram of the player's points per season over their given time period
        fig = Figure() 
        ax = fig.subplots()
        ax.set_title(f"{player}'s Season Points")
        ax.set_xlabel('Year')
        ax.set_ylabel('Points')
        ax.hist(list(pts_data.keys()), weights=list(pts_data.values()), bins=20)
        return fig

    elif type == 'histogram2':
        # Predict the player's points for a hypothetical year
        X = [list(fg_data.values())[0], 0, 0, 0]
        pts_pred, _ = predict(player, X)

        # Create a histogram of the player's points per season over their given time period, with a predicted point for the hypothetical year
        fig = Figure() 
        ax = fig.subplots()
        ax.set_title(f"{player}'s Projected Season Points")
        ax.set_xlabel('Year')
        ax.set_ylabel('Points')
        ax.hist(list(pts_data.keys()), weights=list(pts_data.values()), bins=20)
        ax.axhline(y=pts_pred, color='r', linestyle='-')
        return fig
        

    elif type == 'bar chart1':
        # Create a bar chart of the player's field goal percentage per season over their given time period
        fig = Figure() 
        ax = fig.subplots()
        ax.set_title(f"{player}'s Field Goal Percentage")
        ax.set_xlabel('Year')
        ax.set_ylabel('Field Goal Percentage')
        ax.bar(list(fg_data.keys()), list(fg_data.values()), align='center')
        return fig

    elif type == 'bar chart2':
        # Predict the player's field goal percentage for a hypothetical year
        X = [list(fg_data.values())[0], 0, 0, 0]
        _, fg_pred = predict(player, X)

        # Create a bar chart of the player's field goal percentage per season over their given time period, with a predicted point for the hypothetical year
        fig = Figure() 
        ax = fig.subplots()
        ax.set_title(f"{player}'s Projected Field Goal Percentage")
        ax.set_xlabel('Year')
        ax.set_ylabel('Field Goal Percentage')
        ax.bar(list(fg_data.keys()), list(fg_data.values()), align='center')
        ax.axhline(y=fg_pred, color='r', linestyle='-')
        return fig

    




if __name__ == '__main__':
    app.run(debug=True)
