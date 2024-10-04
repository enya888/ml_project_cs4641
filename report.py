import streamlit as st
import pandas as pd

# CSS to set the background image
def add_bg_from_local():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("./static/smile.png");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local()

# Set up the Streamlit app
st.title("Predicting League of Legends World Championship Winner using Machine Learning")
st.subheader("A Data-Driven Approach to Champion, Meta, and Role Analysis")
# Create a sub-subheader using Markdown
st.markdown("#### Group members: Vansh, Joshua, Enya, Derek, and Eric")

# Background Section
st.markdown("<u><h2>Background</h2></u>", unsafe_allow_html=True)
st.write("""
League is a globally popular esports game[4], with a World Championship featuring teams 
from major regions like South Korea, China, Europe and NA. Players select from many champions, but few are strong in the current meta. Our models will use data from the Worlds patch (14.18)[5] to identify meta champions and 2024 season data to rank player proficiency. We will calculate role impact to assign player scores, which will be aggregated to predict team performance at Worlds. Our dataset includes statistics from players and overall games from this past year in .csv format.
""")

# Problem Definition Section
st.markdown("<u><h2>Problem Definition</h2></u>", unsafe_allow_html=True)
st.write("""- **Problem**: Esports outcomes, especially in the League of Legends World Championships, are difficult to predict due to evolving metas, player performance, and regional differences.
- **Motivation**: Analyzing the meta, player strengths, and role impact can provide deeper insights into team performance, making it valuable for teams, fans, and analysts.
""")


# Data Preprocessing Section
st.markdown("<u><h2>Data Preprocessing</h2></u>", unsafe_allow_html=True)
st.write("""- **Data Cleaning**: Remove duplicates, handle missing values, and ensure consistent formatting for champions, roles, and player stats.
- **Feature Encoding**: Use one-hot encoding or label encoding for champions, regions, and roles.
- **Normalization**: Normalize stats like KDA, damage, and pick rates to ensure consistency across datasets.
""")

# Clustering Section
st.markdown("<u><h2>Clustering (Unsupervised Learning)</h2></u>", unsafe_allow_html=True)
st.write("""
Use K-Means to cluster players based on champion performance, in-game stats, and role proficiency, identifying top performers. K-Means can also cluster champions frequently picked together, revealing potential synergies and meta insights.
""")

# K-Nearest Neighbors Section
st.markdown("<u><h2>K-Nearest Neighbors (KNN)</h2></u>", unsafe_allow_html=True)
st.write("""
KNN will predict if a champion is meta based on win rates and pick rates, classifying them based on similar champions. KNN will also find similar players in the same role, helping infer role impact by averaging outcomes of comparable players.
""")

# Linear Regression Section
st.markdown("<u><h2>Linear Regression Model</h2></u>", unsafe_allow_html=True)
st.write("""
Linear regression will map player, meta, and champion data to optimized functions, predicting game outcomes. The model’s coefficients will indicate the importance of factors like champion picks and role proficiency in determining team success.
""")

# LSTM Section
st.markdown("<u><h2>Long Short-Term Memory (LSTM) Networks</h2></u>", unsafe_allow_html=True)
st.write("""
LSTM networks will model player performance over time, helping capture trends in champion picks, player growth, and meta shifts. This will be crucial for predicting consistency and long-term success throughout the tournament.
""")

# Metrics Section
st.markdown("<u><h2>Classification Metrics</h2></u>", unsafe_allow_html=True)
st.write("""
- **Accuracy**: Measures how often the model correctly predicts the outcome (whether a champion is meta or not, or the winner of a game).
- **F1-Score**: A balance between precision and recall, useful for binary classification (e.g., predicting if a champion is meta or not). It can be further broken down into macro, micro, or weighted F1-scores based on the task.
- **Precision**: Measures how many predicted positive outcomes (meta champions, game winners) are truly positive.
- **Recall**: Measures how many actual positives (meta champions, winning teams) were identified by the model.
- **ROC-AUC**: Used for binary classification to measure the area under the ROC curve, helpful in predicting champions or game outcomes based on probability estimates.
""")

st.markdown("<u><h2>Regression Metrics</h2></u>", unsafe_allow_html=True)
st.write("""
- **R² Score (Coefficient of Determination)**:  Used to assess how well meta, player, and game outcome data fit into linear regression models for game predictions.
- **Mean Squared Error (MSE)**: Used to measure the average squared difference between the predicted and actual values, applicable when predicting game outcomes or player scores.
""")

# Example of Data Display
st.markdown("<u><h2>Data Example</h2></u>", unsafe_allow_html=True)
st.write("""
Here’s a mock data table for player statistics and team performance:
""")

# Mock data (replace with real data)
data = {
    'Player': ['Player A', 'Player B', 'Player C', 'Player D'],
    'KDA': [5.5, 3.4, 7.2, 6.1],
    'Pick Rate (%)': [20.3, 25.4, 15.1, 30.2],
    'Ban Rate (%)': [33.3, 22.8, 76.1, 34.5],
    'Win Rate (%)': [65.2, 52.4, 70.1, 55.2],
    'Champion': ["Rengar", "Samira", "Volibear", "Ahri"]
}

df = pd.DataFrame(data)
st.dataframe(df)

# Project Goals Section
st.markdown("<u><h2>Project Goals</h2></u>", unsafe_allow_html=True)
st.write("""
Our project goals include achieving high accuracy and precision, while considering sustainability by using diverse regional data and ensuring ethical considerations, such as avoiding bias toward specific regions/playstyles. Proficiency in meta champions and key roles should impact team success, improving the accuracy of our championship predictions.
We are trying to predict match winners based on player statistics and their champion proficiencies, champion synergies.
""")

# Set the title of the app
st.markdown("<u><h2>References</h2></u>", unsafe_allow_html=True)

# Add references using markdown
st.markdown("""
[1] K. Arık, “Machine learning models on MOBA Gaming: League of Legends Winner Prediction,” Acta Infologica, vol. 7, no. 1, pp. 139–151, Jun. 2023. doi:10.26650/acin.1180583  

[2] F. Bahrololloomi, S. Sauer, F. Klonowski, R. Horst, and R. Dörner, “A machine learning based analysis of e-sports player performances in League of Legends for winning prediction based on player roles and performances,” Proceedings of the 17th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications, 2022. doi:10.5220/0010895900003124  

[3] Q. Shen, “A machine learning approach to predict the result of League of Legends,” 2022 International Conference on Machine Learning and Knowledge Engineering (MLKE), pp. 38–45, Feb. 2022. doi:10.1109/mlke55170.2022.00013  

[4] A. Walker, “More people watched League of Legends than the NBA finals,” Kotaku Australia, [https://www.kotaku.com.au/2016/06/more-people-watched-league-of-legends-than-the-nba-finals/](https://www.kotaku.com.au/2016/06/more-people-watched-league-of-legends-than-the-nba-finals/) (accessed Oct. 4, 2024).  

[5] L. Cabreros, “Patch 14.18 notes,” leagueoflegends.com, [https://www.leagueoflegends.com/en-us/news/game-updates/patch-14-18-notes/](https://www.leagueoflegends.com/en-us/news/game-updates/patch-14-18-notes/) (accessed Oct. 4, 2024).
""")

# Create the data with line breaks (\n)
data = {
    'Name': ['Vansh', 'Eric', 'Joshua', 'Derek', 'Enya'],
    'Proposal Contributions': [
        '''- Contributed to the proposal topic brainstorming
        - Came up with potential statistics such as KDA to analyze which would help us make our final predictions
        - Found the csv files to use as data
        - Brainstormed the methods to be used for the project
        - Wrote the methods section
        - Created the slides for the presentation video''',
        
        '''- Brainstormed the methods to be used for the project
        - Came up with potential statistics such as KDA to analyze which would help us make our final predictions
        - Worked on setting up Streamlit''',
        
        '''- Created the Gantt chart and contribution table
        - Brainstormed the methods to be used for the project
        - Came up with potential statistics such as KDA to analyze which would help us make our final predictions
        - Proofreading
        - Did the presentation video''',
        
        '''- Wrote the references section of the proposal
        - Contributed to the proposal topic brainstorming
        - Proofreading
        - Brainstormed the methods to be used for the project''',
        
        '''- Came up with potential statistics such as KDA to analyze which would help us make our final predictions
        - Wrote problem definition section
        - Wrote the potential results and discussion section
        - Brainstormed the methods to be used for the project
        - Arranged meetings with the TA advisor'''
    ]
}
# Convert to DataFrame
df = pd.DataFrame(data)

# Replace newlines with <br> to make it HTML-friendly and wrap it in a div with left alignment
df['Proposal Contributions'] = df['Proposal Contributions'].apply(lambda x: f'<div style="text-align: left;">{x.replace("\n", "<br>")}</div>')

# Inject CSS for aligning table header to the left
st.markdown(
    """
    <style>
    th {
        text-align: left !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the table without indices using HTML rendering
st.markdown("<u><h2>Contributions</h2></u>", unsafe_allow_html=True)
st.write(df.to_html(index=False, escape=False), unsafe_allow_html=True)


# Set the title of the app
st.markdown("<u><h2>[Click here to view Gantt Chart](https://docs.google.com/spreadsheets/d/1idPuudhL47Bchb-AWMgB4tfxVMzHD75OXCI5U_DXzew/edit?gid=287543152#gid=287543152)</h2></u>", unsafe_allow_html=True)





