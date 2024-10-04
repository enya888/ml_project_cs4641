import streamlit as st
import pandas as pd
from PIL import Image

# CSS to set the background image
def add_bg_from_local():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("./static/league_background.jpg");
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

# Worlds Image
# Create two columns
col1, col2 = st.columns(2)

# Load and resize the images
img1 = Image.open('./static/damwon.jpg').resize((400, 300))  # Specify desired width and height
img2 = Image.open('./static/edg.jpeg').resize((400, 300))  # Specify desired width and height
# Place the images in the respective columns
with col1:
    st.image(img1)  # Adjust height as needed

with col2:
    st.image(img2)  # Ensure the same height

# Background Section
st.markdown("<u><h2>Background</h2></u>", unsafe_allow_html=True)
st.markdown("""
_League of Legends_ is a globally popular esports game<sup>[4]</sup>, with a **World Championship (Worlds)** featuring teams from major regions like **South Korea, China, Europe** and **NA**. Current papers have analyzed League with different ML models. (Arık, 2023) studies a LightGBM classifier<sup>[1]</sup>, (Bahrololloomi et al., 2022) utilize a GradBoost-based model<sup>[2]</sup>, and (Shen, 2022) implements a voting classifier<sup>[3]</sup> for results prediction of games based on player and game data. Our models will **predict the Worlds winner** via data from the **Worlds patch (14.18)<sup>[5]</sup>**. First, we will identify meta champions; then, we will analyze **strong players** based on the meta. We will calculate **role impact** to proportionate player strength, **aggregating** to predict team performance at Worlds. Our dataset includes statistics from players and overall games from this year in .csv format.
""", unsafe_allow_html=True)

# Problem Definition Section
st.markdown("<u><h2>Problem Definition</h2></u>", unsafe_allow_html=True)
st.write("""- **Problem**: Esports outcomes are difficult to predict due to evolving metas, player performance, and regional differences.
- **Motivation**: Analyzing the meta, player strengths, and role impact can provide deeper insights into team performance–a valuable asset for teams, fans, and analysts.
""")

# Methods Section
st.markdown("<u><h2>Methods</h2></u>", unsafe_allow_html=True)

st.markdown("<u><h4>Data Preprocessing</h4></u>", unsafe_allow_html=True)
st.write("""- **Data Cleaning**: Remove duplicates, handle missing values, and ensure consistent formatting for champions, roles, and player stats.
- **Feature Encoding**: Use one-hot encoding or label encoding for champions, regions, and roles.
- **Normalization**: Normalize stats like **KDA**, **damage**, and **pick rates** to ensure consistency across datasets.
""")

# ML Algorithms
st.markdown("<u><h4>ML Algorithms/Models</h4></u>", unsafe_allow_html=True)
st.write("""- **Clustering (Unsupervised Learning)**: **K-Means** can cluster players based on various stats, identifying top performers. **K-Means** can also cluster champions frequently picked together, revealing potential synergies and meta insights.
- **K-Nearest Neighbors (KNN)**: **KNN** will predict if a champion is meta based on win rates and pick rates, classifying them based on similar champions. **KNN** will also find similar players in the same role, helping infer role impact by averaging outcomes of comparable players.
- **Linear Regression Model**: **Linear regression** will map player, meta, and champion data to optimized functions, predicting game outcomes. The model’s coefficients will indicate the importance of factors like champion picks and role proficiency in determining team success.
- **Long Short-Term Memory Networks**: **LSTM** networks will model player performance over time, helping capture trends in champion picks, player growth, and meta shifts. This will be crucial for predicting consistency and long-term success throughout the tournament.
""")

# Results and Discussion
st.markdown("<u><h2>(Potential) Results and Discussion</h2></u>", unsafe_allow_html=True)
st.markdown("<u><h4>Classification Metrics</h4></u>", unsafe_allow_html=True)
st.write("""
- **Accuracy and Precision**: Measures how often the model correctly predicts the outcome (whether a champion is meta or not, or the winner of a game) and which predicted outcomes matched.
- **F1-Score**: A balance between precision and recall, useful for binary classification (e.g., predicting if a champion is meta or not). It can be further broken down into macro, micro, or weighted F1-scores based on the task.
- **ROC-AUC**: Uses binary classification to measure the area under the ROC curve, helpful in predicting champions/game outcomes based on probability.
""")

st.markdown("<u><h4>Regression Metrics</h4></u>", unsafe_allow_html=True)
st.write("""
- **R² Score (Coefficient of Determination)**:  Assesses how well meta, player, and game outcome data fit into linear regression models for game predictions.
- **Mean Squared Error (MSE)**: Measures the average squared difference between the predicted and actual values, applicable when predicting game outcomes or player scores.
""")

# # Example of Data Display
# st.markdown("<u><h4>Data Example</h4></u>", unsafe_allow_html=True)

# # Mock data (replace with real data)
# data = {
#     'Player': ['Player A', 'Player B', 'Player C', 'Player D'],
#     'KDA': [5.5, 3.4, 7.2, 6.1],
#     'Pick Rate (%)': [20.3, 25.4, 15.1, 30.2],
#     'Ban Rate (%)': [33.3, 22.8, 76.1, 34.5],
#     'Win Rate (%)': [65.2, 52.4, 70.1, 55.2],
#     'Champion': ["Rengar", "Samira", "Volibear", "Ahri"]
# }

# df = pd.DataFrame(data)
# st.dataframe(df)

# Project Goals Section
st.markdown("<u><h4>Project Goals</h4></u>", unsafe_allow_html=True)
st.write("""
We wish to achieve high accuracy and precision, while considering sustainability by using diverse regional data and ensuring ethical considerations, such as avoiding bias toward specific regions/playstyles. Proficiency in **meta champions** and key roles should impact team success, improving the accuracy of our championship predictions. 
""")

# Expected Results
st.markdown("<u><h4>Expected Results</h4></u>", unsafe_allow_html=True)
st.write("""
- Predicting winners in the World’s 2024 bracket with 80%+  accuracy.
- Gaining additional insight/visualizations into the meta of champions at Worlds and how champions impact team compositions.
- Ranking the teams at Worlds across several metrics and game-states.
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
        - Made the presentation slides and script
        - Did the presentation video''',
        
        '''- Wrote the references section of the proposal
        - Contributed to the proposal topic brainstorming
        - Proofreading
        - Brainstormed the methods to be used for the project
        - Wrote the literature reviews''',
        
        '''- Came up with potential statistics such as KDA to analyze which would help us make our final predictions
        - Wrote problem definition section
        - Wrote the potential results and discussion section
        - Brainstormed the methods to be used for the project
        - Arranged meetings with the TA advisor
        - reviewed slides and script for the presentation'''
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






