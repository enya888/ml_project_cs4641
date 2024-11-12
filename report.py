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


# Load and resize the images
img1 = Image.open('./static/damwon.jpg').resize((400, 300))
img2 = Image.open('./static/edg.jpeg').resize((400, 300))  

# Create two columns
col1, col2 = st.columns(2)

# Place the images in the respective columns
with col1:
    st.image(img1)  # Adjust height as needed

with col2:
    st.image(img2)  # Ensure the same height

# Background Section
st.markdown("<u><h2>Background</h2></u>", unsafe_allow_html=True)
st.markdown("""
            _League of Legends_ is a globally popular esports game<sup>[4]</sup>, with a **World Championship (Worlds)** 
            featuring teams from major regions like **South Korea, China, Europe**, and **NA**. Current papers have analyzed League 
            with different ML models. (Arık, 2023) studies a LightGBM classifier<sup>[1]</sup>, (Bahrololloomi et al., 2022) utilizes 
            a GradBoost-based model<sup>[2]</sup>, and (Shen, 2022) implements a voting classifier<sup>[3]</sup> for results prediction 
            of games based on player and game data. 
            
            The game revolves around two teams, each consisting of 5 players, who must siege towers and destroy the enemy’s base (“nexus”). 
            Each team is composed of 5 different roles, mostly corresponding the lane they primarily play in at the beginning of the game: 
            the toplaner, midlaner, botlaner (also commonly known as Attack Damage Carry or ADC), support (who plays alongside the glass cannon ADC), 
            and jungler (who resides in the “jungle,” the space between lanes). The players in the lanes are also commonly referred to as simply 
            their lane’s position, such as “top” or “bot.”

            Our original proposal involved predicting **the Worlds winner** via data from the **Worlds patch (14.18)<sup>[5]</sup>**. 
            First, we wanted to identify meta champions; then, we wanted to analyze **strong players** based on the meta. We wanted to calculate **role impact** to 
            proportionate player strength, **aggregating** to predict team performance at Worlds. 
            
            However, we have decided to move away from this exact approach, as we are uncertain about the feasibility of identifying meta 
            champions (due to restrictions based on the changing nature of the game and available data). Our target now is to find “strong players” 
            and role impact with our machine learning algorithms, ultimately using this data to predict the winner of Worlds.

            Due to the large number of statistics we have for each player and each game played throughout the season, finding **which statistics** are the **most 
            important** for **each role** in League would illuminate **the factors that make a player good at their role**. Based on this information, we can then 
            determine which players are best at their respective positions. 

            We would then like to employ additional models to find the **impact of each role** on the outcome of the game, similar to in our original proposal. 
            With the best players for each position and the impact of their role, we can ascertain overall team strength to predict the winner of Worlds.

            Our [dataset](https://oracleselixir.com/tools/downloads) includes statistics from players and overall games from this year in .csv format. This dataset 
            is extremely comprehensive, and features data garnered from several sources, including official Riot API and Match History pages. The dataset contained 
            data for all league of legends matches played in the past year, detailing which teams played each other for which match, what date they played, which 
            tournament they played in and also various statistics for each player during the game.
""", unsafe_allow_html=True)

processed_botlane_data = Image.open('./Images/processed_botlane_data.png').resize((700, 600))  
st.image(processed_botlane_data)
# Problem Definition Section
st.markdown("<u><h2>Problem Definition</h2></u>", unsafe_allow_html=True)
st.write("""- **Problem**: Esports outcomes are difficult to predict due to evolving metas, player performance, and regional differences.
- **Motivation**: Analyzing what makes a player good at their role and the impact of each role can provide deeper insights into team performance–a valuable asset for teams, fans, and analysts
""")

# Methods Section
st.markdown("<u><h2>Data Preprocessing</h2></u>", unsafe_allow_html=True)
st.write("""
        Several Data Preprocessing methods were implemented for our model to function on proper data, including **Data Cleaning**. Our dataset contained several 
         empty columns and unnecessary information, such as data regarding teams that did not qualify for Worlds in 2024; these were removed. This left us 
         with data for 1900 games. Additionally, we had numerous rows of missing data due to incomplete data from games in China. This included first bloods, 
         as well as data for in-game objectives like rift heralds and dragons per team. We manually filled in this data by watching replays of the games.

         The data for the role of Jungler had to be specifically modified as well, which we accomplished through a script. Some of the features attributed to 
         Junglers were listed under the team data, so we had to shift those specific rows/columns. A sample of our script is as follows:

""")

sample_script = Image.open('./Images/sample_script.png').resize((900, 450))
st.image(sample_script)

st.write("""
         Additionally, we employed Feature Selection. This included deleting features that were unhelpful towards determining individual player performance, 
         such as team-based statistics like team kills per minute and game length.

         We also implemented Feature Engineering, where we created new features to better capture some statistics that would further elucidate player performance 
         for each role, such as KDA (a statistic determined by summing the kills and assists of a player, divided by their number of deaths + 1), and kill 
         participation, which is the total number of kills and assists a player gets divided by the number of kills the team gets in total.

        We had originally planned to use Feature Encoding as well, but found that this was unnecessary as our data was already entirely in numerical format 
        (with no categorical data).
         
        After all our data was processed, we split the data into 5 separate unique datasets corresponding to each of the 5 roles. This allowed us to run our 
        models on each role separately and draw conclusions based on each individual role rather than all the players as a whole. Additionally, we could then 
        include additional features for the jungle role to train our models as securing objectives such as dragons and void grubs is exclusively a statistic 
        for junglers and not for other roles.

        Our final datasets in total each had 16 different features which we considered relevant with the Jungle role having an extra 9 for a total of 25 different 
        features.
""")


st.write("""Below is an example subset of our processed data for the botlane role.""")

real_bot = Image.open('./Images/real_bot.png').resize((900, 450))
st.image(real_bot)

# ML Algorithms
st.markdown("<u><h2>ML Algorithms/Models</h2></u>", unsafe_allow_html=True)
st.markdown("<u><h3>Logistic Regression</h3></u>", unsafe_allow_html=True)
st.write("""
    We implemented a logistic regression model to determine these weights, which was determined as a good choice due to several factors. 
1. Our model will predict whether a player won or lost in a particular game based on player statistics data. This is a binary classification task between win (1) and loss (0) making logistic regression a natural fit for our problem. 
2. Logistic regression provides interpretable coefficients (weights) for each feature. This helps in understanding the impact of each player statistic on the game's outcome. By examining these weights, we can identify which statistics are most important in determining whether a team wins or loses.
3. Logistic regression is computationally efficient, which is advantageous when dealing with our large dataset with many features and data points. This provides quick training and results while preserving performance. 
    
To find the impact of each player statistic for each position, we trained a logistic regression model on five separate datasets for each individual lane (top, mid, bot, jungle, support). In this model, we used 80% of the 1900 games for training and the remaining 20% for testing.          
""")

st.markdown("<u><h3>LSTM</h3></u>", unsafe_allow_html=True)
st.write("""
    In the future, we could potentially implement a Long Short-Term Memory (LSTM) model to track each individual player’s performance over the course of 
         the year and determine their overall expected performance at worlds. With the features determined from our logistic regression model, we can have 
         our LSTM focus on the important player statistics to model the best players at Worlds. We could also customize the attention layer of the LSTM with 
         the weights from the logistic regression model to further guide the LSTM network. Additionally, the weights should help minimize overfitting as 
         the model will focus on the important features already determined. 
""")

st.markdown("<u><h3>Deep Neural Network</h3></u>", unsafe_allow_html=True)
st.write("""
    Through a deep neural network, we can analyze the performance of each player per game in order to determine which roles are the most important to 
         winning–determining role impact. We can stack several connected layers for every input that is role-specific, and integrate its output with 
         concatenation layers in order so the model can learn about what roles most contribute to the success of the team. We plan to implement a model 
         like this in the future to reveal, in conjunction with the other models, what team will win Worlds.
""")

# Results and Discussion
st.markdown("<u><h2>Results and Discussion</h2></u>", unsafe_allow_html=True)

st.markdown("<u><h3>Visualizations</h3></u>", unsafe_allow_html=True)

st.markdown("<u><h4>Logistic Regression</h4></u>", unsafe_allow_html=True)
top_features_logistic = Image.open('./Images/top_features_logistic.png').resize((600, 500))  
jng_features_logistic = Image.open('./Images/jng_features_logistic.png').resize((600, 500))  
mid_features_logistic = Image.open('./Images/mid_features_logistic.png').resize((600, 500))  
bot_features_logistic = Image.open('./Images/bot_features_logistic.png').resize((600, 500))  
sup_features_logistic = Image.open('./Images/sup_features_logistic.png').resize((600, 500))  

#st.image(top_features_logistic) 
#st.image(jng_features_logistic) 
#st.image(mid_features_logistic) 
#st.image(bot_features_logistic)
#st.image(sup_features_logistic)
# Create two columns
col1, col2 = st.columns(2)
# Place the images in the respective columns
with col1:
    st.image(top_features_logistic) 
    st.image(jng_features_logistic) 
    st.image(mid_features_logistic) 
with col2:
    st.image(bot_features_logistic)
    st.image(sup_features_logistic)


st.markdown("<u><h4>Quantitative Metrics</h4></u>", unsafe_allow_html=True)
st.write("""
    From our model, we can clearly see the most impactful features for each role based on the weight, as denoted. These differ depending on the role; 
    for top, jungle, and support, the feature that matters the most is assists, but for mid and bot, kills matter the most. Per the graphs, 
    it is evident that assists are by far the most important factor for supports, with its weight being about 2.8, while top has a weight 
    of right under 2.0 and jungle only having a weight of about 1.8 for assists. Meanwhile, kills matter more for botlaners compared to midlaners, 
    with a weight of about 2.6 to 2.1, respectively.

    We have reason to believe that this is accurate; midlaners and botlaners are often thought to have what is known as “carry-potential,” 
    or the ability to single-handedly win a fight or subsequently, even the entire game. If they are carrying, or in order to carry a game, 
    they would need many kills. As such, kills heavily impacting midlane and botlane performance would be accurate. Meanwhile, toplaners, 
    supports, and junglers often set up the rest of their team for success, so these roles having assists indicate good performance makes 
    sense. The other metrics also line up with what can be reasonably expected from each role.
         
    Some initially surprising features with negative correlation include kill participation and gold share. However, upon deeper analysis, 
         this makes sense. In regards to kill participation, for very one-sided games, kills are usually very high on one team and very low 
         on the other. In games like these, some players on the winning team have a low kill participation as it naturally gets difficult to 
         maintain a high ratio when there is a large amount of total kills. On the other hand, the losing team usually has a higher kill 
         participation as there are not very many kills to begin with, so even having 1 kill or assist could result in a large kill participation. 
         As such, having lower kill participation can very well indicate the team will win the game, so there is a negative correlation.

    Meanwhile, a similar explanation exists for gold share. As gold share is proportional, if one player has low gold share, it can be inferred 
         that gold share is very skewed; as such, the other 4 players would have most of the gold, which can indicate good performance from them. 
         If the gold share is relatively even among the players and your gold share is average, then it means that everyone on the team is performing 
         about the same, so no one really has a big advantage and thus the team as a whole is likely not ahead of their opponents. As such, low gold 
         share can point to better team performance.

    As our model was run off of a random sample of 80% of the data and tested on the remaining 20% (380 games), we were able to produce accuracy 
         calculations based off of predicting the outcome of the games. For the toplane, the accuracy of our logistic regression model was 0.9536, i.e. 
         it was able to predict the outcome of the game correctly 95.36% of the time. For the jungle, this statistic was 0.9447; for midlane, 0.9316; 
         for ADC, 0.9155; and for support, 0.8941. These are relatively high statistics overall–the combined botlane of ADC and support suffer a bit 
         more in terms of accuracy due to some unique characteristics of the two roles. Supports, especially, are difficult to pin down because of 
         how different things are prioritized, depending on the champion of the support.
    
    From sk-learn, we also have precision, recall, and F1 scores. They are all relatively similar, indicating that there is a healthy balance 
         between precision and recall, so our model should be finding the true positives with minimal error. 
    
""")
data = {
    "Role": ["Top", "Jungle", "Mid", "Bot", "Sup"],
    "Accuracy": [0.9536, 0.9447, 0.9316, 0.9155, 0.8941],
    "Precision": [0.9536, 0.9447, 0.9326, 0.9158, 0.8937],
    "Recall": [0.9536, 0.9447, 0.9316, 0.9155, 0.8941],
    "F1 Score": [0.9536, 0.9447, 0.9311, 0.9148, 0.8931]
}
# Convert data to a DataFrame
df = pd.DataFrame(data)
# show table
st.table(df)
st.write("""
    Here are our associated confusion matrices for each role.
""")
confusion_multi = Image.open('./Images/confusion_multi.png').resize((600, 500)) 
confusion_sup = Image.open('./Images/confusion_sup.png').resize((300,265)) 
st.image(confusion_multi)
st.image(confusion_sup)


st.markdown("<u><h4>Model Analysis</h4></u>", unsafe_allow_html=True)
st.write("""
Overall, our model is strong and runs with minimal errors. We were able to garner the features that best correlate to strong performance 
         per position with high accuracy, precision, recall, and F1 score. The model likely performed well due to the sheer number of games 
         we were able to train the model on (almost 2000 games), with each game having detailed statistics. As such, we were able to find 
         which statistics would lead to a win or a loss, per position. Our model did decrease in terms of accuracy, etc. especially in 
         the case of supports, as mentioned earlier, due to the inconsistent nature of what they prioritize. There are a few different 
         “classes” of supports in pro-play that, depending on what they are, will result in different stats. For instance, enchanters 
         are champions that primarily focus on healing or shielding other characters, but are usually very fragile themselves. Meanwhile, 
         engage-supports/tanks are played to soak up damage. As of right now, our model does not differentiate between the types of supports, 
         so it assumes that the champions themselves are monolithic and should prioritize the same features, which creates volatility in the results.""")

# Project Goals Section
st.markdown("<u><h4>Next Steps</h4></u>", unsafe_allow_html=True)
st.write("""
        With the insights we have developed through our models, our next steps are to rank all the players at worlds in order of skill based on how 
         well they perform in their match history with regards to the most impactful statistics. We will do this with our logistic regression model 
         along with our LSTM model to not only determine who the best players are, but also their expected performance at worlds. Once the rankings 
         have been identified, our next goal will be to identify which position on a team is the most impactful i.e for a team to perform well, which 
         role is required to be consistently playing well and which roles do not matter as much to a teams overall performance. With this data, combined 
         with our current models, we will then be able to rank each team at worlds and therefore predict the winner of the tournament.
""")

# Create the data with line breaks (\n)
proposal_data = {
    'Name': ['Vansh', 'Eric', 'Joshua', 'Derek', 'Enya'],
    'Proposal Contributions': [
        '''- Contributed to the proposal topic brainstorming
        - Came up with potential statistics such as KDA to analyze which would help us make our final predictions
        - Found the csv files to use as data
        - Brainstormed the methods to be used for the project
        - Wrote the methods section of the report
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
        - Worked on setting up the Github'''
    ]
}
# Convert to DataFrame
df_proposal = pd.DataFrame(proposal_data)

# Replace newlines with <br> to make it HTML-friendly and wrap it in a div with left alignment
df_proposal['Proposal Contributions'] = df_proposal['Proposal Contributions'].apply(lambda x: f'<div style="text-align: left;">{x.replace("\n", "<br>")}</div>')

midterm_data = {
    'Name': ['Vansh', 'Eric', 'Joshua', 'Derek', 'Enya'],
    'Midterm Contributions': [
        '''- Created a linear regression model to understand what stats contribute the most to winning per role.
        - Cleaned data by removing unnecessary columns using scripting.
        - Brainstormed the methods to be used for the project
        - Created the contribution table for the midterm checkpoint
        - Formulated our inputs and outputs for the models and potential insights.''',
        
        '''- Wrote the midterm report.
        - Tabulated Summer and Spring split LPL data.
        - Assisted in verifying the insights generated by the models and adjusting them.
        - Brainstormed inputs and outputs for the models and useful features.
        ''',
        
        '''- Tabulated Summer and Spring split LPL data.
        - Helped with writing midterm report
        - Cleaned data by removing unnecessary columns using Excel.
        - Formulated our inputs and outputs for the models and potential insights.
        - Assisted in verifying the insights generated by the models and adjusting them.
        - Created the data visualizations''',
        
        '''- Created a logistic regression model to understand what stats contribute the most to winning per role.
        - Assisting  in verifying the insights generated by the models and adjusting them.
        - Identified the ML techniques to be used in this phase.
        - Formulated our inputs and outputs for the models and potential insights.''',
        
        '''- Contributed towards creating the Logistic regression model.
        - Assisted in verifying the insights generated by the models and adjusting the model.
        - Updating the Streamlit with the midterm checkpoint.
        - Formulated our inputs and outputs for the models and potential insights.
        '''
    ]
}
# Convert to DataFrame
df_midterm = pd.DataFrame(midterm_data)

# Replace newlines with <br> to make it HTML-friendly and wrap it in a div with left alignment
df_midterm['Midterm Contributions'] = df_midterm['Midterm Contributions'].apply(lambda x: f'<div style="text-align: left;">{x.replace("\n", "<br>")}</div>')


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
st.write(df_proposal.to_html(index=False, escape=False), unsafe_allow_html=True)
st.write(df_midterm.to_html(index=False, escape=False), unsafe_allow_html=True)

# Set the title of the app
st.markdown("<u><h2>[Click here to view Gantt Chart](https://docs.google.com/spreadsheets/d/1idPuudhL47Bchb-AWMgB4tfxVMzHD75OXCI5U_DXzew/edit?gid=287543152#gid=287543152)</h2></u>", unsafe_allow_html=True)

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




