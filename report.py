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
            featuring teams from major regions like **South Korea, China, Europe**, and **NA**. 
            
            
            The game revolves around two teams, each consisting of 5 players, who must siege towers and destroy the enemy’s base (“nexus”). 
            Each team is composed of 5 different roles, mostly corresponding the lane they primarily play in at the beginning of the game: 
            the toplaner, midlaner, botlaner (also commonly known as Attack Damage Carry or ADC), support (who plays alongside the glass cannon ADC), 
            and jungler (who resides in the “jungle,” the space between lanes). The players in the lanes are also commonly referred to as simply 
            their lane’s position, such as “top” or “bot.”  It should be noted that although the support and ADC share the bottom lane, for the purposes 
            of this report, the “botlaner” will refer to the ADC.


            Current papers have analyzed League with different ML models. (Arık, 2023) studies a LightGBM classifier<sup>[1]</sup>, (Bahrololloomi et al., 2022) utilizes 
            a GradBoost-based model<sup>[2]</sup>, and (Shen, 2022) implements a voting classifier<sup>[3]</sup> for results prediction 
            of games based on player and game data. 
""", unsafe_allow_html=True)

# Problem Definition Section
st.markdown("<u><h2>Problem Definition</h2></u>", unsafe_allow_html=True)
st.write("""- **Problem**: Esports outcomes are difficult to predict due to evolving metas, player performance, and regional differences.
- **Motivation**: Analyzing what makes a player good at their role and the impact of each role can provide deeper insights into team performance–a valuable asset for teams, fans, and analysts
""")
st.markdown("""
            Our original proposal involved predicting **the Worlds winner** via data from the **Worlds patch (14.18)<sup>[5]</sup>**. 
            First, we wanted to identify meta champions; then, we wanted to analyze **strong players** based on the meta. We wanted to calculate **role impact** to 
            proportionate player strength, **aggregating** to predict team performance at Worlds. 
            
            However, we have decided to move away from this exact approach, as we are uncertain about the feasibility of identifying meta 
            champions (due to restrictions based on the changing nature of the game and available data). Our target shifted to finding “strong players” 
            and role impact with our machine learning algorithms, ultimately using this data to predict the winner of Worlds.

             In our new approach, we decided to first determine which statistics are the most important for each role in helping a team achieve victory, i.e, what features 
             does an ideal player in their respective role need to excel at in order to provide a team with the highest chance of winning. With this information, we could 
             then assign players a performance rating for each of their games depending on how they performed with regards to each important feature.

            We then employed an additional model to find the impact of each role on the outcome of the game, similar to in our original proposal. By using this information, 
            we could then scale our player performances accordingly and help us better ascertain overall team performance.

            Finally to also account for team performance changes over time, we employed a model to predict how each team would perform at worlds given their match history 
            leading up to the tournament. 

            With these 3 models combined, we believed we could achieve a high accuracy in predicting the overall winner of the competition
""", unsafe_allow_html=True)

st.markdown("<u><h2>Dataset</h2></u>", unsafe_allow_html=True)
st.markdown("""
            Our [dataset](https://oracleselixir.com/tools/downloads) includes statistics from players and overall games from this year in .csv format. This dataset 
            is extremely comprehensive, and features data garnered from several sources, including official Riot API and Match History pages. The dataset contained 
            data for all league of legends matches played in the past year, detailing which teams played each other for which match, what date they played, which 
            tournament they played in and also various statistics for each player during the game.
""", unsafe_allow_html=True)

processed_botlane_data = Image.open('./Images/processed_botlane_data.png').resize((700, 600))  
st.image(processed_botlane_data)

# Methods Section
st.markdown("<u><h2>Methods</h2></u>", unsafe_allow_html=True)
st.markdown("<u><h3>Data Preprocessing 1</h3></u>", unsafe_allow_html=True)
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
         
        After all our data was processed, we applied Data Transformation where we split the data into 5 separate unique datasets corresponding to each of the 5 roles. This allowed us to run our 
        models on each role separately and draw conclusions based on each individual role rather than all the players as a whole. Additionally, we could then 
        include additional features for the jungle role to train our models as securing objectives such as dragons and void grubs is exclusively a statistic 
        for junglers and not for other roles.

        Our final datasets for logistic regression in total each had 16 different features which we considered relevant with the Jungle role having an extra 9 for a total of 25 different 
        features.
""")


st.write("""Below is an example subset of our processed data for the botlane role.""")

real_bot = Image.open('./Images/real_bot.png').resize((900, 450))
st.image(real_bot)

# ML Algorithms
st.markdown("<u><h3>Model 1: Logistic Regression</h3></u>", unsafe_allow_html=True)
st.write("""
    The first model we implemented was a logistic regression model to determine the weights of each feature, which was determined as a good choice due to several factors. 
1. Our model will predict whether a player won or lost in a particular game based on player statistics data. This is a binary classification task between win (1) and loss (0) making logistic regression a natural fit for our problem. 
2. Logistic regression provides interpretable coefficients (weights) for each feature. This helps in understanding the impact of each player statistic on the game's outcome. By examining these weights, we can identify which statistics are most important in determining whether a team wins or loses.
3. Additionally, it is computationally efficient, which is advantageous when dealing with our large dataset with many features and data points. This provides quick training and results while preserving performance. 
4. Also, due to our chosen features potentially having simple linear relationships, to avoid overfitting, we decided to pick a simpler model as opposed to a model like Random Forest or GradBoost.
    
To find the impact of each player statistic for each position, we trained a logistic regression model on five separate datasets for each individual lane (top, mid, bot, jungle, support). In this model, we used 80% of the 1900 games for training and the remaining 20% for testing.     

From this, we retrieved the numerical values of each coefficient and graphed them in order to determine the order of importance for these features      
""")

st.markdown("<u><h3>Data Preprocessing 2</h3></u>", unsafe_allow_html=True)
st.write("""
   With our feature weights from Logistic Regression, by applying them to the actual feature values (e.g. the number of kills or number of assists), we summed them up to get a total “score” of a player. This is reflected in a new data spreadsheet for our next model to work off of. 
    
    With these new player performance scores, we created a new dataset with each game on a new row, featuring the team who played (as such, one game will equate to two rows), the date of the game, the result, each player and their respective score, as well as a the sum of these scores for a total team score.
         """)

Data_Preprocessing_2 = Image.open('./final_images/Data_Preprocessing_2.png').resize((900, 450))
st.image(Data_Preprocessing_2)

st.markdown("<u><h3>Model 2: Random Forest</h3></u>", unsafe_allow_html=True)
st.write("""
   The next model we employed with our new dataset was Random Forest. While the relationship between various statistics is likely linear due to the focus being on a singular lane doing their own job, the relationships between roles and how they affect a game could be nonlinear, with roles interacting differently between each other. For this reason, Random Forest was the perfect choice.
    
    To find how impactful each role is to the outcome of a game, we ran a random forest model with the input being the player score values for each game, as aforementioned. Again, 20% of the data was used for testing, and 80% of the data was used for training out of a total 1900 games. 

    To find the importance of each role in determining game result, we once again retrieved the numerical values of each coefficient of our features (the player performances in each role) and graphed them in order to determine the order of importance for these features   
         """)

st.markdown("<u><h3>Data Preprocessing 3</h3></u>", unsafe_allow_html=True)
st.write("""
   With the player scores obtained from the logistic regression model and how impactful each role is obtained from random forest, we can determine the overall impact of a player on their team, and the team’s overall strength. The formula for this is simple: (player_score * role_impact) for each player, summing it to get team strength. 
    
    In the next step, we applied Feature Scaling to better balance each team and to account for teams playing in weaker regions. To do this, we took the Regional Strength Score found on the RIOT Games website, which is computed by AWS, to determine how well a region performs overall on the international stage. The multipliers are computed through analyzing historical team performances from the region across all seasons of the game, from the first ever world championship to present.
    
    The region multipliers are summarized in the table below:
         """)

# Data
data = {
    "Region": [
        "Korea (LCK)",
        "China (LPL)",
        "North America (LCS)",
        "Europe (LEC)",
        "Asia-Pacific (PCS)",
        "Vietnam (VCS)",
        "Latin America (LLA)",
        "Brazil (CBLOL)",
    ],
    "Multiplier": [1873, 1826, 1486, 1542, 863, 839, 583, 634],
}

# Create DataFrame
df = pd.DataFrame(data)

# Streamlit app
st.title("Region Multipliers Table")

# Display table
st.table(df)

st.write("""
   To apply these multipliers, we multiplied our calculated team strengths by the multiplier for each team’s respective region. Before applying the multiplier, however, we scaled all the strengths up by adding a constant to them (magnitude of lowest value in our data) so that there were no longer negative team strengths. This prevented the region multipliers from increasing variance in our data.
    
    We then created our final CSV containing only the dates and adjusted performance scores of each team across each game.
         """)

Data_Preprocessing_3 = Image.open('./final_images/Data_Preprocessing_3.png').resize((300,565)) 
st.image(Data_Preprocessing_3)

st.markdown("<u><h3>Model 3: Long Short-Term Memory</h3></u>", unsafe_allow_html=True)
st.write("""
   As we have the data of every game throughout the year (excluding Worlds), we decided to use an LSTM model to predict the strength of teams on October 14th, which is around the midpoint of the actual Worlds tournament. As an LSTM is able to take in information that happens over a period of time (as memory), it can capture any trends that appear over the time period and use that to predict a point of time in the future. As the performance of teams is generally variable and not all teams are always performing well or poorly, an LSTM is perfect to find what teams will be strong during Worlds.
    
As such, we ran LSTM on each team in order to determine a score for the team during Worlds, which produced a graph for each team and a trendline (with a prediction). Scores were scaled to be between 0 and 1, our batch number was set to 32 and our epoch number was set to 300. With this, we were able to get a prediction, as well as calculate the error in our predictions in the form of MSE and MAPE. 

         """)

st.markdown("<u><h2>Results and Discussion</h2></u>", unsafe_allow_html=True)
st.markdown("<u><h3>Model 1: Logistic Regression</h3></u>", unsafe_allow_html=True)
st.markdown("<h3>Data Visualizations</h3>", unsafe_allow_html=True)

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


st.markdown("<u><h4>Discussion</h4></u>", unsafe_allow_html=True)
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
As our model was run off of a random sample of 80% of the data and tested on the remaining 20% (380 games), we were able to produce accuracy 
         calculations based off of predicting the outcome of the games. For the toplane, the accuracy of our logistic regression model was 0.9536, i.e. 
         it was able to predict the outcome of the game correctly 95.36% of the time. For the jungle, this statistic was 0.9447; for midlane, 0.9316; 
         for ADC, 0.9155; and for support, 0.8941. These are relatively high statistics overall–the combined botlane of ADC and support suffer a bit 
         more in terms of accuracy due to some unique characteristics of the two roles. Supports, especially, are difficult to pin down because of 
         how different things are prioritized, depending on the champion of the support.
    
From sk-learn, we also have precision, recall, and F1 scores. They are all relatively similar, indicating that there is a healthy balance 
between precision and recall, so our model should be finding the true positives with minimal error. """)


st.write("""
    Here are our associated confusion matrices for each role.
""")
confusion_multi = Image.open('./Images/confusion_multi.png').resize((600, 500)) 
confusion_sup = Image.open('./Images/confusion_sup.png').resize((300,265)) 
st.image(confusion_multi)
st.image(confusion_sup)


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

st.markdown("<u><h3>Model 2: Random Forest</h3></u>", unsafe_allow_html=True)
st.markdown("<h4>Data Visualizations</h4>", unsafe_allow_html=True)

Model_2_Random_Forest = Image.open('./final_images/Model_2_Random_Forest.png').resize((500,365)) 
st.image(Model_2_Random_Forest)

st.markdown("<h4>Discussion</h4>", unsafe_allow_html=True)
st.write("""
Our model clearly indicates that jungle is the most important role, followed by top, mid, bot, then support. Jungle has by-far the highest weight, being 0.385, whereas top is 0.251, mid is 0.181, bot is 0.110, and support is 0.072. This is rather reasonable and can be believed to be accurate–junglers set the tempo for the game and are expected to impact every other role’s performance, either through ganking (essentially ambushing the enemy laner), or gaining objectives, such as defeating void grubs or dragons, which can be a great boon to the entire team (for instance, void grubs allow everyone on the team to essentially defeat towers easier and quicker, while defeating enough dragons will produce a permanent buff for the team, aka make the team stronger). Junglers have consistently had a large role in the game and many Worlds Finals were decided by outstanding Jungle performances, such as the championships from 2018-2021.
         
Meanwhile, this year has, for the most part, had a very top-sided meta, due to the nature of toplane champions being very strong as well as the existence and importance of void grubs–as void grubs exist on the topside of the map, if a toplaner is doing better than their opponent, they can disrupt the enemy jungler from being able to take the void grubs and potentially help their own jungler secure them instead. Good toplane players have a notable impact on the game, and there is generally a lot of hype and excitement around the best toplane players due to the role’s potential for flashy and critical plays.
         
Midlaners, on the other hand, are also a crucial entity to a team, but perhaps less so than a toplaner or jungler. A midlaner also has the potential to “roam,” or visit another lane, potentially making an impact across the map and outside of their lane. Since the lane is in the middle of the map, they are able to easily access and join fights over objectives. Botlaners, on the other hand, have noticeably had minimal impact in the metas leading up to worlds. One of the top botlane players in recent years, Gumyausi of T1, likened the role to an egg in instant noodles–without water, noodles, and soup base (the other roles), a botlaner cannot make an impact. However, it can be noted that in hindsight, Worlds featured some crucial performances from botlaners that effectively decided the entire game; as it was impossible for our dataset to take this into account, this could potentially introduce a level of error in our results.
         
It makes sense that the model would output support as the lowest impact role, due to how difficult it is to quantify the contributions of a support–although there are some metrics that can point to the makings of a good support, such as high vision score, many times supports are responsible for starting teamfights and protecting the glass-cannon damage dealers during the fight, which is harder to quantify. Additionally, supports by themselves, in professional play, do not have carry-potential, i.e. a support player cannot single-handedly win a game by themselves. However, they can change the entire course of a game by starting a crucial teamfight for their team that they win, but it is very difficult to quantify this for it is truly a single moment in time that can change the course of a game. Additionally, since our model is based off of logistic regression and it was mentioned earlier how the value of supports may not be reflected accurately in that model, our random forest would be affected a bit as well. Still, overall, support players as individuals likely have the lowest impact on the game, which our model reflects.
         
         """)

st.markdown("<u><h3>Quantitative Metrics and Analysis</h3></u>", unsafe_allow_html=True)
# Data
data = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Value": [0.97, 0.97, 0.97, 0.97],
}

# Create DataFrame
df = pd.DataFrame(data)

# Display table
st.table(df)

st.write("""
As with logistic regression, we were able to produce accuracy, precision, recall, and f-1 score calculations based on training on 80% of the data and testing on 20%. With the feature weights generated by Random Forest, we were able to accurately predict the result of games from player performances 97% of the time. The high accuracy, along with the varied estimated feature weights, reveal that player importance definitely does exist in League of Legends. Specific roles have more impact on the game than others.The precision, recall and F1-score values are all once again very similar (identical to 2 significant figures), meaning our model should be finding true positives with minimal error. 
         
Below the confusion matrix for our model’s predictions:""")

Above_LSTM_Confusion = Image.open('./final_images/Above_LSTM_Confusion.png').resize((300,265)) 
st.image(Above_LSTM_Confusion)

st.write("""
Overall, our Random Forest model performs well with minimal error, similar to logistic regression, likely due to a large number of games and how accurate our logistic regression model was (since we used scores derived from the output of the first model). Our model may not necessarily accurately reflect the impact of supports as explained prior, but overall and in terms of relative impact, our model does a good job of quantifying the impact of a role through the feature coefficients. We have high precision and accuracy values, indicating the high quality of our model.""")


st.markdown("<u><h3>Model 3: LSTM</h3></u>", unsafe_allow_html=True)
st.markdown("<h4>Visualizations</h4>>", unsafe_allow_html=True)
st.write("""
        Sample of end results for 3 of the 20 teams
""")

LSTM_1 = Image.open('./final_images/LSTM_1.png').resize((300,265)) 
st.image(LSTM_1)
LSTM_2 = Image.open('./final_images/LSTM_2.png').resize((300,265)) 
st.image(LSTM_2)
LSTM_3 = Image.open('./final_images/LSTM_3.png').resize((300,265)) 
st.image(LSTM_3)

st.write("""
        Visualization of outputted predicted scores, corresponding to the teams accordingly.
""")

Predicted_Score_v_Team = Image.open('./final_images/Predicted_Score_v_Team.png').resize((300,265)) 
st.image(Predicted_Score_v_Team)

Table_Rank_1 = Image.open('./final_images/Table_Rank_1.png').resize((300,265)) 
st.image(Table_Rank_1)

st.markdown("<h4>Discussion</h4>>", unsafe_allow_html=True)
st.write("""
        The LSTM produced the final results: a ranking of each team at Worlds. In the visualizations section, there is a table with the predicted ranks of each team, as well as the LoL Esports Global Power Rankings powered by AWS produced right before Worlds, and the actual results in the last column. Note that for the AWS Ranking, teams that did not make it to worlds were removed from the ranking to ensure consistency across the table. Additionally, for the actual results, due to the formatting of the tournament, some teams share rankings–for instance, Weibo Gaming and Gen.G share 3rd-4th place since they both lost in the semifinals (to Bilibili Gaming and T1, respectively). As such, those cells have been merged since there is no way of telling which team is 3rd or 4th.

         There are clear discrepancies between the actual results and the predicted results, but there is a less drastic difference between the predicted results and the AWS rankings. The AWS rankings were not able to match up very well to the actual results either, which proves that a ranking of teams may be inherently flawed from the beginning–this logic holds in traditional sports as well, where it is beyond extremely rare for someone to predict a perfect March Madness bracket, even with the best minds and technologies at work.
         
         """)

avg_sdbr = Image.open('./final_images/avg_sdbr.png').resize((700,265)) 
st.image(avg_sdbr)

st.write("""
        We were able to come within 1.2 ranks in comparison to the AWS Rankings, and 2.5 in comparison to Worlds. Although we are unable to ascertain how exactly the AWS Rankings are calculated, our rankings come fairly close to the AWS rankings, implying that either our methodologies are similar or they are different but come to similar conclusions, validating our process. Meanwhile, we have a 2.5 average standard deviation when it comes to the actual Worlds results, which can likely be attributed to the factors discussed above (oversimplification and the variable nature of sports). This means that our model performed a little worse than the AWS Rankings, for the AWS Rankings produced an average standard deviation of 2.429, while our model is 2.5, relative to the actual results of Worlds, so the slight discrepancies between our models results in more inaccuracy for us.
         
        Our model was able to accurately predict the placement of 3 teams–Bilibili Gaming, LNG Esports, and Team Liquid but also had some large mispredictions worth noting. For instance, our model ranked G2 at 8, whereas the AWS ranking put G2 at 5–in all actuality, our model came much closer to the true result, as G2 placed 9th-11th at the actual tournament. Though we are unsure of the actual AWS methodology, they say they value strength of opponent, margin of victory, and context of play (valuing say, international tournaments over regular season games). G2 performed well domestically and had a high finish at the Mid-Season Invitational, the other international event held in May; as such, it makes sense that AWS would rate them highly, whereas our model focused more on actual game performance and did not differentiate between games. While our model worked better in this instance, the average standard deviation in comparison to the actual results for the AWS rankings was 2.429, whereas our model was 2.5, so the AWS model is generally more accurate–it could be that looking at game performance is less accurate than the amalgamation of features used by AWS to determine their power rankings.
        
         Clearly, our model was unable to predict the winner of Worlds accurately. T1 barely made it to Worlds, entering as the last seed from the Korean region and winning their Worlds qualification game in an extremely close best-of-5 that took all 5 games to decide the winner. The team underperformed heavily during the summer, which likely caused our model to rank them lower. In this sense, the AWS rankings agreed, as both our model and AWS did not have T1 in the top 4. Based on pure results throughout the season, Gen.G were favorites to win Worlds, having won the Mid-Season Invitational as well as the spring season, and came 2nd place in the summer season in the Korean league. However, there is superstition that some players on Gen.G underperform at Worlds, perhaps due to the immense amount of pressure or another factor. Meanwhile, there is also a superstition that T1–the team of the most renowned esports player of all time, Faker–steps up and overperforms at Worlds. These are, for the most part, claims made by fans and haters, but there may be a kernel of truth to the superstitions that cannot be reflected by our current models.
         
         While our model predicted FlyQuest to rank around 13, they actually placed 5-8th, getting eliminated in the quarterfinals (having made it to the playoff stage of the competition). Some argue that FlyQuest got “easy” matchups in the round before playoffs, only needing to beat minor region teams and another team from their own region (North America) to make it to playoffs. Additionally, FlyQuest had a rather poor showing at the Mid-Season Invitational, getting knocked out before even making it to the main stage of the competition. To FlyQuest’s credit, they had a strong summer season in North America and finished as the first seed in the region; however, North America has generally been a weaker region, which is reflected in the Regional Strength Score that was used as a scaling factor, as denoted in Data Preprocessing 3. Although they are the 1st seed from North America, due to Western teams being generally weaker than Eastern teams for the past decade and generally not performing well at worlds, FlyQuest making it to playoffs was a shock to many and defied many analyst–and evidently model–predictions.
         
         Another large discrepancy was Weibo Gaming. Though they performed poorly in the spring season in China, they were able to pick it up in the summer, placing 2nd in the Chinese summer playoffs; still, they barely made it to Worlds, as they lost to LNG Esports in the Chinese Regional Gauntlet that decides the 3rd and 4th seeds who make it to Worlds from China. As such, they had to beat JD Gaming and qualified to Worlds as the 4th seed from China. Although both our model and AWS predicted Weibo Gaming to barely miss playoffs (placing 9th), they made it and were eliminated in semifinals, which could potentially be attributed to the team stepping up to the occasion–they even beat LNG Esports to make it to the semifinals, the very team they had lost to previously for the 3rd seed. 
         
         Overall, although our model was not able to correctly predict who would win Worlds 2024, it came relatively close to the AWS rankings, indicating that there is some validity to our model’s results and that the problems may come inherently from a ranking of teams in such a way.
         
         """)
st.markdown("<h4>Quantitative Metrics and Analysis</h4>>", unsafe_allow_html=True)

st.write("""
        Unfortunately, our LSTM model was not as accurate as the other models. This is indicated by both our individual team results/graphs and the final results (a prediction of who will win at worlds). In regards to the former, our model was able to produce MSE and Mean Absolute % Error numbers, shown in the table below.
         
         """)

MSE_Table = Image.open('./final_images/MSE_Table.png').resize((300,265)) 
st.image(MSE_Table)

st.write("""
        Our models were only capable of predicting future values with an error of about 50%, meaning that the prediction of the next value would likely be within plus or minus of 50% of the true value–a very large error margin, especially in comparison to our logistic regression and random forest models.
         
         There are a number of potential explanatory factors. For one, the performance of a team can vary greatly throughout a season. A team that starts off performing very well can experience some troubles later on in the year, or vice versa. As such, our model can become confused, not being able to capture any trends due to the variability of performances game-to-game. Additionally, we are undoubtedly simplifying down League of Legends immensely through assigning scores to each role/player; it is possible that our metric of determining what makes a player good is inherently flawed due to a misunderstanding of the game or perhaps the game’s innate complexity, which would lead to difficulties in predicting performance.
         
         Still, the largest determinant of such a large accuracy error is likely due to the fact that our model relies on an element of consistency or induction; unfortunately, just because a team plays well one day does not mean they will play well on the next day. Just like with any sports predictor, the true results are sometimes very surprising; though strong teams will generally beat weak teams, sometimes the underdog surprises us.
         """)
st.markdown("<h4>Model Comparisons</h4>", unsafe_allow_html=True)
st.write("""
        The first stage is right after our logistic regression model, meaning we only have weights indicating how important each statistic is to winning the game for each role. As such, we multiply each weight by the statistics for each game, creating a total team score. Averaging the team scores across every match produces the final performance rating. 
         
         The second stage is right after our random forest model. Now, our team score is calculated by multiplying a player’s score (calculated by the weights from the logistic regression model) by the role impact weight, and is calculated for every game and then averaged, as it was in the first stage. The final stage is the result of our project, with the predictions from LSTM.
         
         This provided us with team performance ratings:
         
        1. Only taking player performance into account
        2. Taking player performance and role impact into account
        3. Taking player performance, role impact, and change over time into account
         
        The team rankings from each stage are as follows.""")
st.write("""
        Our project was built off of models working off of each other. We used logistic regression to find the factors that make a player good at their role, and used random forest to find how impactful each role is to the outcome of a game. Finally, LSTM worked to predict the strength of teams during the midpoint of when Worlds 2024 is (October 14th), based on its memory of all the games across the season. While our first two models were rather accurate, with high accuracy and precision across the board, the LSTM model was not, with high MSE and Mean Absolute Percentage Error (MAPE). To illustrate the effectiveness of each model, the Regional Strength Score was applied to our summed team performance ratings from each stage in our model.
         
         """)

Stage_1 = Image.open('./final_images/Stage_1.png').resize((300,265)) 
st.image(Stage_1)
Stage_2 = Image.open('./final_images/Stage_2.png').resize((300,265)) 
st.image(Stage_2)
Stage_3 = Image.open('./final_images/Stage_3.png').resize((300,265)) 
st.image(Stage_3)

st.write("""
        A cross-comparison is as follows.""")

cross_table = Image.open('./final_images/cross_table.png').resize((300,265)) 
st.image(cross_table)

st.write("""
        From the rankings from the models above, we can clearly see a difference in the ordering across all 3 models. This means that there is definitely an impact to factoring in role impact and performance over time.
         
         To check for the error of each model, we took the average standard deviation between predicted and real ranks for each of the models regarding both the AWS Rankings and the True results:
         
         """)

Above_next_steps = Image.open('./final_images/Above_next_steps.png').resize((300,265)) 
st.image(Above_next_steps)

st.write("""
        From our error table, it is evident that applying the role impact from random forest made significant improvements to our model whereas the LSTM with the high error rates could not track patterns well enough, and did not change our results in regards to AWS and had a slight increase in inaccuracy when it came to the actual results.""")

# Project Goals Section
st.markdown("<u><h4>Next Steps</h4></u>", unsafe_allow_html=True)
st.write("""
        Moving forward, there are several ways to improve upon this project. It is clear that there may be an inherent problem with a ranking system in order to predict sports results. As such, it may be more beneficial to pursue a different prediction system, such as predicting head-to-head results between two teams. This could involve other ML models, such as SVMs, decision trees, or neural networks. Additionally, we could work on simulating drafts and taking into account the champions that are picked and banned, as well as finding the champion meta. We could also potentially factor in historical results; for instance, T1 has historically performed well at Worlds, so we could try to take that into account (perhaps through neural network layers or similar). Overall, there are several future steps that could be taken to better predict the winner of Worlds.""")

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


final_data = {
    'Name': ['Vansh', 'Joshua', 'Eric', 'Derek', 'Enya'],
    'Final Contributions': [
        '''- Data processing for the LSTM model.
        - Cleaned data by removing unnecessary columns using scripting.
        - Defining LSTM inputs and outputs.
        - Developed the final presentation.
        - Developed video script.''',
        
        '''- Data processing for the LSTM model
        - Defining LSTM inputs and outputs
        - Developed the final presentation.
        - Developed script
        ''',
        
        '''- Developed the Random Forest model
        - Extracted team scores from Random Forest.
        - Developed the final presentation.
        - Streamlit page modifications.''',
        
        '''- Developing the LSTM model.
        - Developing the final presentation.
        - Data processing for the LSTM model.
        - Extracting team rankings from LSTM for final results.''',
        
        '''- Developed the Random Forest model.
        - Developed the final presentation.
        - Writing final report sections.
        - Defined model goals and next steps.
        '''
    ]
}
# Convert to DataFrame
df_final = pd.DataFrame(final_data)

# Replace newlines with <br> to make it HTML-friendly and wrap it in a div with left alignment
df_final['Final Contributions'] = df_final['Final Contributions'].apply(lambda x: f'<div style="text-align: left;">{x.replace("\n", "<br>")}</div>')


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
st.write(df_final.to_html(index=False, escape=False), unsafe_allow_html=True)

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

[6] "Global Power Rankings," LoL Esports, Web Archive, Sep. 19, 2024. [Online]. Available: [https://web.archive.org/web/20240919192541/https://lolesports.com/en-US/gpr](https://web.archive.org/web/20240919192541/https://lolesports.com/en-US/gpr). [Accessed: Dec. 3, 2024].

[7] A. Singh, "T1 Gumayusi's ADC analogy: ‘It’s like eating noodles,’" ONE Esports, Oct. 30, 2023. [Online]. Available: [https://www.oneesports.gg/league-of-legends/t1-gumayusi-adc-analogy-noodle/](https://www.oneesports.gg/league-of-legends/t1-gumayusi-adc-analogy-noodle/). [Accessed: Dec. 3, 2024].
""")




