# Scoutium Player Potential Prediction

<p align="center">
  <img src="https://scoutium.com/_nuxt/img/banner.60ba99f.png" alt="Rating"/>
</p>

---

### Business Problem

Predicting which class (average, highlighted) players are based on the scores given to the characteristics of the footballers tracked by scouts.

---

###  Dataset Story:

The data set consists of information from Scoutium, which includes the features and scores of the football players evaluated by the scouts according to the characteristics of the footballers observed in the matches.

Attributes: It contains the points that the users who evaluate the players give to the characteristics of each player they watch and evaluate in a match. (Independent variables)

Potential_labels: Contains potential tags from users who rate players, with their final opinions about the players in each match. (target variable)

---

### Features:

 Sr. | Feature  | Description |
--- | --- | --- | 
1 | task_response_id | The set of a scout's assessments of all players on a team's roster in a match.| 
2 | match_id | The id of the relevant match. | 
3 | evaluator_id | The id of the evaluator(scout). | 
4 | player_id | The id of the respective player. | 
5 | position_id | The id of the position played by the relevant player in that match. |
6 | analysis_id | A set containing a scout's attribute evaluations of a player in a match. |
7 | attribute_id | The id of each attribute the players were evaluated for. |
8 | attribute_value | The value (points) given to a player's attribute of a scout.|
9 | potential_label | Label indicating the final decision of a scout regarding a player in a match. (target variable) |

---

The dataset is not shared because it's exclusive to Miuul Data Science & Machine Learning Bootcamp.
