The system checks out only the complete transcript that users write and submit on their own. It skips any content the app generates automatically. Once you send in your text, the scoring engine kicks off right away. It wraps up the entire evaluation in about 52 seconds. The model relies on a hybrid approach for scoring. That mixes rule-based elements, such as detecting keywords, matching required phrases, and validating word counts. It also brings in NLP methods for semantic similarity. Those use sentence-embedding tools to line up the users transcript against the rubric details. When the process ends, the system puts out an overall score. It includes scores for each criterion too. Plus some notes on ways to improve. Users get more clarity this way. They can download a JSON file for transparency. That file holds the full scoring details. It lists detected keywords, similarity numbers, and every intermediate step in the process.

FORMULA USED:
1.Rule-Based Score
Rule_Score = 
  Salutation_Score
+ MustHave_Score
+ GoodToHave_Score
+ Grammar_Score
+ TTR_Score
+ SpeechRate_Score
+ Clarity_Score

2.NLP Semantic Similarity Conversion
NLP_Score = cosine_similarity(transcript, rubric_keywords) × MaxCategoryPoints

3.Combined Category Score
Combined = Rule_Score + NLP_Score

4.Weighted Normalization
Weighted_Salutation = (Salutation_Combined / 5) × 5
Weighted_MustHave   = (MustHave_Combined / 30) × 30
Weighted_GoodToHave = (GoodToHave_Combined / 15) × 15


5.Final Score (0–100)
Final_Score = 
(
  Weighted_Salutation
+ Weighted_MustHave
+ Weighted_GoodToHave
) / 50 × 100
