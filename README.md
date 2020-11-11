# Masterthesis
## Hier werden im Rahmen meiner Masterarbeit verwendeten Programmiercode und einige Datensätze abgelegt
### Der ursprüngliche Datensatz liegt auf  https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29.
-balanced_short_Reviews.csv ist der ausbalancierte DAtensatz mit Reviews, die nicht länger als 150 Tokens sind. Dieses Korpus enthält 42.021 Medikamentbewertungen;
-not_balanced_11K_Samples.csv ist nicht ausbalanciertes Korpus mit 11.212 Beiträgen;
-balanced_2classes.csv ist ausbalanciertes Korpus für die Klassifikation mit zwei Klassen. Es enthält 46.000 Bewertungen, die nicht länger als 150 Tokens haben.
-All diese Korpora wurde mit preprocessing.py aus dem uhrsprühglichen. 
-requirements.txt enthält die Liste der für preprocessing.py benötigten Bibliotheken.
-Die eigentliche Sentiment Klassifizierung wurde auf Google Colab durchgeführt, der Notebook dafür heisst training_evaluation.ipynb
