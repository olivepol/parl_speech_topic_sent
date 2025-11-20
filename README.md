# parl_speech_topic_sent
Analyzing how policy makers spoke on specific policy topicy in the Bundestag, leveraging NLP.

research question: “Wie unterscheiden sich Themen und Sentiment zwischen Fraktionen im Deutschen Bundestag über Zeit, und wie verändern sich diese Muster in politischen Krisen?”

Operationalisierung:
- Themen = Topics aus Topic Modeling auf speechContent
- Sentiment = Scores (−, 0, +) aus einem Sentiment-Modell
- Gruppen = factionId / Partei, positionShort (Minister, MP, Präsidium …)
- Zeit = date (z.B. nach Legislaturperiode / Jahr / vor/nach Event)
- Damit hast du eine klar politikwissenschaftliche Frage + spannenden NLP-Teil.