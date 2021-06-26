# Trampolin - Sportdatenanalyse <br>
*Zuständigkeit in Klammern dahinter* <br>
## Ordner Aufbau 
- GB                      -> gradient boosting classifier (Till)
- kNN                     -> k- nearest neighbours (Till)
- models                  -> beste Neural Network Models (Lukas und Lisa)
- Organisatorisches       -> Aufgabenstellung, Aufgabenverteilung und Sprungübersicht
- plots                   -> Shap plots und Confusion Matrix für die Verschiedenen Ansätze
- sgd                     -> Stochastic Gradient Descent (Anna)
- 
- Spungdaten Innotramp  	-> Orginal Daten, werden nicht mehr auf Github gespeichert, müssen sich vom CLoudstore geholt werden
- Spungdaten_integrated 	-> Aufbereitete Daten mit und ohne Vorverarbeitete Daten

## Dateien
- all_data.py             -> verabreitet die Rohdaten aus 'Sprungdaten Innotramp'/ fügt Rphdaten zusammen (Lukas und Lisa)
- feature_svc.py          -> (Weiyun)
- find_missing_data.py    -> ermöglicht das finden Fehlender Daten /Datenlücken, wird aber nicht benötigt (Lukas und Lisa)
- metrics.xlsx            -> Übersicht der Parameter für die einzelenen Ansätze für die percentage Datensätze
- neural_networks.py      -> für CNN und DFF Modelle erstellen austesten und Shap Diagramme erstellen, ... (Lukas und Lisa)
- process_all_data.py     -> verabreitet die zusammengefügten Daten von all_data.py weiter (Lukas und Lisa)
- random classifier       -> Schwellwert für die Modelle, basiert nur auf Verteilung der Sprünge (Lisa)
- svc.py                  -> Support Vector Classification (Weiyun)
