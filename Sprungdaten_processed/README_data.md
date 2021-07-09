# Datenintegration
*befinden sich auf dem [TU Cloudstore](https://cloudstore.zih.tu-dresden.de) (unter innotramp/Sprungdaten_processed)*

Die Datensätze zum Trainieren und Testen der Modelle befinden sich in den Unterordnern.

## Übersicht der csv Dateien:
*Dateinen die auf dem Cloudstore liegen wurden **fett** hervorgehoben*
| Name | Inhalt |
| --- | --- |
| **all_data.csv** | beinhaltet alle Daten (auch Datenfehler und Rechtschreibfehler; Pausen wurden schon entfernt um Datei zu verkleinern)|
| **all_data_\*.csv**| beinhaltet alle vereinten Rohdaten Daten aus dem jeweiligen Ordner|
| class_std_mean.xlsx| alle Sprüunge der selben Art gemittelt und die Standardabweichung berechnet|
| ... | ... |

## Übersicht der Ordner:
| Name | Inhalt |
| --- | --- |
|with_preprocessed | beinhaltet alle weiterverarbeiteten Datein, die auch die vorverarbeiteten Daten (Djumps) enthalten <br> beinhaltet die Datei **data_only_jumps.csv** mit vorverarbeiteten Daten|
|without_preprocessed | beinhalten nur noch die Rohdaten (ohne vorverarbeitete Daten/ DJumps) <br> beinhaltet die Datei **data_only_jumps.csv** ohne vorverarbeiteten Daten|
| ... | ... |
