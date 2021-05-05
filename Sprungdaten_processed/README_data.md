# Datenintegration
*befinden sich auf dem [TU Cloudstore](https://cloudstore.zih.tu-dresden.de) (unter den Orginaldaten im Ordner Sprungdaten_processed*

Die nützlichen Daten sind nun in einem jeweils eigenen Unterordner mit jeweils einem Trainings- und Test-Datensatz.

Übersicht der csv Dateien:
| Name | Inhalt |
| -- | -- |
| all_data.csv | beinhaltet alle Daten, Pausen und Datenfehler etc. hintereinander |
| data_only_jumps.csv | beinhaltet alle Sprünge. Pausen, Datenfehler, Unbekannte, Fehlende Daten und Einturnen sind aussortiert |
| data_point_jumps.csv | beinhaltet alle Sprünge, welche Punkte geben. Außerdem sind hier die Namen der Sprünge angepasst, sodass eigentlich gleiche Sprünge nicht mehr mehrere Klassen einnehmen durch andere Schreibweise |
| averaged_data.csv | beinhaltet für jeden Sprung genau eine Zeile mit nur den gemittelten Werten |
| std_data.csv | beinhaltet für jeden Sprung genau eine Zeile mit nur den Werten der Standardabweichung |
| normalized_data.csv | beinhaltet Daten, welche für jede Spalte min-max normalisiert sind |
| jumps_time_splits.csv | Jumps with Acceleation and Gyro measurements at the time of % of the whole jump duration. For example, by 1/4, 2/4, 3/4 of the jump duration.|
| ... | ... |
