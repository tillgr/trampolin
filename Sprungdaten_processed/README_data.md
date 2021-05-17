# Datenintegration
*befinden sich auf dem [TU Cloudstore](https://cloudstore.zih.tu-dresden.de) (unter den Orginaldaten im Ordner Sprungdaten_processed*

Die nützlichen Daten sind nun in einem jeweils eigenen Unterordner mit jeweils einem Trainings- und Test-Datensatz.

Übersicht der csv Dateien:
| Name | Inhalt |
| --- | --- |
| all_data.csv | beinhaltet alle Daten, Pausen und Datenfehler etc. hintereinander |
| data_only_jumps.csv | beinhaltet alle Sprünge. Pausen, Datenfehler, Unbekannte, Fehlende Daten und Einturnen sind aussortiert |
| data_point_jumps.csv | beinhaltet alle Sprünge, welche Punkte geben. Außerdem sind hier die Namen der Sprünge angepasst, sodass eigentlich gleiche Sprünge nicht mehr mehrere Klassen einnehmen durch andere Schreibweise |
| averaged_data.csv | beinhaltet für jeden Sprung genau eine Zeile mit nur den gemittelten Werten |
| std_data.csv | beinhaltet für jeden Sprung genau eine Zeile mit nur den Werten der Standardabweichung |
| normalized_data.csv | beinhaltet Daten, welche für jede Spalte min-max normalisiert sind |
| jumps_time_splits.csv | Jumps with Acceleation and Gyro measurements at the time of % of the whole jump duration. For example, by 1/4, 2/4, 3/4 of the jump duration.|
| same_length | Alle Sprünge haben die gleiche Länge: <br> cut_first: Immer die ersten Daten eines Sprung sind entfernt <br> cut_last: Immer die letzten Daten eines Sprungs sind entfernt <br> padding_0: Am Ende der Sprünge wird auf die gleiche Länge alles mit 0 aufgefüllt|
| percentage | Sprünge haben die gleiche Länge. <br>-> ohne mean: Datenpunkte mit gleichmäßigen Abstand. percentage_X: X gibt Abstand der Datenpunkte in Prozent an. <br>-> mit mean: Datenpunkt gemittelt von dem gleichmäßigen Abstand. <br> Bsp: mean_25 : 4 Daten Zeilen pro Sprung, 0-25% Daten gemittelt = ersten Zeile, 25-50% = zweite Zeile... <br> - vector: Daten wurden Vektorisiert|
| ... | ... |
