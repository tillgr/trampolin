<!---
Die Erkennung von Sprunganfang auf dem Graph (Beispiel: Folie 6 PowerPoint Präsentation):<br>
	1. Die Spitzen müssen gefiltert werden (so dass sie leicht zu erkennen sind)<br>
	2. Die positive Beschleunigung ACCy zeigt Sprunganfang <br>
        3. Die negative Beschleunigung ACCz zeigt Sprunganfang <br>
	
Die Erkennung von Sprunganfang basierend auf dem all_data.csv:<br>
	1. Wenn die Beschleunigung Acc_x und Acc_z den Wert 5 von unten nach oben überschreitet, kann man davon ausgehen, dass eine Sprung vorliegt. Das 
           heißt man muss von dem erkannten Beschleunigungswert = 5,00 zurück gehen bis zum nächsten lokalen Minimum und dort beginnt der neue Sprung und 
           endet der Vorherige.
--->


### Ansätze
**Features:** ACC_x_Fil, ACC_z_Fil

_Lokale Maxima sind die Sprunggrenzen._

1. **Sprung erkennen:** Lokales Maximum finden (Ende des Sprungs --> Orientierung am absolut ersten Sprung), nächstes l.M. (Ende des nächsten Sprungs)
    1. Vergleich Nachbarwerte: wenn kleiner, dann lokales Maximum
    2. schnellere Alternative?
2. **Pause/Laufen erkennen:** Sehr kleine Werte
    1. Intervall für kleine Werte aufstellen, innerhalb dieses ist ein kein Sprung
    2. Counter für geringe Messwerte, falls überschritten, dann Pause
