**Plain English v2: Training abgeschlossen, Vereinfachung noch uneinheitlich**

Stand: 9. September 2026. Ausgewertet wurden die gespeicherten Antworten aus Job 2522643. Der Lauf dauerte von 09:02 bis 09:34 Uhr MESZ, insgesamt 32 Minuten 18 Sekunden einschließlich Modellladen und Antwortgenerierung. Es wurden 18 Optimierungsschritte über drei Epochen durchgeführt. Alle drei Checkpoints und der finale Adapter sind gespeichert. Für diese Auswertung wurden ausschließlich vorhandene Dateien gelesen und lokal ausgewertet.

**Entscheidung:** Keinen Checkpoint als verlässlich verbesserten Plain-English-Adapter auswählen. V2 zeigt einzelne verständlichere Antworten, aber keinen einheitlichen Stilwechsel. Der niedrigste Validierungsverlust allein rechtfertigt keine Auswahl von Epoche 3. Diese Bewertung ist als Pilotbefund mit Einschränkungen verwendbar, nicht als allgemeiner Leistungsnachweis.

Das finale Adapterverzeichnis enthält exakt dieselben Gewichte wie `checkpoint-18`: SHA256 `f0967aca9005da133626e66d7a3bfc6729197329144ddf7a223af45c2ed70dd1`. Die drei Checkpoints haben unterschiedliche Gewichtedateien. Es wurde kein Modell für die Chat-App umgestellt und kein weiterer Clusterjob gestartet.

| Variante | Validierungsverlust | Wörter je erster Antwort, Mittelwert | Median | Erste Antworten kürzer als Basis | Abgeschnittene Antworten insgesamt |
|---|---:|---:|---:|---:|---:|
| Basismodell, neutraler Prompt | nicht separat gemessen | 222,7 | 243,0 | – | 0/13 |
| Epoche 1 / checkpoint-6 | 2,520 | 241,8 | 241,5 | 2/10 | 0/13 |
| Epoche 2 / checkpoint-12 | 2,042 | 223,5 | 210,0 | 4/10 | 0/13 |
| Epoche 3 / checkpoint-18 | 1,923 | 234,3 | 241,5 | 3/10 | 0/13 |
| Basismodell mit Stil-Anweisung | nicht separat gemessen | 128,6* | 122,5* | 8/10* | 2/13 |

Grundlage der Wortmittelwerte und Mediane sind dieselben zehn ersten Fragen. Die drei Anschlussfragen sind separat ausgewertet, damit längere Dialoge den Hauptvergleich nicht stärker gewichten. Bei den Anschlussfragen liegen Basis, Epoche 1, Epoche 2 und Epoche 3 bei durchschnittlich 296,7 / 305,0 / 328,7 / 367,0 Wörtern. Epoche 3 ist dort bei allen drei Antworten länger. Über alle 13 Antworten liegt sie bei 264,9 statt 239,8 Wörtern. Sämtliche Werte wurden aus den Texten neu berechnet; vollständige Kennzahlen stehen in [summary.json](summary.json).

*Die Stil-Anweisung erzeugte beim Cornell-Notizsystem sowohl in der ersten Antwort als auch in der Anschlussantwort endlos weitgehend leere Tabellenzeilen. Beide Ausgaben erreichen 2.048 Tokens. Der Wortzähler erfasst diese Zeichen kaum: wenig Wörter bedeutet hier weder eine kurze noch eine gelungene Antwort. Diese Ausfälle sind in der Tabelle enthalten. Als zusätzliche Sensitivitätsprüfung wurde der gesamte Cornell-Dialog bei allen Varianten entfernt: Für die verbleibenden neun ersten Fragen liegen Basis/Stil-Anweisung bei 216,6/136,9 Wörtern. Auch das belegt nur kürzere Texte in den vollständigen Fällen und gleicht die Ausfälle nicht aus.*

**Beobachtungen aus der Textdurchsicht**

Alle 65 Ausgaben wurden sprachlich und auf offensichtliche Format-, Rechen- und Logikprobleme durchgesehen. Die Bewertung stammt von Codex, war nicht verblindet und ersetzt keine unabhängige menschliche oder vollständige fachliche Prüfung. Wortzahl wurde nicht als alleiniger Maßstab für Verständlichkeit verwendet: Bewertet wurden auch direkte Beantwortung, unnötige Wiederholungen, Fachbegriffe, hilfreiche Beispiele und Vollständigkeit.

| Frage / Dialog-ID | Beobachtung |
|---|---|
| Neue Gewohnheit / 0095 | Epoche 3 reduziert 179 auf 118 Wörter und konzentriert sich auf drei Schritte. Die Formulierungen bleiben teilweise abstrakt; die Variante mit Stil-Anweisung enthält anschaulichere Alltagssituationen. Epoche 1 und 2 behalten den längeren Aufbau mit zusätzlichen Tipps. |
| Cornell-Notizen, erste Antwort / 0034 T1 | Epoche 2 und 3 liefern längere Erklärungen mit viel Biologie im Beispiel. Epoche 3 nennt zusätzlich etwa thylakoid membranes und stroma. Epoche 1 ist kürzer, zeichnet jedoch drei nebeneinanderstehende Spalten, obwohl die eigene Erklärung eine Zusammenfassung unten fordert. Die Stil-Anweisung scheitert an wiederholten Tabellenzeilen. |
| Cornell-Notizen, Anschlussfrage / 0034 T2 | Alle Adapter wählen ein relativ technisches Photosynthese-Beispiel. Epoche 2 und 3 ergänzen gegenüber der Basis mehr Text. Die Stil-Anweisung wiederholt erneut leere Tabellenzeilen und liefert kein vollständiges Beispiel. |
| Vektor normalisieren / 0031 T1 | Alle drei Adapter bleiben sprachlich und im Formelaufbau sehr nah an der Basis. Epoche 3 ergänzt die hilfreiche Einschränkung auf Vektoren ungleich null. Die Stil-Anweisung erklärt zunächst anhand konkreter Zahlen. |
| Vektor ohne Brüche / 0031 T2 | Alle Varianten rechnen zunächst mindestens ein ungeeignetes Beispiel vor, bevor sie einen passenden Vektor nennen. Epoche 3 ist am längsten und behauptet zusätzlich fälschlich, eine ganzzahlige Länge sei notwendig. Gegenbeispiel: (0,5; 0) hat Länge 0,5 und wird zu (1; 0) normalisiert. |
| Zwei-Faktor-Anmeldung / 0097 T1 | Epoche 3 bleibt mit 293 statt 295 Wörtern im ausführlichen Aufbau der Basis. Epoche 2 ist kürzer, führt aber zusätzliche Begriffe wie knowledge, possession und inherence ein. Die Stil-Anweisung erklärt den Ablauf deutlich direkter. |
| Vor-/Nachteile der Anmeldeverfahren / 0097 T2 | Alle Adapter sind länger als die Basis. Epoche 3 enthält zudem einen internen Widerspruch: Bei Authenticator-Apps nennt sie sowohl einfache Sicherungen als Vorteil als auch keine Sicherung als Nachteil. Eine vollständige Sicherheitsprüfung der sonstigen Behauptungen wurde nicht durchgeführt. |
| AR, VR und MR / 0053 | Alle Adapter behalten längere Definitionen, Gerätebeispiele und Zusammenfassungen. Epoche 2 kürzt etwas; Epoche 3 ist ähnlich lang wie die Basis. Die Stil-Anweisung ist einfacher formuliert, erklärt aber den Unterschied zwischen AR und MR wenig trennscharf. |
| Synonyme für reject / 0074 | Alle Varianten führen das Ausgangswort selbst mehrfach auf. Epoche 2 bietet einige zusätzliche Alternativen. Epoche 3 wiederholt reject besonders häufig und wächst von 28 auf 76 Wörter, ohne entsprechend mehr Synonyme zu liefern. |
| Kugelvolumen / 0099 | Die Adapter ändern den Stil kaum und rechnen das angefragte Beispiel korrekt zu rund 4.188,8 cm³. Die Stil-Anweisung verwendet ausdrücklich die gröbere Näherung π≈3,14 und erhält entsprechend 4.186,67 cm³; sie ergänzt ein nicht angefragtes zweites Beispiel. |
| Käseherstellung / 0051 | Die Adapter erweitern 238 Wörter auf 322 / 331 / 306 Wörter. Sie erläutern Starterkulturen und weitere Prozessschritte. Das liefert teils zusätzliche Information, setzt das Ziel einer einfacheren kurzen Erklärung aber nicht überzeugend um. Die Stil-Anweisung vereinfacht deutlich, pauschalisiert dabei einzelne Schritte. |
| Photosynthese / 0026 | Epoche 2 und 3 kürzen auf 183 und 142 statt 248 Wörter; insbesondere die ausführliche ATP/NADPH-Erklärung entfällt. Epoche 3 behält allerdings Fachbegriffe wie thylakoid membranes bei. Dies ist ein positives Einzelbeispiel, kein durchgängiger Stilwechsel. |
| Historische Kleidung / 0029 | Epoche 3 erweitert 356 auf 447 Wörter und wiederholt Materialien über mehrere Epochen. Epoche 2 bleibt ähnlich zur Basis. Die Stil-Anweisung ist kürzer, fügt aber eine hier nicht belegte Behauptung zu Kleidung aus Ton/Schlamm ein; daraus wird keine sachliche Qualitätsverbesserung abgeleitet. |

Die vollständigen Antworten und die jeweiligen ursprünglichen Fragen stehen in [comparison.jsonl](comparison.jsonl), alternativ nach Modellvariante in [raw/validation_samples](raw/validation_samples). Die IDs in der Tabelle haben das Präfix `plain-en-`.

**Was geprüft wurde und was offen bleibt**

Der Audit bestätigt 32 übertragene Quelldateien einschließlich ihrer Prüfsummen, die Übereinstimmung der gespeicherten Code- und Datenkopien mit den Run-Prüfsummen, eindeutige Dialog-IDs, identische Nutzerfragen und neutrale Systemnachrichten für Basis und Adapter, vollständige 13 Antworten je Variante sowie identische gespeicherte Sampling-Einstellungen. Die gespeicherten Gesprächsverläufe enthalten jeweils die eigenen generierten Antworten. Der dazugehörige Generierungscode wurde auf die Verwendung dieser Verläufe statt der Musterantworten geprüft. Wortmittelwerte und Mediane wurden zusätzlich mit einer separaten lokalen Implementierung überprüft.

Alle Varianten nutzen dasselbe auf 4 Bit quantisierte Modell im Trainingsprozess, `enable_thinking=False`, Temperatur 0,2, maximal 2.048 neue Tokens und denselben jeweiligen Frage-/Turn-Seed. Es handelt sich um zehn Validierungsdialoge mit einem einzigen Sampling-Durchlauf. Die Anschlussfragen verwenden verschiedene eigene Antwortverläufe. Die Stabilität bei anderen Seeds, im vLLM-Server und mit dessen Präzision wurde hier nicht gemessen. Ein direkter Leistungsgewinn gegenüber v1 lässt sich aus den unterschiedlichen Fragen und Inferenzverfahren der beiden Untersuchungen nicht ableiten.

Der Verlust sinkt innerhalb dieses Laufs. Das zeigt bessere Vorhersage der vorgegebenen Antworttokens unter dem Trainingsziel, nicht automatisch eine bessere frei generierte Erklärung. Die Stichprobe bietet keine Grundlage für eine allgemeine Erfolgsquote oder für die Behauptung, dass die Datenmenge allein die Ursache ist. Der alte Testdatensatz wurde bereits in der v1-Analyse betrachtet; für eine abschließende Bewertung nach weiteren Anpassungen werden neue, unberührte Testfragen benötigt.

**Nächster sinnvoller Schritt:** Mit wenigen bereits im Training enthaltenen Fragen prüfen, ob der Adapter den gewünschten Stil wenigstens auf bekannten Beispielen reproduziert, und Basis/Adapter dabei unter identischen Inferenzbedingungen vergleichen. Das unterscheidet einen schwachen Lerneffekt von fehlender Übertragung auf neue Fragen. Die auffälligen Wiederholungen sollten bei diesem kleinen Vergleich ebenfalls kontrolliert werden. Erst danach gezielt über mehr bzw. bessere Beispiele oder andere Trainingsparameter entscheiden. Zusätzliche Modellaufrufe oder Clusterjobs wurden dafür noch nicht gestartet.

Reproduktion der Dateiprüfungen und Kennzahlen: [audit_validation.py](audit_validation.py), lokal mit Python und ohne Modellaufruf ausführbar. Die separate Nachrechnung steht in [independent-check.json](independent-check.json). Die Herkunft und Prüfsummen stehen in [source-manifest.json](source-manifest.json), der abgeschlossene Lauf in [raw/run.json](raw/run.json).
