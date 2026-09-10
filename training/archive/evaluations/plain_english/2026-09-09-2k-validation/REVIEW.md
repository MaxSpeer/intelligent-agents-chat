# Beide 2k-Trainings sind fertig; kurzer Stil gelernt, zusätzliche Vereinfachung uneinheitlich

Stand: 9. September 2026, 16:15 Uhr Berlin. Auswertung durch Codex, keine menschliche Verständlichkeitsstudie.

Der fortgesetzte Lauf **qwen3-8b-plain-english-2k-simplified-v1** wurde um **15:46 Uhr Berlin** erfolgreich beendet: 579 Optimierungsschritte, drei Epochen, finaler Adapter gespeichert. Der ursprüngliche 2k-Lauf war bereits um 12:44 Uhr fertig. Beide Run-Dateien melden `training_complete_checkpoint_selection_pending`: Das Training ist abgeschlossen, ein bester Checkpoint wurde noch nicht verbindlich ausgewählt.

Die auf dem Cluster gelesenen SHA256-Prüfsummen bestätigen für beide Läufe, dass der Adapter im Hauptverzeichnis dem jeweiligen `checkpoint-579` entspricht. Die beiden Adapter haben unterschiedliche Gewichte. Der Resume des neueren Laufs ist als erfolgreich vermerkt. Quellen: [Run-Datei](raw/qwen3-8b-plain-english-2k-simplified-v1/run.json), [Prüfung der Gewichte](weights-check.json).

## Gemessene Antwortlänge

Grundlage sind dieselben **10 festgelegten Validierungsdialoge mit insgesamt 19 Antworten**, davon zehn erste Antworten und neun Antworten auf Folgemeldungen. Alle Ausgaben waren vollständig; keine erreichte die Token-Grenze. Dies sind während des Trainings gespeicherte Ausgaben des 4-Bit-Modells, kein neuer Test über die Chat-UI.

| Variante | Wörter je erster Antwort, Mittelwert (n=10) | Wörter je Antwort insgesamt, Mittelwert (n=19) |
|---|---:|---:|
| Basismodell, neutraler Systemtext | 198,9 | 219,2 |
| Basismodell, zusätzlicher Plain-English-Systemtext | 129,1 | 133,3 |
| Plain English 2k, Checkpoint 579 | 22,6 | 24,6 |
| Plain English 2k simplified v1, Checkpoint 579 | 25,4 | 26,6 |

Beide finalen Adapter sind bei **allen 19 Antworten kürzer als die neutrale Basis**. Die durchschnittliche erste Antwort schrumpft um 88,6 % beziehungsweise 87,2 %. Der vereinfachte Adapter ist gegenüber dem normalen 2k-Adapter bei sieben Antworten kürzer, bei acht länger und bei vier gleich lang. Zwei Antworten sind wortgleich. Über alle Antworten ist er im Mittel 7,9 % länger; bei den ersten Antworten 12,4 %.

**Die Wortzahl belegt Kürze. Sie misst weder Verständlichkeit noch sachliche Richtigkeit oder Vollständigkeit.** Die Zahlen sind aus den Texten neu berechnet und gegen die gespeicherten Metadaten geprüft: [summary.json](summary.json), [alle acht eindeutigen Varianten im direkten Vergleich](comparison.jsonl).

## Inhaltliche Beurteilung

Die ausführlichen Listen des Basismodells werden meist zu ein bis drei Sätzen. Das reduziert die Menge, die ein Leser verarbeiten muss. Manche hilfreichen Beispiele und Einschränkungen verschwinden dabei ebenfalls. Die neue Datenrevision zeigt einzelne Verbesserungen, aber keinen durchgängigen Vorteil gegenüber dem normalen 2k-Adapter.

Ein gelungenes Beispiel ist die Frage nach besserem Leseverständnis. Die vereinfachte Version gibt eine konkrete Handlung:

> Yes, one helpful tip is to break down the text into smaller parts and try to summarize each part in your own words. This can help you better understand the material.

Beim normalen 2k-Adapter steht an dieser Stelle der ebenfalls einfache Tipp, kleine Abschnitte laut zu lesen. Die neue Antwort ergänzt eine andere konkrete Lernhandlung; ihre zusätzliche Länge ist hier kein Nachteil.

Bei der ersten Erklärung von Active Recall fehlt dagegen weiterhin ein einfaches Anwendungsbeispiel. Die vereinfachte Version formuliert:

> Active recall is a study technique where you try to remember information from memory, rather than just reading or highlighting it. This helps to strengthen your memory and improve your ability to retain information.

Der normale Adapter verwendet fast dieselbe Erklärung. In den Folgeantworten bleiben Wendungen wie `retrieve the information`, `solidify it in your memory` und `increasing intervals`. Der Basislauf mit Stilvorgabe zeigt hier bereits das konkretere Beispiel, nach dem Lesen das Buch zu schließen und das Erinnerte aufzuschreiben. Kürzen allein übernimmt dieses hilfreiche Erklärungsmuster also noch nicht zuverlässig.

Zwei auffällige Verschlechterungen der vereinfachten Endversion:

- **Kamera, Rückfrage nach Objektiven:** Der normale Adapter erklärt, dass zusätzliche Objektive am Anfang möglicherweise nicht nötig sind. Die vereinfachte Version beginnt mit „Yes, you may need …“ und empfiehlt anschließend, mit einigen wesentlichen Objektiven zu starten. Damit wird eine hilfreiche Einschränkung abgeschwächt. Die archivierten Musterantworten beider Läufe enthalten an dieser Stelle ausdrücklich „Not necessarily“ und empfehlen zunächst das vorhandene Objektiv.
- **Trailrunning-Schuhe:** Die vereinfachte Version ergänzt Nike React Infinity Run und Hoka One One Bondi als Beispiele. Die Hersteller führen diese als Straßenlaufschuhe. Daraus folgt: Die pauschale Empfehlung beantwortet die Trailrunning-Frage unzuverlässig. Der normale finale 2k-Adapter nennt diese Modelle nicht, behält aber ebenfalls unerklärte Ausdrücke wie `responsive midsoles`. Herstellerbelege: [Nike React Infinity Run](https://www.nike.com/in/w/nike-react-infinity-run-black-cushioned-shoes-4ewlxz90poyzakby0zy7ok), [HOKA Bondi](https://au.hoka.com/categories/styles/bondi), geprüft am 9. September 2026. Dies ist ein gezielter Faktencheck dieses Beispiels, keine vollständige Prüfung aller Aussagen.

## Durchsicht sämtlicher 19 finalen Antwortpaare

Die ID-Spalte verkürzt das gemeinsame Präfix `plain2k-everyday-train_sft-`. Die folgenden Beobachtungen sind qualitative Einschätzungen durch Codex, keine verblindeten Bewertungen und kein automatischer Qualitätsscore.

| Dialog / Antwort | Beobachtung: simplified gegenüber normalem 2k |
|---|---|
| 0993-qa2 / 1, Garten | Wortgleich; drei einfache Vorschläge. |
| 0515-dialogue / 1, Active Recall | Fast gleiche abstrakte Erklärung; vier Wörter mehr, kein konkretes Beispiel. |
| 0515-dialogue / 2, Vergleich mit Lesen | Fast gleiche Erklärung; `retrieve` und `solidify` bleiben. |
| 0515-dialogue / 3, weitere Lerntechnik | Etwas kürzer; `spaced repetition` wird mit abstrakten Intervallen erklärt, ohne Zeitbeispiel. |
| 0842-dialogue / 1, Kamera | Budget wird berücksichtigt; mehr Kamerakategorien und zehn Wörter mehr. |
| 0842-dialogue / 2, Objektive | Weniger hilfreiche Einschränkung; drängt stärker zu mehreren Objektiven. |
| 0546-dialogue / 1, Lebensmittel | Etwas kürzer; Begriffe wie `legumes` bleiben ohne Beispiel. Keine medizinische Faktenprüfung. |
| 0546-dialogue / 2, ausgewogene Ernährung | Mehrere kurze Aufforderungen, aber das konkrete Beispiel einer Mahlzeit entfällt. |
| 0546-dialogue / 3, Lebensmittel begrenzen | Kürzer; Salz als einzelner Punkt und einige konkrete Beispiele entfallen. Keine medizinische Faktenprüfung. |
| 0618-qa4 / 1, Regen | Kürzer; `fine rain` gibt eine konkrete Beschreibung. |
| 1500-qa4 / 1, Trailschuhe | Mehr Fachwörter/Markennamen; Straßenlaufschuhe als unpassende pauschale Beispiele. |
| 0316-dialogue / 1, französisches Gericht | Ähnlich einfach; zusätzliche allgemeine Aussage statt weiterer Erklärung. |
| 0316-dialogue / 2, Desserts | Deutlich länger; zusätzliche pauschale Zutaten- und Verzehraussagen. Keine vollständige Faktenprüfung. |
| 0316-dialogue / 3, Herkunft | Wortgleich. Kürze ist kein Nachweis, dass die Herkunftsaussage richtig ist. |
| 1878-qa2 / 1, Leseverständnis | Konkretes Zusammenfassen mit eigenen Worten; beide Varianten geben eine leicht umsetzbare Handlung. |
| 1683-qa4 / 1, Anpassungsfähigkeit | Gleich lang, aber konkretere Handlungen wie neue Aufgaben üben und Hilfe erfragen. |
| 0994-dialogue / 1, Karteikarten | `repeating it to yourself` ist als Handlungsbeschreibung weniger präzise als das vorherige `testing yourself`. |
| 0994-dialogue / 2, Karteikarten anwenden | Konkrete Vorderseite/Rückseite-Anleitung; länger. Das Zeitabstands-Beispiel des normalen Adapters entfällt. |
| 0994-dialogue / 3, Dank | Gleich kurze, passende Verabschiedung. |

Vollständige Antworten: [normaler 2k-Adapter](raw/qwen3-8b-plain-english-2k/validation_samples/checkpoint-579.md), [vereinfachter Adapter](raw/qwen3-8b-plain-english-2k-simplified-v1/validation_samples/checkpoint-579.md), [neutrale Basis](raw/qwen3-8b-plain-english-2k/validation_samples/base-neutral.md), [Basis mit Stilvorgabe](raw/qwen3-8b-plain-english-2k/validation_samples/base-style-prompt.md).

## Epochenvergleich

| Lauf | Epoche 1: erste Antwort, Wörter | Epoche 2 | Epoche 3 |
|---|---:|---:|---:|
| Plain English 2k | 26,0 | 26,4 | 22,6 |
| Plain English 2k simplified v1 | 25,5 | 28,5 | 25,4 |

Die geringste Wortzahl ist kein ausreichendes Auswahlkriterium. Bei simplified ist die erste Active-Recall-Erklärung in Epoche 1 beispielsweise etwas einfacher als später. Gleichzeitig behauptet die Antwort auf die Objektivfrage dort zu pauschal, separate Objektive seien nötig. In Epoche 2 und 3 treten die genannten Straßenlaufschuh-Beispiele auf. Keine der drei vereinfachten Epochen beseitigt also alle beobachteten Schwächen.

Der Validierungs-Loss des vereinfachten Laufs sinkt innerhalb dieses Laufs von **1,250 → 1,215 → 1,206**. Beim ursprünglichen 2k-Lauf sinkt er von **1,128 → 1,095 → 1,089**. Die beiden Reihen dürfen nicht als direkte Qualitätsrangliste verwendet werden: Die Musterantworten der Validierung wurden zwischen den Datensatzversionen ebenfalls geändert.

## Entscheidung und nächster Test

Der kurze Antwortstil wurde auf diesen Beispielen deutlich gelernt. Für den nächsten direkten Vergleich ist der **normale 2k-Adapter mit Checkpoint 579** eine sinnvolle vorläufige Ausgangsbasis: Die zusätzliche Datenrevision zeigt bislang keinen überzeugenden Gesamtvorteil und schwächt einzelne Antworten ab. Das ist keine statistisch abgesicherte Auswahl des besten Modells. Die vorhandenen Adapter und UI-Zuordnungen wurden bei dieser Prüfung nicht geändert.

Als nächstes sollten beide finalen Adapter im vom Nutzer gestarteten vLLM mit denselben bisher unbenutzten Fragen verglichen werden. Kriterien vorab: verständliche Wörter, umsetzbare Erklärung, notwendige Einschränkungen und korrekte Kernaussage. Erst danach weitere Datenänderungen oder Trainingsläufe festlegen. Wegen der wiederholten Sichtung dieser zehn Validierungsdialoge eignen sie sich nicht als unabhängiger Nachweis einer späteren Verbesserung.

**Aktuelle Grenze:** Die lokalen Modell-Endpunkte `127.0.0.1:8001` und `:8002` verweigerten die Verbindung. Daher wurde kein neuer Modellaufruf über die Chat-UI ausgeführt. Es wurden keine Cluster-Jobs, Trainings- oder Modellserver-Prozesse gestartet. Beleg: [endpoint-check.json](endpoint-check.json).

## Reproduzierbarkeit und Grenzen

`python3 audit_validation.py` führt ausschließlich eine lokale Auswertung mit der Python-Standardbibliothek aus. Es lädt kein Modell und sendet keine Netzwerkanfragen. Geprüft wurden:

- 52 übertragene Quelldateien gegen die bei der Übertragung erfassten SHA256-Prüfsummen; archivierte Trainings- und Validierungsdateien zusätzlich gegen die Run-Metadaten.
- Gleiche IDs, Nutzerfragen, neutrale Systemtexte und Trainingseinstellungen beider Läufe; beide gespeicherten Baselines sind inhaltlich identisch.
- Keine Überschneidung von Trainings-/Validierungs-IDs oder normalisierten ersten Fragen. Dies schließt semantisch ähnliche Themen nicht aus.
- Generierungsparameter: Seed 42, Temperatur 0,2, maximal 2048 neue Tokens, Thinking deaktiviert. Der Stilbaseline wurde derselbe zusätzliche Systemtext gegeben. Alle anderen Varianten nutzen den neutralen Text.
- Übereinstimmung von ausgegebenen Antworten, Turn-Reihenfolge, Wortzahl und Metadaten. Musterantworten waren laut Generierungsmetadaten nicht Teil des Prompts; Folgefragen nutzen die eigenen bisherigen Modellantworten.
- Finaler Trainer-Zustand: Schritt 579, Epoche 3. Die Gewichte im Hauptverzeichnis stimmen laut separater Cluster-Prüfung mit Checkpoint 579 überein.

Die zehn Dialoge sind nur ein Ausschnitt der 100 Validierungsdialoge. Sie liefern 19 sprachlich geprüfte Antworten je Variante; der Validierungs-Loss wird über die größere Validierungsmenge berechnet. Einzelner Seed, synthetische Themen, eigene Verlaufstexte bei Folgefragen und bereits verwendete Validierungsdaten begrenzen die Aussagekraft. Ein kontrollierter Vergleich mit dem Testdatensatz, ein vLLM-Livetest, menschliche Verständlichkeitsmessungen und eine vollständige sachliche Prüfung stehen aus.
