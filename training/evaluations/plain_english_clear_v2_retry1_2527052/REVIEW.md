# Clear v2: Training abgeschlossen, verständlicherer Stil mit verbleibenden Fehlern

Geprüft am 10. September 2026. Job **2527052** auf einer A40 hat alle **579 Schritte / drei Epochen** abgeschlossen. Beginn: 9. September, 23:26 Uhr; Ende: 10. September, 00:29 Uhr, jeweils Berlin. Laufdauer einschließlich erzeugter Vergleichsantworten: rund 63 Minuten.

Der Adapter liegt auf dem Cluster unter:

```text
/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/adapters/qwen3-8b-plain-english-clear-v2-retry1
```

Gespeichert sind Checkpoint 193, 386 und 579. Die Gewichte im Hauptordner stimmen per SHA256 mit Checkpoint 579 überein. Alle vier Safetensors-Dateien enthalten 288 Tensoren; die Dateigröße passt zu den im Header aufgeführten Daten. Der alte abgebrochene Lauf im Ordner ohne `-retry1` wurde nicht verändert. Belege: [Run-Metadaten](raw/run.json), [Gewichtsprüfung](weights-check.json).

## Was sich an den Antworten geändert hat

Die gespeicherten Ausgaben umfassen dieselben zehn festgelegten Validierungsdialoge mit jeweils 19 Antworten. Die beiden Basisvarianten sind inhaltlich identisch mit den archivierten Basisantworten des vorherigen vereinfachten Laufs. Alle neuen Adaptervarianten verwenden den neutralen Systemtext; nur die gesonderte Stilbaseline enthält eine zusätzliche Stilvorgabe.

| Variante | Wörter je erster Antwort, Mittelwert (10 Antworten) | Wörter je Antwort insgesamt (19 Antworten) |
|---|---:|---:|
| Basismodell, neutral | 198,9 | 219,2 |
| Basismodell, Stilvorgabe | 129,1 | 133,3 |
| Vorheriger simplified-v1-Adapter, final | 25,4 | 26,6 |
| Clear v2, Checkpoint 193 | 35,6 | 34,2 |
| Clear v2, Checkpoint 386 | 32,7 | 31,8 |
| Clear v2, Checkpoint 579 | 31,4 | 33,3 |

Gegenüber dem vorherigen vereinfachten Adapter sind die vollständigen Antworten beim neuen finalen Checkpoint länger, aber die einzelnen Sätze kürzer: Die einfache Satzgrenzenzählung ergibt etwa **8,7 statt 13,3 Wörter**. Überschriften, Listen und Abkürzungen beeinflussen diese Näherung; sie ist kein Verständlichkeitsscore.

Inhaltlich zeigt sich die gewünschte Richtung. Zum zeitlich verteilten Wiederholen nennt der finale Checkpoint jetzt einen konkreten Ablauf:

> Spaced repetition is one technique. You review material at increasing intervals. For example, review it after one day, then after a few days, then after a week. This can help you remember it longer.

Der frühere Adapter erklärte nur abstrakt, dass zunehmende Zeitabstände die Erinnerung stärken. Das neue Beispiel macht die Bedeutung sichtbar, auch wenn „increasing intervals“ weiterhin vermeidbar wäre. Checkpoint 193 gibt bei Active Recall die anschauliche Folge Abdecken, Aufschreiben und Prüfen. Checkpoint 579 erklärt „legumes“ durch Bohnen, Linsen und Erbsen. Bei Trailrunning entfallen die früheren Markenbeispiele und mehrere unnötige Fachausdrücke.

## Warum noch kein Checkpoint als bester ausgewählt ist

Die Durchsicht aller 57 neuen Antworten sowie 19 Antworten des vorigen finalen Adapters zeigt auch Rückschritte:

- **193:** Gute konkrete Lernanleitung, aber „minimally processed“ definiert Vollkorn nicht korrekt. Entscheidend sind die erhaltenen Kornbestandteile. Quelle: [USDA Food Buying Guide](https://foodbuyingguide.fns.usda.gov/Content/TablesFBG/USDA_FBG_Section4_Grains.pdf).
- **386:** Die Antwort grenzt Trailrunning-Schuhe fälschlich von Laufschuhen allgemein ab; sinnvoll wäre die Unterscheidung von Straßen- und Trailrunning-Schuhen. Quelle: [Brooks zur Wahl von Laufschuhen](https://www.brooksrunning.com/en_us/blog/advice-tips/find-the-best-running-shoe-for-you.html). Außerdem ist der Bezug von „These are fats“ nach einer Liste aus Zucker, Salz und Fett missverständlich.
- **579:** „Lens mount“ wird einem wechselbaren Objektiv falsch zugeordnet. Gemeint ist dessen Befestigung am Kameragehäuse. Quelle: [Nikons Anleitung zum Objektivwechsel](https://www.nikonusa.com/learn-and-explore/c/tips-and-techniques/getting-started-how-to-change-a-dslr-lens). Die Herkunftsfrage zu Crème brûlée wird überwiegend mit Zubereitung beantwortet.
- **Alle drei:** Die frühere konkrete Karteikartenanleitung mit Frage, Antwort und Selbstprüfung wird nicht vollständig erhalten.

Der Validierungs-Loss beträgt **1,469 → 1,437 → 1,439**. Checkpoint 386 hat den geringsten Loss, liefert in den gelesenen Antworten aber zusätzliche Fehler. Ich würde **193 und 579 für den nächsten Vergleich behalten** und 386 zurückstellen. Keiner wird anhand dieser kleinen Stichprobe als eindeutig bester festgelegt. Die vollständige Bewertung steht in [PEER_REVIEW.md](PEER_REVIEW.md); alle Antworten sind in [comparison.jsonl](comparison.jsonl) nebeneinander zugeordnet.

## Prüfgrenzen und nächster Schritt

Dies ist eine Auswertung bereits erzeugter Antworten des 4-Bit-Trainingsmodells. Es wurden keine neuen Modellanfragen, Clusterjobs oder Server gestartet und keine UI-Zuordnungen geändert. Die Prüfungen erfassen 23 übertragene Dateien, Datensatz-Prüfsummen, IDs, Nutzerfragen, Rollen, Generierungsparameter, vollständige Antworten und Trainerzustand. Reproduzierbar mit `python3 audit.py`; Ergebnisse: [summary.json](summary.json).

Die zehn Validierungsdialoge wurden bereits mehrfach für Entwicklungsentscheidungen verwendet. Ein einzelner Seed, die eigenen bisherigen Modellantworten bei Folgefragen und die KI-gestützte, nicht verblindete Durchsicht begrenzen die Aussage. Es gibt keine menschliche Verständlichkeitsmessung oder vollständige Faktenprüfung. Die Referenzantworten unterscheiden sich von früheren Datenrevisionen; ihre Loss-Werte sind deshalb nicht direkt vergleichbar.

Vor weiterer Datenänderung oder neuem Training sollten 193 und 579 auf vorher festgelegten, bisher unbenutzten Fragen verglichen werden. Kriterien: verständliche Wörter, korrekt erklärte Begriffe, umsetzbare Schritte, erhaltene Bedingungen und direkte Antworten auf Rückfragen. Den reservierten Testsplit erst nach der Checkpoint-Auswahl für die abschließende Beurteilung verwenden.
