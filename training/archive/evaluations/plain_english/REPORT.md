# Plain-English-Adapter: Vergleich vom 8. September 2026

**Das Training wurde technisch abgeschlossen, aber das gewünschte Ergebnis – verlässlich einfachere Antworten – ist in diesem Test nicht erreicht.** Der Adapter verändert die Antworten. Sie sind jedoch meist länger und enthalten häufig zusätzliche Fachbegriffe. Der sinkende Validation Loss reicht deshalb nicht als Nachweis einer gelungenen Stiländerung.

## Ergebnis

Verglichen wurden `qwen3-8b` ohne Adapter und `plain-english` auf demselben bereits laufenden vLLM-Server. Grundlage sind die zehn Testdialoge, die beim Fine-Tuning nicht zum Training oder zur Checkpoint-Auswahl verwendet wurden. Zwei Dialoge enthalten eine Rückfrage.

Die folgende Hauptauswertung verwendet Temperature **0,2**, entsprechend der Temperatureinstellung des Chats. Alle Antworten dieses Durchlaufs wurden vollständig beendet.

| Kennzahl für die zehn ersten Antworten | Ohne Adapter | Plain-English-Adapter |
| --- | ---: | ---: |
| Durchschnittliche Wörter je Antwort | 227,9 | 298,8 |
| Median der Antwortlänge | 223 | 306,5 |
| Bei ausgeblendeten Modellnamen als verständlicher bewertet | 6 | 1 |

Bei drei Fragen gab es keinen klaren Verständlichkeitsvorsprung. Die Adapterantworten waren im Mittel **31,1 % länger** und bei **neun von zehn Fragen** länger. Die beiden Rückfragen sind separat ausgewertet: Auch dort waren beide Adapterantworten länger und nach der qualitativen Bewertung weniger zugänglich.

Die qualitative Bewertung stammt von Codex, nicht von unabhängigen menschlichen Testpersonen. Bewertet wurden geläufige Wörter, erklärte Fachbegriffe, direkte Antworten und hilfreiche Beispiele. Länge ist nur eine zusätzliche Messgröße: **Kürzer bedeutet nicht automatisch verständlicher.**

| Frage/Thema | Wörter ohne Adapter | Wörter mit Adapter | Verständlicher laut Bewertung |
| --- | ---: | ---: | --- |
| Englischen „th“-Laut aussprechen | 389 | 437 | Ohne Adapter; beide mit inhaltlichen Problemen |
| Schrödingers Katze | 242 | 185 | Kein klarer Unterschied |
| Wie entstehen Erdbeben? | 133 | 311 | Ohne Adapter |
| Sterne in der Stadt | 104 | 173 | Ohne Adapter |
| Identitätsfunktion | 179 | 227 | Kein klarer Unterschied |
| Gespräch mit Fremden im Flugzeug | 301 | 436 | Ohne Adapter |
| Was ist ein DHT-Knoten? | 219 | 302 | Ohne Adapter; beide mit inhaltlichen Problemen |
| Newtons drei Gesetze | 227 | 357 | Mit Adapter |
| Stricken und Häkeln | 132 | 147 | Ohne Adapter |
| Markov-Kette | 353 | 413 | Kein klarer Unterschied |

## Konkrete Beispiele

**Erdbeben:** Beide Varianten beginnen mit denselben zwei Sätzen über freigesetzte Energie und Plattenbewegung. Der Adapter kündigt anschließend „Here's a more detailed explanation“ an. Es folgen sechs Abschnitte mit zusätzlichen Begriffen wie *lithosphere*, *asthenosphere*, *divergent/convergent/transform boundaries* und *hypocenter*. Einige werden erklärt, sie erhöhen aber den Aufwand für eine einfache Einstiegsfrage. Die Antwort ohne Adapter erklärt den wesentlichen Ablauf bereits in einem Absatz. Das ist ein deutliches Gegenbeispiel zur angestrebten Vereinfachung. Die Kernerklärung zu Spannung, plötzlichem Gleiten und Erschütterung lässt sich mit der [USGS-Erklärung](https://www.usgs.gov/faqs/what-earthquake-and-what-causes-them-happen) abgleichen; die „semi-fluid“-Formulierung beider Varianten ist eine grobe Vereinfachung.

**Newtons Gesetze:** Hier ist die längere Adapterantwort tatsächlich hilfreicher. Sie ergänzt Alltagsbeispiele mit einem Ball, schweren und leichten Kisten sowie dem Abstoßen vom Boden. Zum zweiten Gesetz steht zusätzlich: „A greater force causes greater acceleration, and a larger mass requires more force to achieve the same acceleration.“ Die Antwort ohne Adapter bleibt stärker bei formalen Definitionen und Symbolen. Dieses Beispiel wurde deshalb zugunsten des Adapters bewertet.

**Inhaltliche Qualität:** Auch die kürzere Basisantwort ist nicht automatisch richtig. Bei der Aussprachehilfe enthalten beide Varianten irreführende Anweisungen. Beim DHT-Beispiel nennt der Adapter Bitcoin als DHT-artiges Beispiel; die Basisantwort behauptet zu pauschal, jeder BitTorrent-Peer sei ein DHT-Knoten. Zum Abgleich dienen die [Bitcoin-Dokumentation zur Peer-Suche](https://developer.bitcoin.org/devguide/p2p_network.html), [BitTorrent BEP 5](https://www.bittorrent.org/beps/bep_0005.html), [BEP 27 zu privaten Torrents](https://www.bittorrent.org/beps/bep_0027.html) und die [Ausspracheanleitung für th](https://pronuncian.com/pronounce-th-sounds/). Es wurde keine vollständige fachliche Richtigkeitsquote erhoben.

## Durchführung und Nachprüfung

- Insgesamt **48 generierte Antworten**: zwei Einstellungen × zwei Modellvarianten × zwölf Antworten. Es wurden ausschließlich Anfragen an den vom Nutzer gestarteten Server gestellt; keine neuen Cluster-Jobs, Trainings oder Server gestartet.
- Alle ersten Antworten erhielten exakt dieselben System- und Nutzernachrichten. Der neutrale Systemtext lautete: `You are a helpful assistant. Answer the user's question directly and clearly.` Es gab keine zusätzliche Aufforderung zu einfacher Sprache.
- Bei Rückfragen wurde jeweils die eigene vorherige Modellantwort verwendet. Musterantworten aus dem Datensatz wurden nie als Gesprächsverlauf an das Modell übergeben. Deshalb werden die zehn ersten Antworten und die zwei Rückfragen getrennt ausgewertet.
- Gleiche Einstellungen pro Paar: Thinking aus, Seed 42, Top-p 1, Top-k −1, Min-p 0, keine zusätzlichen Wiederholungsstrafen, maximal 2.048 Antworttokens. Das höhere Tokenlimit verhindert unnötige Abschneidungen. Der eigentliche Chat nutzt ein anderes Systemprompt und ein Limit von 1.024 Tokens; dies war daher ein kontrollierter API-Vergleich, kein vollständiger Test der Chat-Oberfläche.
- Zunächst wurde Temperature 0 geprüft. Dabei geriet die Adapterantwort zur th-Aussprache in eine Wiederholung und erreichte das Tokenlimit. Einschließlich dieses Ausreißers waren die ersten Antworten im Mittel 86,3 % länger. **Auch nach Ausschluss des gesamten betroffenen Antwortpaars** waren die übrigen neun Adapterantworten noch 29,2 % länger. Der zweite Durchlauf mit der Chat-Temperature 0,2 prüfte, ob das Ergebnis davon abhängt; dort gab es keine Abschneidung, und die Richtung blieb gleich.
- Die Modellnamen wurden für die qualitative Bewertung zufällig als A/B ausgeblendet. Die Bewertungen beider Durchläufe wurden vor dem Öffnen der Zuordnung gespeichert. Anschließend wurden Zuordnung, angefragte und zurückgegebene Modell-ID sowie Serverkonfiguration kontrolliert.
- Die Prüfungen der Rohdaten bestanden: 24 eindeutige Anfragen pro Durchlauf, vollständige Fragepaare, identische Anfangsprompts und Einstellungen, eigene Verläufe bei Rückfragen, keine Referenzantworten im Input und keine separate Thinking-Ausgabe.
- Die Wortzählung zählt mit `\b\w+(?:['’\-]\w+)*\b`. Überschriften, Listennummern und passende Formelsymbole zählen mit. Eine unabhängige Zählung anhand von Leerraum ergab ebenfalls rund 30 % längere Adapterantworten. Es handelt sich um eine Längenmessung, nicht um einen zertifizierten Lesbarkeitsgrad.

## Trainings- und Modellnachweis

Der gespeicherte Trainingszustand bestätigt drei Epochen mit 15 Optimierungsschritten. Der Validation Loss sank von **2,704** über **2,337** auf **2,190**; Checkpoint 15 wurde als bester gespeichert. Das ist eine Verbesserung der Vorhersage der vorgegebenen Validierungsantworten innerhalb dieses Trainingslaufs, kein direkter Test der Verständlichkeit frei generierter Antworten.

Der laufende Server meldete vLLM **0.27.0**, das Basismodell **Qwen/Qwen3-8B** mit Revision `b968826d9c46dd6066d109eabc6255188de91218` und den Adapter `plain-english` unter `/project/adapters/qwen3-8b-plain-english`. Die API-Modell-ID `qwen3-8b` wählt das Basismodell, `plain-english` den Adapter; dieses Verhalten entspricht der [vLLM-LoRA-Dokumentation](https://docs.vllm.ai/en/v0.27.0/features/lora/).

Die Adapterdatei auf dem Cluster hat SHA-256 `921755883da0ed6da18be5fd67a161c77046995e2cf55ddfa12087cc7667d215`. Die Prüfsummen der lokalen und auf dem Cluster gespeicherten Train-, Validation- und Testdateien stimmten überein. Die ältere Angabe `training_started: false` im Datensatzmanifest beschreibt den damaligen Vorbereitungsstand, nicht den heutigen Trainingsstatus.

## Aussagegrenzen und nächste Entscheidung

Die Auswertung ist **mit Einschränkungen belastbar**: Die gemessene Länge und Modellzuordnung sind überprüfbar; die Verständlichkeitsbewertung ist eine begründete Einschätzung an einer kleinen Stichprobe. Es sind nur zehn Themen, ein Seed je Einstellung und keine unabhängige Nutzerstudie. Die Fragen sind vom Fine-Tuning zurückgehalten, könnten dem Basismodell aber bereits aus seinem Vortraining bekannt sein. Der zweite Temperature-Durchlauf wurde nach dem ersten Ergebnis ergänzt und ist keine vorab registrierte Studie.

Für diesen Adapter unter diesen Einstellungen gibt es **keinen überzeugenden Nachweis einer verlässlichen sprachlichen Vereinfachung**. Die Ursache ist noch nicht geklärt. Vor einem weiteren Training sollten die tatsächlich trainierten Antworttexte, die Verarbeitung des Chat-Formats und das Maskieren der zu lernenden Antworttokens geprüft werden. Änderungen an Daten oder Einstellungen sollten auf dem Validierungssatz entschieden werden. Für einen abschließenden Nachweis nach weiteren Anpassungen ist ein neuer, unangetasteter Testsatz sinnvoll, da dieser Vergleich nun bekannt ist.

## Gespeicherte Belege

- [Vollständige Antworten bei Temperature 0,2](2026-09-08-vllm-temp02/comparison.md)
- [Kennzahlen und Bewertung pro Frage bei Temperature 0,2](2026-09-08-vllm-temp02/summary.json)
- [Unveränderte API-Anfragen und Antworten bei Temperature 0,2](2026-09-08-vllm-temp02/requests.jsonl)
- [Vollständige Antworten beim ersten Durchlauf mit Temperature 0](2026-09-08-vllm/comparison.md)
- [Kennzahlen des ersten Durchlaufs](2026-09-08-vllm/summary.json)
- [Gesicherter Trainingszustand und Adapterkonfiguration](2026-09-08-vllm/training-evidence.json)

Beide Durchlaufordner enthalten außerdem die Einstellungen, Modellliste, Prüfsummen, anonymisierte Bewertung und Zuordnung sowie eine Kopie des verwendeten Auswertungscodes. Die ausführbaren Skripte liegen im Trainingsverzeichnis als `evaluate_plain_english_vllm.py` und `summarize_plain_english_eval.py`. Das erste zeigt ohne `--run` nur den Plan; das zweite berechnet die Kennzahlen ausschließlich aus gespeicherten Antworten und erzeugt keine Modellanfragen.
