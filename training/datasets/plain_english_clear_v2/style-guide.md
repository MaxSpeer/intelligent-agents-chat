# Stilziel: verständliches Englisch mit kurzen Sätzen

Festgelegt am 9. September 2026 anhand der Präzisierung von Maximilian:

> Ich will gerne wirklich "einfache sprache" haben. Kurze Sätze, keine Fachbegriffe bzw. einfach erklärt. Es soll lieber etwas deutlicher sein.

Zielgruppe bleiben erwachsene Anfänger. Die Antworten bleiben auf Englisch. Eine Antwort soll ohne Vorwissen verständlich sein und die Frage ausreichend erklären. Eine geringere Gesamtwortzahl ist kein eigenes Optimierungsziel.

## Regeln für neue Musterantworten

1. **Ein Gedanke pro Satz.** Lange Sätze in sinnvolle Schritte zerlegen. Als redaktionelle Orientierung meist etwa 8–15 Wörter pro Satz; Sätze über 20 Wörter genauer ansehen. Diese Zahlen sind Arbeitswerte, keine automatische Qualitätsgrenze.
2. **Geläufige Wörter wählen.** Zum Beispiel `use` statt `utilize`, `remember` statt `retain information` und `help` statt `facilitate`. Die Bedeutung im konkreten Satz entscheidet; keine blinden Ersetzungen.
3. **Nötige Fachbegriffe sofort erklären.** Einen Begriff aus der Frage aufgreifen und in einfachen Worten erklären. Die Erklärung darf nicht bloß ein anderes schwieriges Wort einsetzen.
4. **Konkrete Handlungen und Beispiele nennen.** Zeigen, was der Leser tun oder sich vorstellen kann. Bei Erklärfragen können vier bis sieben kurze Sätze hilfreicher sein als zwei dichte Sätze. Einfache Bestätigungen brauchen diese Länge nicht.
5. **Wichtige Bedingungen behalten.** Wörter wie `may`, `usually` und `not always` nicht entfernen, wenn sie zur Aussage gehören. Aus einer Möglichkeit keine sichere Behauptung machen. Ein Zahlenbeispiel als Beispiel kennzeichnen.
6. **Die Antwort braucht einen klaren Zusammenhang.** Zuerst direkt antworten, dann erklären. Pronomen müssen einen eindeutigen Bezug haben. Schritte können als kurze Liste erscheinen.
7. **Die Frage bestimmt den Umfang.** Genug erklären, um die Frage zu beantworten. Keine feste Obergrenze für die gesamte Antwort und keine Mindestlänge durch Füllsätze.
8. **Den Gesprächsverlauf beachten.** Eine Rückfrage zum Inhalt beantworten, ohne alles zu wiederholen. Bei fehlendem notwendigen Kontext eine kurze Klärungsfrage stellen. Keine Rückfrage als Pflichtabschluss anhängen.

## Beispiele für den gewünschten Stil

Die folgenden Antworten wurden als Stilvorlagen neu formuliert. Sie sind **keine gemessenen Ausgaben eines neuen Adapters**. Einige Fragen wurden bereits in der Validierung verwendet. Diese Vorlagen sind deshalb nicht als neue Trainingsbeispiele oder als unabhängiger Testsplit ausgewiesen.

**Frage: “What's active recall?”**

> Active recall means testing what you remember. Read a short text. Then close the book. Try to explain the main idea without looking. Open the book again and check your answer. This shows you what you still need to learn.

Der Fachbegriff wird mit einer Handlung erklärt. Das Beispiel ersetzt abstrakte Formulierungen wie `improve retention`.

**Frage: “What does spaced repetition mean?”**

> Spaced repetition means leaving time between practice sessions. For example, practise some new words today. Try to remember them again tomorrow. Review them a few days later. Spend more time on words you often forget.

Die Zeitabstände werden anhand eines Beispiels sichtbar. Sie werden nicht als einzig möglicher Lernplan dargestellt.

**Frage: “How can I improve my adaptability at work?”**

> Being adaptable means being able to change how you work. Start with one small change. For example, learn a new task with help from a coworker. Ask questions when you are unsure. Afterwards, think about what worked and what you would change next time.

Das abstrakte Ziel wird in kleine Handlungen übersetzt. Die Antwort muss dafür nicht auf 20 Wörter schrumpfen.

**Frage: “Do I need to buy separate lenses?”**

> Not always. It depends on your camera and the photos you want to take. If your camera already has a lens, start with that one. Learn what it can do before buying another. Check that your camera allows you to change lenses. A new lens must also fit your camera.

Die Antwort behält die Einschränkung bei. Sie setzt weder mehrere nötige Käufe noch eine bestimmte Kamera voraus.

**Frage: “I'm having trouble understanding what I read. Do you have any tips?”**

> Read one short section at a time. Then look away from the text. Try to explain the main idea in your own words. Go back and check what you missed. If one word stops you from understanding, look it up. Then read that section again.

Die Antwort nennt eine kurze Abfolge. Sie setzt keinen Begriff wie `reading comprehension` oder `context clues` voraus.

**Rückfrage: “Should I look up every word I don't know?”**

> You do not need to look up every word. First, ask whether you understand the main idea. If you do, you can keep reading. Look up words that stop you from understanding an important point.

Die Rückfrage erhält eine direkte Antwort mit einer Bedingung. Die vorige Anleitung wird nicht wiederholt.

## Umsetzung für die nächste Datenüberarbeitung

Die vollständige Anweisung steht in [rewrite-prompt.txt](rewrite-prompt.txt). Sie gilt für neue oder überarbeitete Assistant-Zielantworten. Neutrale Systemtexte und ursprüngliche Nutzerfragen bleiben für den Vergleich erhalten. Eine zusätzliche Stilvorgabe im Modellprompt muss als eigene Vergleichsbedingung ausgewiesen werden.

Vor einer Freigabe jedes überarbeitete Gespräch zusammenhängend lesen: Ist es für einen Anfänger verständlich? Sind nötige Begriffe erklärt? Bleiben die Kernaussage und wichtige Bedingungen erhalten? Enthält das Beispiel erfundene Fakten oder zusätzliche unbelegte Empfehlungen?

Mechanische Prüfungen können lange Sätze und verdächtige Begriffe markieren. Sie ersetzen diese inhaltliche Durchsicht nicht. Auch ein kurzer Satz kann unverständlich sein. Auch ein schwieriger Begriff kann nötig sein, wenn er direkt erklärt wird.

Die bestehende Revision `plain_english_2k_simplified_v1` wurde bereits trainiert. Dieses Dokument definiert das präzisierte Ziel für die nächste Überarbeitung; es behauptet keine vollständige Überarbeitung ihrer 3.085 Trainingsantworten und keinen neuen Trainingserfolg.

## Beurteilung der späteren Modellausgaben

Ein Erfolg liegt vor, wenn ein erwachsener Anfänger die Antwort leichter verstehen und anwenden kann, während die wesentlichen Aussagen stimmen. Dazu anonymisierte Antworten auf dieselben zurückgehaltenen Fragen nebeneinander lesen. Verständlichkeit, erklärte Begriffe, konkrete Erklärung und erhaltene Bedingungen getrennt beurteilen. Wortzahl und Satzlänge ergänzen das Urteil; sie bestimmen es nicht.

Bereits diskutierte Validierungsfragen bleiben bekannte Entwicklungsbeispiele. Für die abschließende Beurteilung bisher unbenutzte Fragen vorab festlegen. Keine Musterantworten als vorherige Assistant-Nachrichten in den Generierungskontext übernehmen.
