# Späterer Vergleich: Wird die Modellsprache wirklich einfacher?

Die folgenden Fragen stammen aus dem vorhandenen Testsplit. Ihre Formulierungen wurden nicht geändert. Sie eignen sich als kleine feste Demonstration; ein sichtbarer Unterschied ist noch nicht nachgewiesen. Die Themen überschneiden sich zum Teil mit Trainingsthemen. Diese Auswahl prüft deshalb den Antwortstil bei zurückgehaltenen Fragen, nicht die Übertragung auf ausschließlich neue Themen.

Erst den Checkpoint anhand der **Validierungsantworten** wählen und Einstellungen festlegen. Danach die Testfragen auswerten; nicht mit den Testergebnissen die nächste Epoche auswählen.

| Frage für einen frischen Chat | Test-ID | Woran man eine gute einfache Antwort erkennt |
|---|---|---|
| I'm learning about chemistry in school, and I was wondering what's the difference between a mixture and a solution. | `plain2k-everyday-test_sft-0018-dialogue` | Erklärt „dissolve“, nennt ein Beispiel und sagt korrekt, dass eine Lösung auch ein Gemisch ist. |
| I'm learning about energy in school and I'm a bit confused about potential energy. Can you explain it to me? | `plain2k-everyday-test_sft-0052-dialogue` | Erklärt gespeicherte Energie anhand eines angehobenen Balls; wenig unerklärter Fachwortschatz. |
| I'm learning about the human body in school, and I was wondering what the circulatory system does. | `plain2k-everyday-test_sft-0033-dialogue` | Beschreibt konkret Bluttransport, Sauerstoff, Nährstoffe und Abfallstoffe. Erklärt schwierige Begriffe. |
| I'm learning about food chains in school. What are producers in a food chain? | `plain2k-everyday-test_sft-0103-dialogue` | Erklärt Produzenten in gewöhnlichen Worten mit Beispielen. Bei der vorhandenen Folgefrage: Nährstoffe im Kreislauf, Energiefluss davon unterscheiden. |
| I'm having some issues with my manager at work. What are some ways to improve our working relationship? | `plain2k-everyday-test_sft-0032-dialogue` | Gibt konkrete Handlungsschritte statt abstrakter Begriffe wie „maintaining open communication“. |
| I have trouble managing my time effectively. Can you give me some tips? | `plain2k-everyday-test_sft-0070-dialogue` | Kurze umsetzbare Schritte ohne Floskeln; bleibt hilfreich statt nur kürzer. |

Die passenden vollständigen Gespräche stehen in `test.jsonl`. Für Folgefragen jeweils denselben Gesprächsablauf bei allen Varianten verwenden und die vom jeweiligen Modell erzeugte Vorgeschichte beibehalten. Nicht versehentlich die trainierten Musterantworten in den Modellkontext kopieren.

Vergleiche drei Bedingungen: Basismodell mit neutralem Systemtext, dasselbe Basismodell mit neuem Adapter und neutralem Systemtext, sowie das Basismodell mit einer ausdrücklichen Bitte um einfache Sprache. Die dritte Bedingung zeigt, ob ein einfacher Prompt denselben Nutzen erreicht. Der neutrale Systemtext steht im Manifest. Modellversion, Thinking-Einstellung, Generierungslimit und übrige Einstellungen müssen gleich sein; für eine reproduzierbare erste Gegenüberstellung eignet sich deterministische Generierung.

Beurteile anonymisierte Antworten nach denselben Kriterien: verständliche Wörter, kurze übersichtliche Sätze, erklärte Fachbegriffe, konkrete Beispiele und vollständige richtige Inhalte. Markiere ausgelassene wichtige Bedingungen oder falsche Aussagen gesondert. Eine kürzere, aber falsche Antwort ist kein Erfolg. Zusätzlich können Wortzahl und Satzlänge berichtet werden. Der genaue Wortlaut der Musterantwort ist kein Bestehenskriterium.

Diese sechs Fragen vorab festlegen und auch Fälle ohne Verbesserung zeigen. Für eine belastbarere Aussage den gesamten Testsplit auswerten und mindestens eine Person die Verständlichkeit beurteilen lassen. Die hier überarbeiteten Testziele sind AI-Musterantworten, keine unabhängigen menschlichen Qualitätsurteile.
