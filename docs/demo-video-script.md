# Demo-Video-Skript

1. Architektur-Überblick (~30s) — UI, Modellauswahl. Architektur Grafik nochmal zeigen.

**Qwen3.5 9B** als Standard-Hauptmodell mit Tools und optionalem Thinking zeigen.
Qwen3 8B und seine zwei Adapter dienen dem Fine-Tuning-Vergleich in Punkt 2.

2. Fine-Tuning (LoRA) — Basismodell, Conspiracy und Simple English

Neues Projekt "Flat Earth"
Neuer Chat

```
Who built the pyramids?
```

```
I think it were just many slaves"
```


**Simple English — gleicher Prompt, zwei neue Chats**

Je einen frischen Chat mit **Qwen3 8B** und **Qwen3 8B (Simple English)** öffnen.

```text
How does Attention work in LLMs?```


3. Memory — Projekt „Lisbon Trip", zwei fertige Konversationen:

Für diesen und alle folgenden Agent-Demos wieder **Qwen3.5 9B** auswählen,
damit Tools, Websearch, Subagents und RAG verfügbar sind.

"Lisbon Trip Budget" (Budget-Fakt: 800€/Person)
"What best time to travel to lisbon?" (unabhängiges Thema)

Neuer Chat mit prompt

```
What's a good area to stay in Lisbon for nightlife and easy metro access and according to our budget?
```

Bei dem Prompt memory und auch schon Websearch erklären

4. Websearch — größere Frage.

Neuer Chat unter "General"

```
How did the population of Berlin develop after the reunification 1990?"
```

5. Subagents 

Subagents werden schon im Websearch und in der Compression verwendet. Zusätzlich als tool "delegate_task"

Limitation erwähnen: Wird selten von selber verwendet da das LLM den Bedarf dafür nicht hat. Wenn subagents tools verwenden könnten wäre das wahrscheinlich besser -> future work

Neuer Chat unter "General"

Prompt der dem agent *explizit* sagt dass es subagents benutzen soll. Anders würde das Modell die noch nicht benutzen.


```
I want to celebrate my birthday. I need different things: A plan for how to organize the event including location, food, and activites. A rough budget calculation needed for 20 people, and a funny poem about a guy named Michael. Use subagents to split the work.
```


6. RAG — Projekt „Bioinformatics" (9 Vorlesungsfolien, fertig indexiert). 

Einmal zeigen wie man ein neues dokument und vielleicht ein pdf hochladen kann.

Dann neuen Chat mit komplexerer Frage die search_documents ab besten mehrfach benutzen soll
```
What does the uploaded lecture notes tell about Motifs and how are they used?
```

```
Compare the two approaches of Genome Assembly mentioned in the uploaded slides
```

7. Context Management

Kontext Grafik zeigen
Memory budget nochmal erwähnen

Token Usage tooltip zeigen
Keine Tools in Chat History und kein Reasoning
- History bekommt nur eine Zeile mit tool call, parametern, binärer status (ok/fail) und ID 

"recall tool output" gibt tool output anhand von ID wieder

zeigen an Beispiel Prompt, selber chat wie population search in "General"
```
You fetched the german wikipedia, what did you find there?
```

9. Context Compression — Projekt „Computing History", ein langer, echter Chat 

Context Usage Icons zeigen, werden immer mehr. 
- letzte Nachricht hat ~ 22k verbraucht
- Wir reservieren output von 8k -> würde jetzt noch passen, aber wir estimaten die tokens und nehmen nicht die echten werte. Hier überschätzen wir, daher wird direkt bei der nächsten Nachricht compacted.

Eine kurze neue Nachricht in diese Konversation schreiben → sollte sofort „🗜️ Compacting..." zeigen.


```
How does research in robotics influence AI research?
```
