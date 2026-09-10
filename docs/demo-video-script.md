# Demo-Video-Skript

### 1. Architektur-Überblick — UI, Modellauswahl, Modell Reachability Status

**Qwen3.5 9B** als Standard-Hauptmodell mit Tools und optionalem Thinking zeigen.
Qwen3 8B und seine zwei Adapter. 


### 2. Fine-Tuning (LoRA) — Basismodell, Conspiracy und Simple English

**Conspiracy**

Zeigen wie man ein **neues Projekt erstellt** "Flat Earth"

Neuer Chat mit Prompt
```
Who built the pyramids?
```
Zweiter Prompt
```
I think it were just many slaves
```


**Simple English**

Wechseln zu Projekt General.

Je einen frischen Chat mit **Qwen3 8B** und **Qwen3 8B (Simple English)** öffnen.

```text
How does Attention work in LLMs?
```

Denselben Prompt **zweimal** abschicken in zwei eigenen chats, einmal an  **Qwen3 8B** und **Qwen3 8B (Simple English)**.

Zeigen wie unterschiedlich beide antworten.


### 3. Memory — Projekt „Lisbon Trip", zwei fertige Konversationen:

Für diesen und alle folgenden Agent-Demos wieder **Qwen3.5 9B** auswählen mit Thinking und mit Memories.

Zeigen es gibt bereits zwei Chats mit jeweils einem Turn:

- "Lisbon Trip Budget" (Budget-Fakt: 800€/Person)

- "What best time to travel to lisbon?" (unabhängiges Thema)

Zeigen "Manage Project Memories" dass diese zwei Turns als memory gespeichert sind. Insbesondere das geplante Budget für die Reise.

Neuer Chat mit prompt:
```
What's a good area to stay in Lisbon for nightlife and easy metro access and according to our budget?
```
Modell sollte über die Memory schon das Budget kennen. Hier einmal den Tool Trace zeigen und erklären: inklusive Memory und Websearch
Achtung: Der "memory" Eintrag im trace kommt erst am Ende hinzu wenn die finale Antwort da ist, beim Streaming ist der memory Eintrag noch nicht da (nur ein UI Problem, natürlich wird memory auch vorher gefetched)


### 4. Websearch nochmal mit komplexerer und größere Frage

Neuer Chat unter "General"

```
How did the population of Berlin develop after the reunification 1990?"
```

Zeigen dass es auch für komplexe Recherchen mit wiederholtem Websuchen funktioniert.

### 5. Subagents 

Erklären dass Subagents schon im Websearch und in der Compression verwendet werden. 

Zusätzlich gibt es sie aber auch noch als tool "delegate_task"

Limitation erwähnen: Wird selten von selber verwendet da das LLM den Bedarf dafür nicht hat. Wenn subagents tools verwenden könnten wäre das wahrscheinlich besser -> future work

Neuer Chat unter "General"

Prompt der dem agent *explizit* sagt dass es subagents benutzen soll. Anders würde das Modell die noch nicht benutzen:

```
I want to celebrate my birthday. I need different things: A plan for how to organize the event including location, food, and activites. A rough budget calculation needed for 20 people, and a funny poem about a guy named Michael. Use subagents to split the work.
```

Zeigen dass die drei Subtasks jeweils an Subagents übergeben werden und sequentiell abgearbeitet.


### 6. RAG — Projekt „Bioinformatics" 

Zeigen in der UI "Manage Project Documents"
- Vorlesungsfolien, fertig indexiert 

Einmal zeigen wie man ein neues pdf hochladen kann.
Erwähnen dass es gechunked und embedded wird.

Dann neuen Chat mit komplexerer Frage die search_documents ab besten mehrfach benutzen soll:

```
Compare the two approaches of Genome Assembly mentioned in the uploaded slides
```

Zeigen dass search_documents **(mehrfach)** verwendet wird. Bei meinem Test wurde zuerst search_documents mit **allgemeiner** Query genutzt ("genome assembly methods") und beim zweiten Mal dann mit den **exakten** Methoden Namen in der Query ("De Brujin Graph und OLC Graph").

Erklären dass das LLM selber also implizit als **Judge** und **Query Rewriter** fungiert.


### 7. Context Management

Memory budget nochmal erwähnen (Context Selection)

Token Usage tooltip zeigen. Das sind die echten Token Werte von vLLM reported. Unser Context Assembler hingegen nutzt nur Schätzungen.

Erwähnen dass Tool Output in Chat History nur auf eine Zeile verkürzt wird, nämlich das ein Tool aufgerufen wurde, mit welchen Parametern, ein binärer success/fail, **und eine ID (wichtig für recall_tool_output)**

Dass Tool "recall tool output" gibt tool output anhand von ID wieder

Tool zeigen an Beispiel Prompt, zum Beispiel im selben chat wie die population search Frage in "General"

```
You fetched the german wikipedia, what did you find there?
```
Zeigen dass recall_tool_output verwendet wird.

### 9. Context Compression — Projekt „Computing History", ein langer, echter Chat 

Wechseln in das Projekt "Computing History" und den einen Chat den es schon gibt.

**thinking enablen!!** (damit output budget auf 8k tokens gesetzt wird)

Context Usage Icons zeigen, jede Nachricht werden immer mehr Tokens verbraucht. 

Letzte Nachricht hat ~ 22k verbraucht

Wir reservieren (when thinking enabled ist) immer ein output von etwa 8k Token -> würde hier jetzt noch gerade so passen
**ABER** wir nutzen im Context Assembler keinen echten Tokenizer sondern schätzen die tokens (3 char per token) und nehmen **nicht** die echten werte. Hier überschätzen wir (hab ich getestet), daher denken wir dass es bei der nächsten Nachricht schon nicht reicht und es sollte schon bei der nächsten Nachricht compacted werden.

Eine kurze neue Nachricht in diese Konversation schreiben:

```
How does research in robotics influence AI research?
```

Sollte sofort „🗜️ Compacting..." im Step trace zeigen. Allerdings nur solange die Antwort streamed. Sobald sie fertig ist verschwindet das "Compacting" aus dem Trace, das wir das nicht in die DB schreiben. 

Zeigen dass die neue Nachricht wieder viel weniger Tokens verbraucht hat als die davor.
