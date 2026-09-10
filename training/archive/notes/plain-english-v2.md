# Plain English v2: korrigierten Lauf selbst starten

Der Code ist vorbereitet. Diese Anleitung und der Prüfmodus starten keinen
Cluster-Job. Das eigentliche Training startest du ausdrücklich mit `--run` in
deiner eigenen GPU-Zuteilung.

## Was sich ändert

Die erste Version markierte in den 13 Trainingsdialogen mit Rückfragen jeweils
vier Tokens des nächsten Nutzerbeitrags als Lernziel. Qwen3 fügt bei der letzten
Assistant-Antwort einen leeren Thinking-Block ein, bei früheren Antworten im
Verlauf jedoch nicht. Die frühere Berechnung über Präfixlängen übersah das.

Jetzt bildet jede Antwort ein eigenes Beispiel. Frühere Nachrichten bleiben
Kontext und werden vollständig vom Loss ausgeschlossen. Die Prüfung vergleicht
die tatsächlich markierten Tokens mit der vollständigen Zielantwort und ihrem
Abschlusszeichen. Der leere Thinking-Block ist ebenfalls maskierter Kontext.

Die Trainingsdaten bleiben gleich. Aus 80 Dialogen werden 93 Antwortbeispiele;
aus zehn Validierungsdialogen werden 13. Damit ergeben sich bei unverändertem
Batch und drei Epochen **18 statt 15 Optimierungsschritte**. Lernrate `1e-4`,
LoRA-Rang 8, Alpha 16, Dropout 0,05, Attention-Projektionen und das Basismodell
bleiben gleich. Das ist ein kontrollierter Versuch der korrigierten Aufbereitung,
kein Beweis, dass der gefundene Fehler allein das erste Ergebnis erklärt.

## 1. Umgebung und Daten prüfen

Auf einem Rechenknoten innerhalb deiner Zuteilung. Auf dem Login-Knoten `lx01`
ist der Projektspeicher mit `noexec` eingebunden; dort lässt sich dieses Python
nicht starten. Wenn du noch keine Zuteilung hast, zuerst Schritt 2 ausführen.

```bash
PLAIN_EN_ROOT="/sc/projects/sci-lippert/intelligent-agents/project_matthias_max"
export UV_PROJECT_ENVIRONMENT="$PLAIN_EN_ROOT/venvs/training-plain-english-maximilian.speer"
export HF_HOME="$PLAIN_EN_ROOT/models/huggingface"
export TOKENIZERS_PARALLELISM=false
cd "$PLAIN_EN_ROOT/code/intelligent-agents-chat/training"

"$UV_PROJECT_ENVIRONMENT/bin/python" train_plain_english_v2.py --check
```

Erwartet: `Preflight passed`, 93 Trainingsbeispiele, 18 Optimierungsschritte und
`all_supervised_text_matches_reference_answer_and_end_marker: true`.
Es werden nur Tokenizer-Dateien geladen. Der Tokenizer-Snapshot stammt aus dem
bereits geprüften Verzeichnis des älteren Adapters; dessen Modellgewichte werden
nicht geladen. Das Training beginnt mit dem frischen, festgelegten
`Qwen/Qwen3-8B` und erstellt einen neuen Adapter.

Der Standard-Zielordner ist:

```text
/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/adapters/qwen3-8b-plain-english-v2
```

Wenn dieser Ordner bereits existiert, bricht das Skript ab. Für einen weiteren
Versuch einen neuen Ordner über `--output-dir` wählen. Der v1-Adapter bleibt
erhalten.

## 2. Eine freie GPU in deiner Zuteilung verwenden

Das Training benötigt genau eine sichtbare GPU mit BF16-Unterstützung, etwa eine
A40. Vor dem Start in der GPU-Shell prüfen:

```bash
hostname
echo "Job-ID: ${SLURM_JOB_ID:-nicht gesetzt}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

Eine reine SSH-Verbindung ist kein Ersatz für eine laufende GPU-Zuteilung.
Falls du eine neue Zuteilung brauchst, verwende auf dem Login-Knoten deinen
bekannten interaktiven Aufruf mit dem Projektkonto:

```bash
srun \
  --account=sci-lippert-intelligent-agents \
  --partition=gpu-interactive \
  --job-name=plain-english-v2 \
  --nodes=1 --ntasks=1 --gpus=1 \
  --exclude=gx26,ga01,ga02 \
  --cpus-per-task=8 --mem=64G --time=02:00:00 \
  --pty bash
```

Die Ausschlussliste stammt vom ersten erfolgreichen A40-Lauf; sie garantiert
nicht denselben GPU-Typ bei einer neuen Zuteilung. Danach die GPU prüfen und den
Umgebungsblock aus Schritt 1 in dieser neuen Shell wiederholen. Der Runner lehnt
ältere GPUs ohne BF16-Unterstützung ab.

Wenn dein vLLM-Server dieselbe GPU belegt, muss sie vor dem Training freigegeben
werden oder du brauchst eine andere eigene Zuteilung. Das Skript beendet keine
Prozesse und bricht bei weniger als 16 GiB freiem GPU-Speicher vor dem Laden der
Modellgewichte ab.

## 3. Training ausdrücklich starten

Erst nach erfolgreicher Vorprüfung, in deiner GPU-Shell:

```bash
mkdir -p "$HOME/plain-english-logs"
set -o pipefail
"$UV_PROJECT_ENVIRONMENT/bin/python" -u train_plain_english_v2.py --run 2>&1 |
  tee -a "$HOME/plain-english-logs/v2-${SLURM_JOB_ID}.log"
```

Der eigene Logordner vermeidet das Schreibrechteproblem des ersten Laufs.
Das Skript reserviert einen neuen Ausgabeordner und speichert Konfiguration,
Code, Daten-Prüfsummen und Trainings-/Validierungsdateien zur Nachvollziehbarkeit.

## 4. Was während des Laufs passiert

Vor dem Training werden die zehn Validierungsdialoge zweimal frei beantwortet:
einmal vom Basismodell mit neutralem Prompt, einmal vom Basismodell mit einer
klaren Plain-English-Anweisung. Bei deaktiviertem Adapter werden hierfür dieselben
4-Bit-Basisgewichte verwendet wie beim Training.

Nach jeder Epoche werden Loss und Checkpoint gespeichert. Der zugehörige Adapter
beantwortet anschließend dieselben Validierungsfragen mit dem neutralen Prompt.
Rückfragen verwenden jeweils die gerade erzeugte eigene Antwort. Keine Musterantwort
wird als Gesprächsverlauf an die Generierung weitergereicht. Die ursprünglichen
Testfragen werden nicht generiert oder zur Checkpoint-Auswahl verwendet.

Generierung: Thinking aus, Temperature 0,2, maximal 2.048 neue Tokens, feste Seeds
je Frage und Runde. Diese Kontrolle der Seeds verändert nicht den Zufallszustand
des anschließenden Trainings. Die zusätzlichen Antworten dauern deutlich länger
als das reine Optimieren des ersten 91-Sekunden-Laufs. Verwende eine Zuteilung mit
ausreichender Restlaufzeit; der zweite Lauf wurde noch nicht zeitlich vermessen.

## 5. Checkpoint anhand der Antworten wählen

Alle drei Checkpoints bleiben erhalten, voraussichtlich `checkpoint-6`,
`checkpoint-12` und `checkpoint-18`. Ihre Antworten stehen im neuen Adapterordner:

```text
validation_samples/base-neutral.md
validation_samples/base-style-prompt.md
validation_samples/checkpoint-6.md
validation_samples/checkpoint-12.md
validation_samples/checkpoint-18.md
```

Daneben stehen JSONL-Rohdaten und Metadaten mit Tokenlimits, Wortlängen und
abgeschnittenen Antworten. Das Modell im Wurzelordner entspricht der **letzten
Epoche**, nicht automatisch dem besten Stil. `run.json` endet deshalb mit
`training_complete_checkpoint_selection_pending`.

Nach dem Lauf prüfen wir gemeinsam:

1. Bleibt die Antwort sachlich richtig und beantwortet sie die Frage?
2. Sind Wörter und Sätze für Anfänger verständlich?
3. Werden nötige Fachbegriffe und Zusammenhänge erklärt?
4. Helfen Beispiele, ohne unnötige Nebeninformationen hinzuzufügen?
5. Bringt der Adapter einen Vorteil gegenüber einer bloßen Stil-Anweisung?

Erst danach wird ein Checkpoint für vLLM ausgewählt. Die laufende Chat-Konfiguration
wird durch dieses Training nicht umgestellt. Für einen abschließenden Vergleich
nach weiteren Anpassungen verwenden wir neue, bis dahin unangetastete Testfragen.

## Tests der Aufbereitung ohne GPU

Im Repository, mit dem vorhandenen Trainings-Python. Dabei werden keine echten
Modellgewichte heruntergeladen und kein Training ausgeführt. Ein zusätzlicher
Integrationstest nutzt ein winziges, zufällig initialisiertes CPU-Modell, um die
installierten Qwen-/PEFT-Generierungsaufrufe zu prüfen.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
QWEN_TOKENIZER_PATH="$PLAIN_EN_ROOT/adapters/qwen3-8b-conspiracy" \
  "$UV_PROJECT_ENVIRONMENT/bin/python" -m unittest discover -s tests -v
```

Dieser letzte Befehl wird aus dem Verzeichnis `training/` ausgeführt. Die Tests
prüfen insbesondere Rückfragen, vollständig maskierten Kontext, Abschlusszeichen,
die Ablehnung zu langer Beispiele und die Wiederherstellung des Zufallszustands
nach der Validierung.
