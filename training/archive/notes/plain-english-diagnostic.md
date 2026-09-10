# Kleiner Lerntest nach Plain English v2

`diagnose_plain_english.py` vergleicht das Basismodell mit dem gespeicherten finalen v2-Adapter auf fünf ausgewählten, bereits trainierten ersten Fragen: braune Äpfel, Git, blauer Himmel, Schreiben und Dezimal-/Binärzahlen. Es entstehen zehn Antworten. Dieser Test prüft, ob der Stil auf bekannten Beispielen erkennbar ist; er bewertet weder neue Fragen noch den Testdatensatz und wählt keinen besten Checkpoint.

Die Musterantworten werden nur in der Ergebnisdatei daneben angezeigt und niemals an das Modell übergeben. Basis und Adapter erhalten dieselben neutralen Nachrichten, denselben Seed je Frage und dieselben Einstellungen. Die NF4-Quantisierung entspricht dem Training. Das Ausgabelimit beträgt 1.024 Tokens statt 2.048 bei der vorausgegangenen Validierung; abgeschnittene Antworten werden markiert.

Das Skript ist lokal ohne Modellbibliotheken prüfbar und startet standardmäßig nur eine Dateiprüfung. Erst `--run` lädt das Modell und erzeugt Antworten. Es erstellt keine Slurm-Zuteilung und führt kein Training aus. Die fünf IDs sind vor der Generierung festgelegt, nach unterschiedlichen Themen ausgewählt und keine Zufallsstichprobe.

In der vorhandenen GPU-Shell:

```bash
PLAIN_EN_ROOT="/sc/projects/sci-lippert/intelligent-agents/project_matthias_max"
export UV_PROJECT_ENVIRONMENT="$PLAIN_EN_ROOT/venvs/training-plain-english-maximilian.speer"
export HF_HOME="$PLAIN_EN_ROOT/models/huggingface"
export TOKENIZERS_PARALLELISM=false
cd "$PLAIN_EN_ROOT/code/intelligent-agents-chat/training"

"$UV_PROJECT_ENVIRONMENT/bin/python" diagnose_plain_english.py --check
```

Erwartet wird `Diagnostic preflight passed. No model loaded and no inference started.` Die Dateiprüfung bestätigt nicht, dass die GPU-Zuteilung noch läuft. Vor einem späteren Start die eigene Restzeit prüfen:

```bash
squeue --account=sci-lippert-intelligent-agents --user="$USER" -o "%.18i %.20j %.8T %.10M %.10L %.12N"
```

Die folgenden Zeilen startet der Nutzer selbst, solange die GPU-Zuteilung aktiv ist:

```bash
mkdir -p "$HOME/plain-english-logs"
set -o pipefail

"$UV_PROJECT_ENVIRONMENT/bin/python" -u diagnose_plain_english.py --run 2>&1 |
  tee -a "$HOME/plain-english-logs/diagnostic-v2-${SLURM_JOB_ID}.log"
```

Pro Frage erscheint `Diagnostic: N/5 questions complete (base + adapter)`. Am Ende folgt `Diagnostic complete. Read: .../comparison.md`. Antworten und Metadaten liegen in einem neuen Verzeichnis unter `$HOME/plain-english-results/diagnostic-v2-...`. Die Adaptergewichte werden nicht verändert. Bei einem Abbruch durch das Zeitlimit können bereits vollständige Fragepaare aus `comparison.jsonl` ausgewertet werden; ein unvollständiger Lauf darf nicht als kompletter Vergleich gelten.
