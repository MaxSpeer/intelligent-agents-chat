"""Minimal NiceGUI entry point for the Intelligent Agents project."""

from nicegui import ui


ui.add_head_html(
    """
    <style>
        :root {
            --ink: #172033;
            --muted: #657089;
            --primary: #635bff;
            --primary-dark: #4b43df;
            --surface: rgba(255, 255, 255, 0.82);
        }

        body {
            color: var(--ink);
            background:
                radial-gradient(circle at 10% 15%, rgba(99, 91, 255, .18), transparent 32rem),
                radial-gradient(circle at 90% 80%, rgba(38, 198, 218, .16), transparent 34rem),
                #f7f8fc;
        }

        .glass-card {
            width: min(92vw, 760px);
            border: 1px solid rgba(255, 255, 255, .9);
            border-radius: 28px;
            background: var(--surface);
            box-shadow: 0 24px 70px rgba(34, 42, 74, .12);
            backdrop-filter: blur(18px);
        }

        .eyebrow {
            color: var(--primary);
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
        }

        .hero-title {
            margin: 0;
            font-size: clamp(2.25rem, 8vw, 4.8rem);
            font-weight: 800;
            line-height: .98;
            letter-spacing: -.055em;
        }

        .hero-copy {
            max-width: 34rem;
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.7;
        }

        .primary-button {
            border-radius: 14px;
            background: var(--primary) !important;
            box-shadow: 0 12px 28px rgba(99, 91, 255, .28);
            transition: transform .2s ease, box-shadow .2s ease;
        }

        .primary-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 32px rgba(99, 91, 255, .36);
        }

        .status-dot {
            width: .55rem;
            height: .55rem;
            border-radius: 999px;
            background: #20b486;
            box-shadow: 0 0 0 5px rgba(32, 180, 134, .12);
        }
    </style>
    """,
    shared=True,
)


def say_hello() -> None:
    """Show a small confirmation that the starter is interactive."""
    ui.notify("Hello, intelligent world!", type="positive", position="top")


@ui.page("/")
def index() -> None:
    """Render the starter landing page."""
    with ui.column().classes("min-h-screen w-full items-center justify-center p-6"):
        with ui.card().classes("glass-card p-8 sm:p-12 gap-8"):
            with ui.row().classes("w-full items-center justify-between"):
                with ui.row().classes("items-center gap-3"):
                    ui.icon("auto_awesome", size="sm").style("color: var(--primary)")
                    ui.label("AGENT LAB").classes("eyebrow")
                with ui.row().classes("items-center gap-3"):
                    ui.element("span").classes("status-dot")
                    ui.label("Starter ready").classes("text-sm text-slate-500")

            with ui.column().classes("gap-5"):
                ui.label("Hello, world.").classes("hero-title")
                ui.label(
                    "A clean starting point for the Intelligent Agents project, "
                    "built with Python, uv, and NiceGUI."
                ).classes("hero-copy")

            with ui.row().classes("items-center gap-4"):
                ui.button("Say hello", icon="waving_hand", on_click=say_hello).props(
                    "unelevated no-caps size=lg"
                ).classes("primary-button px-5")
                ui.label("Ready for your first agent feature").classes("text-sm text-slate-400")


def main() -> None:
    """Start the NiceGUI development server."""
    ui.run(title="Agent Lab", favicon="✨", reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
