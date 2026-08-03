"""NiceGUI chat interface backed by SQLite and selectable model profiles."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from time import monotonic

from nicegui import ui

from intelligent_agents_chat.config import Settings
from intelligent_agents_chat.database import (
    DEFAULT_CONVERSATION_TITLE,
    ChatRepository,
    Conversation,
    Message,
)
from intelligent_agents_chat.llm import LLMError, VLLMGateway


settings = Settings.from_env()
repository = ChatRepository(settings.database_path)
repository.initialize()
gateway = VLLMGateway(settings)
active_generations: set[str] = set()


ui.add_head_html(
    """
    <style>
        :root {
            --ink: #172033;
            --muted: #657089;
            --primary: #635bff;
            --primary-dark: #4b43df;
            --surface: rgba(255, 255, 255, 0.88);
            --surface-solid: #ffffff;
            --line: rgba(104, 117, 147, 0.16);
            --sidebar: rgba(245, 246, 251, 0.92);
        }

        html, body, #q-app {
            min-height: 100%;
        }

        .nicegui-content {
            padding: 0 !important;
        }

        body {
            color: var(--ink);
            background:
                radial-gradient(circle at 12% 5%, rgba(99, 91, 255, .13), transparent 28rem),
                radial-gradient(circle at 88% 90%, rgba(38, 198, 218, .11), transparent 32rem),
                #f8f9fc;
        }

        .app-shell {
            display: grid;
            grid-template-columns: 290px minmax(0, 1fr);
            width: 100%;
            height: 100vh;
            overflow: hidden;
        }

        .sidebar {
            min-width: 0;
            height: 100vh;
            padding: 1.25rem 1rem;
            gap: 1rem;
            overflow: hidden;
            border-right: 1px solid var(--line);
            background: var(--sidebar);
            backdrop-filter: blur(22px);
        }

        .brand-mark {
            display: grid;
            width: 2.35rem;
            height: 2.35rem;
            place-items: center;
            border-radius: .85rem;
            color: white;
            background: linear-gradient(135deg, var(--primary), #857fff);
            box-shadow: 0 10px 24px rgba(99, 91, 255, .28);
        }

        .brand-name {
            font-size: .96rem;
            font-weight: 800;
            letter-spacing: -.02em;
        }

        .eyebrow {
            color: var(--muted);
            font-size: .68rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
        }

        .new-chat-button {
            width: 100%;
            min-height: 2.8rem;
            border-radius: .9rem;
            color: white !important;
            background: var(--primary) !important;
            box-shadow: 0 10px 24px rgba(99, 91, 255, .22);
        }

        .conversation-list {
            min-height: 0;
            flex: 1 1 auto;
            gap: .4rem;
            overflow-y: auto;
            padding-right: .2rem;
        }

        .conversation-item {
            width: 100%;
            min-width: 0;
            flex-wrap: nowrap;
            gap: .25rem;
            align-items: center;
        }

        .conversation-select {
            min-width: 0;
            flex: 1 1 auto;
            justify-content: flex-start;
            overflow: hidden;
            border-radius: .78rem;
            color: var(--muted) !important;
        }

        .conversation-select .q-btn__content {
            display: block;
            overflow: hidden;
            text-align: left;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .conversation-select.active {
            color: var(--primary-dark) !important;
            background: rgba(99, 91, 255, .1) !important;
        }

        .conversation-delete {
            color: #9aa2b5 !important;
            opacity: 0;
            transition: opacity .15s ease, color .15s ease;
        }

        .conversation-item:hover .conversation-delete,
        .conversation-delete:focus {
            opacity: 1;
        }

        .conversation-delete:hover {
            color: #d14f68 !important;
        }

        .sidebar-footer {
            padding: .9rem;
            border: 1px solid var(--line);
            border-radius: .9rem;
            background: rgba(255, 255, 255, .58);
        }

        .main-panel {
            min-width: 0;
            height: 100vh;
            gap: 0;
            overflow: hidden;
        }

        .chat-header {
            z-index: 2;
            width: 100%;
            min-height: 5rem;
            flex-wrap: nowrap;
            padding: .85rem clamp(1rem, 3vw, 2rem);
            border-bottom: 1px solid var(--line);
            background: rgba(255, 255, 255, .72);
            backdrop-filter: blur(18px);
        }

        .chat-title {
            max-width: min(42vw, 38rem);
            overflow: hidden;
            font-size: 1.05rem;
            font-weight: 750;
            letter-spacing: -.02em;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .model-select {
            width: min(15rem, 27vw);
        }

        .status-dot {
            width: .5rem;
            height: .5rem;
            border-radius: 999px;
            background: #20b486;
            box-shadow: 0 0 0 5px rgba(32, 180, 134, .11);
        }

        .message-scroll {
            width: 100%;
            min-height: 0;
            flex: 1 1 auto;
        }

        .message-column {
            width: min(100%, 900px);
            min-height: 100%;
            margin: 0 auto;
            padding: 2rem clamp(1rem, 4vw, 2.5rem) 1.5rem;
            gap: 1.15rem;
        }

        .chat-message {
            width: 100%;
        }

        .chat-message .q-message-text {
            max-width: min(44rem, 82vw);
            border-radius: 1rem;
            box-shadow: 0 8px 22px rgba(34, 42, 74, .07);
        }

        .chat-message .q-message-text-content {
            line-height: 1.58;
        }

        .q-message-sent .q-message-text {
            color: white;
            background: linear-gradient(135deg, var(--primary), #7770ff) !important;
        }

        .q-message-received .q-message-text {
            color: var(--ink);
            border: 1px solid var(--line);
            background: var(--surface-solid) !important;
        }

        .streaming-content {
            max-width: min(42rem, 76vw);
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            line-height: 1.58;
        }

        .empty-state {
            width: min(100%, 36rem);
            margin: auto;
            padding: 2.5rem;
            align-items: center;
            gap: .8rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, .9);
            border-radius: 1.5rem;
            background: var(--surface);
            box-shadow: 0 20px 52px rgba(34, 42, 74, .09);
            backdrop-filter: blur(18px);
        }

        .empty-icon {
            display: grid;
            width: 3.4rem;
            height: 3.4rem;
            place-items: center;
            border-radius: 1.1rem;
            color: var(--primary);
            background: rgba(99, 91, 255, .1);
        }

        .composer-area {
            z-index: 2;
            width: 100%;
            padding: .75rem clamp(1rem, 4vw, 2.5rem) 1.2rem;
            background: linear-gradient(to top, #f8f9fc 72%, rgba(248, 249, 252, 0));
        }

        .composer-row {
            width: min(100%, 850px);
            min-height: 3.7rem;
            margin: 0 auto;
            flex-wrap: nowrap;
            gap: .4rem;
            padding: .45rem .5rem .45rem 1rem;
            border: 1px solid var(--line);
            border-radius: 1.15rem;
            background: white;
            box-shadow: 0 14px 36px rgba(34, 42, 74, .11);
        }

        .composer-input {
            min-width: 0;
            flex: 1 1 auto;
        }

        .composer-input .q-field__control:before,
        .composer-input .q-field__control:after {
            border: 0 !important;
        }

        .send-button {
            color: white !important;
            background: var(--primary) !important;
        }

        .stop-button {
            color: #d14f68 !important;
            background: rgba(209, 79, 104, .1) !important;
        }

        @media (max-width: 720px) {
            .app-shell {
                grid-template-columns: 1fr;
                grid-template-rows: 15rem minmax(0, 1fr);
            }

            .sidebar {
                width: 100%;
                height: 15rem;
                padding: .9rem 1rem;
                gap: .7rem;
                border-right: 0;
                border-bottom: 1px solid var(--line);
            }

            .conversation-list {
                flex-direction: row;
                overflow-x: auto;
                overflow-y: hidden;
            }

            .conversation-delete {
                opacity: 1;
            }

            .conversation-item {
                width: 13rem;
                min-width: 13rem;
            }

            .sidebar-footer {
                display: none;
            }

            .main-panel {
                height: calc(100vh - 15rem);
            }

            .chat-header {
                min-height: 4.4rem;
            }

            .status-copy {
                display: none;
            }
        }
    </style>
    """,
    shared=True,
)


@dataclass(slots=True)
class PageState:
    """State that must remain local to one connected browser page."""

    conversation_id: str
    generating: bool = False
    stop_event: asyncio.Event | None = None
    generation_task: asyncio.Task[None] | None = None


def _conversation_title(message: str, limit: int = 44) -> str:
    compact = " ".join(message.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def _display_time(value: datetime) -> str:
    return value.astimezone().strftime("%H:%M")


def _profile_label(profile_key: str | None) -> str:
    if profile_key is None:
        return "Assistant"
    try:
        return settings.profile(profile_key).label
    except KeyError:
        return profile_key


def _completion_messages(messages: list[Message]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if settings.system_prompt:
        result.append({"role": "system", "content": settings.system_prompt})
    result.extend({"role": message.role, "content": message.content} for message in messages)
    return result


@ui.page("/")
def index() -> None:
    """Render a client-local chat workspace backed by shared SQLite persistence."""
    conversations = repository.list_conversations()
    initial = (
        conversations[0]
        if conversations
        else repository.create_conversation(settings.default_profile_key)
    )
    if initial.model_profile not in settings.profile_options:
        repository.set_model_profile(initial.id, settings.default_profile_key)
        normalized_initial = repository.get_conversation(initial.id)
        if normalized_initial is None:
            raise RuntimeError("Conversation disappeared while updating its model profile")
        initial = normalized_initial
    state = PageState(conversation_id=initial.id)

    def current_conversation() -> Conversation:
        conversation = repository.get_conversation(state.conversation_id)
        if conversation is None:
            conversation = repository.create_conversation(settings.default_profile_key)
            state.conversation_id = conversation.id
        if conversation.model_profile not in settings.profile_options:
            repository.set_model_profile(conversation.id, settings.default_profile_key)
            conversation = repository.get_conversation(conversation.id)
            if conversation is None:
                raise RuntimeError("Conversation disappeared while updating its model profile")
        return conversation

    def set_busy(is_busy: bool) -> None:
        state.generating = is_busy
        if is_busy:
            composer.disable()
            send_button.disable()
            stop_button.enable()
            model_select.disable()
            status_label.set_text("Generating")
        else:
            composer.enable()
            send_button.enable()
            stop_button.disable()
            model_select.enable()
            status_label.set_text("Ready")
            composer.run_method("focus")

    def render_conversation_list() -> None:
        conversation_list.clear()
        with conversation_list:
            for conversation in repository.list_conversations():
                active_class = " active" if conversation.id == state.conversation_id else ""
                with ui.row().classes("conversation-item"):
                    ui.button(
                        conversation.title,
                        icon="chat_bubble_outline",
                        on_click=partial(select_conversation, conversation.id),
                    ).props("flat no-caps align=left").classes(
                        f"conversation-select{active_class}"
                    ).tooltip(
                        f"{conversation.title} - updated {_display_time(conversation.updated_at)}"
                    )
                    delete_button = ui.button(
                        icon="delete_outline",
                        on_click=partial(confirm_delete_conversation, conversation.id),
                    ).props("flat round dense").classes("conversation-delete")
                    delete_button.props["aria-label"] = f"Delete {conversation.title}"
                    delete_button.tooltip("Delete conversation")

    def render_header() -> None:
        conversation = current_conversation()
        title_label.set_text(conversation.title)
        model_select.set_options(
            settings.profile_options,
            value=conversation.model_profile,
        )

    def render_messages() -> None:
        messages_container.clear()
        messages = repository.list_messages(state.conversation_id)
        with messages_container:
            if not messages:
                with ui.column().classes("empty-state"):
                    with ui.element("div").classes("empty-icon"):
                        ui.icon("forum", size="md")
                    ui.label("Start a conversation").classes("text-xl font-bold")
                    ui.label(
                        "Messages are saved locally in SQLite, so you can leave and continue later."
                    ).classes("text-sm text-slate-500 leading-relaxed")
                    ui.label(
                        "Use the offline test model or an OpenAI-compatible vLLM profile"
                    ).classes("text-xs text-slate-400")
            else:
                for message in messages:
                    sent = message.role == "user"
                    name = "You" if sent else _profile_label(message.model_profile)
                    ui.chat_message(
                        text=message.content,
                        name=name,
                        stamp=_display_time(message.created_at),
                        sent=sent,
                    ).classes("chat-message")
        message_scroll.scroll_to(percent=1)

    def render_all() -> None:
        render_conversation_list()
        render_header()
        render_messages()

    def select_conversation(conversation_id: str) -> None:
        if state.generating:
            ui.notify("Stop the current response before switching chats.", type="warning")
            return
        if repository.get_conversation(conversation_id) is None:
            render_all()
            return
        state.conversation_id = conversation_id
        render_all()

    def new_conversation() -> None:
        if state.generating:
            ui.notify("Stop the current response before starting a new chat.", type="warning")
            return
        conversation = current_conversation()
        if (
            conversation.title == DEFAULT_CONVERSATION_TITLE
            and not repository.list_messages(conversation.id)
        ):
            composer.run_method("focus")
            return
        created = repository.create_conversation(settings.default_profile_key)
        state.conversation_id = created.id
        render_all()

    async def confirm_delete_conversation(conversation_id: str) -> None:
        if conversation_id in active_generations:
            ui.notify("Stop the response in this chat before deleting it.", type="warning")
            return
        conversation = repository.get_conversation(conversation_id)
        if conversation is None:
            render_all()
            return

        with ui.dialog() as dialog, ui.card().classes("w-96 max-w-full p-6 gap-5"):
            ui.label("Delete conversation?").classes("text-xl font-bold")
            ui.label(
                f'"{conversation.title}" and all of its messages will be removed.'
            ).classes("text-sm text-slate-500")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=lambda: dialog.submit(False)).props(
                    "flat no-caps"
                )
                ui.button("Delete", on_click=lambda: dialog.submit(True)).props(
                    "unelevated no-caps color=negative"
                )

        if not await dialog:
            return
        if conversation_id in active_generations:
            ui.notify("This chat started generating and cannot be deleted yet.", type="warning")
            return
        repository.delete_conversation(conversation_id)
        if conversation_id == state.conversation_id:
            remaining = repository.list_conversations()
            replacement = (
                remaining[0]
                if remaining
                else repository.create_conversation(settings.default_profile_key)
            )
            state.conversation_id = replacement.id
        render_all()

    def change_model(event) -> None:
        if state.conversation_id in active_generations:
            ui.notify("Stop this chat's response before changing its model.", type="warning")
            render_header()
            return
        profile_key = str(event.value)
        try:
            settings.profile(profile_key)
        except KeyError:
            render_header()
            return
        repository.set_model_profile(state.conversation_id, profile_key)
        render_conversation_list()

    def stop_generation() -> None:
        if state.stop_event is not None:
            state.stop_event.set()
            stop_button.disable()
            status_label.set_text("Stopping")
            if state.generation_task is not None and not state.generation_task.done():
                state.generation_task.cancel()

    async def send_message() -> None:
        text = str(composer.value or "").strip()
        if not text:
            ui.notify("Write a message first.", type="warning")
            return
        if state.generating:
            return

        conversation = current_conversation()
        if conversation.id in active_generations:
            ui.notify("This chat is already generating in another tab.", type="warning")
            return
        active_generations.add(conversation.id)

        try:
            previous_messages = repository.list_messages(conversation.id)
            repository.add_message(conversation.id, "user", text)
            if conversation.title == DEFAULT_CONVERSATION_TITLE and not previous_messages:
                repository.rename_conversation(conversation.id, _conversation_title(text))

            composer.value = ""
            render_all()
            set_busy(True)
            stop_event = asyncio.Event()
            state.stop_event = stop_event
            state.generation_task = asyncio.current_task()
            profile = settings.profile(conversation.model_profile)

            with messages_container:
                with ui.chat_message(name=profile.label, sent=False).classes("chat-message"):
                    assistant_label = ui.label(f"Generating with {profile.label}...").classes(
                        "streaming-content"
                    )
            message_scroll.scroll_to(percent=1)
        except Exception:
            active_generations.discard(conversation.id)
            raise

        chunks: list[str] = []
        stopped = False
        last_paint = monotonic()
        try:
            persisted_messages = repository.list_messages(conversation.id)
            async with aclosing(
                gateway.stream_reply(profile, _completion_messages(persisted_messages))
            ) as stream:
                async for chunk in stream:
                    if stop_event.is_set():
                        stopped = True
                        break
                    chunks.append(chunk)
                    now = monotonic()
                    if now - last_paint >= 0.04:
                        assistant_label.set_text("".join(chunks))
                        message_scroll.scroll_to(percent=1)
                        last_paint = now

            if not "".join(chunks).strip() and not stopped:
                raise LLMError(f"{profile.label} returned an empty response.")
        except asyncio.CancelledError:
            if not stop_event.is_set():
                raise
            stopped = True
        except LLMError as error:
            ui.notify(str(error), type="negative", multi_line=True, timeout=8000)
        finally:
            content = "".join(chunks).strip()
            try:
                if content:
                    repository.add_message(
                        conversation.id,
                        "assistant",
                        content,
                        model_profile=profile.key,
                    )
                if stopped:
                    ui.notify(
                        "Generation stopped; the partial response was saved."
                        if content
                        else "Stopped.",
                        type="info",
                    )
            finally:
                state.stop_event = None
                state.generation_task = None
                active_generations.discard(conversation.id)
                set_busy(False)
                if repository.get_conversation(state.conversation_id) is not None:
                    render_all()

    with ui.element("main").classes("app-shell"):
        with ui.column().classes("sidebar"):
            with ui.row().classes("w-full items-center gap-3 px-1"):
                with ui.element("div").classes("brand-mark"):
                    ui.icon("auto_awesome", size="sm")
                with ui.column().classes("gap-0"):
                    ui.label("Agent Lab").classes("brand-name")
                    ui.label("General project").classes("eyebrow")

            ui.button(
                "New conversation",
                icon="add",
                on_click=new_conversation,
            ).props("unelevated no-caps").classes("new-chat-button")

            ui.label("Recent chats").classes("eyebrow px-2 pt-1")
            conversation_list = ui.column().classes("conversation-list w-full")

            with ui.column().classes("sidebar-footer gap-1"):
                ui.label("Local history").classes("text-sm font-semibold")
                ui.label("SQLite persistence enabled").classes("text-xs text-slate-500")

        with ui.column().classes("main-panel"):
            with ui.row().classes("chat-header items-center justify-between"):
                with ui.column().classes("min-w-0 gap-0"):
                    ui.label("Conversation").classes("eyebrow")
                    title_label = ui.label().classes("chat-title")
                with ui.row().classes("items-center gap-4 no-wrap"):
                    model_select = ui.select(
                        settings.profile_options,
                        value=initial.model_profile,
                        label="Model",
                        on_change=change_model,
                    ).props("outlined dense options-dense").classes("model-select")
                    with ui.row().classes("items-center gap-3 no-wrap"):
                        ui.element("span").classes("status-dot")
                        status_label = ui.label("Ready").classes(
                            "status-copy text-sm text-slate-500"
                        )
                        status_label.props["role"] = "status"
                        status_label.props["aria-live"] = "polite"

            message_scroll = ui.scroll_area().classes("message-scroll")
            with message_scroll:
                messages_container = ui.column().classes("message-column")

            with ui.column().classes("composer-area gap-2"):
                with ui.row().classes("composer-row items-center"):
                    composer = ui.input(
                        placeholder="Message the model...",
                    ).props("borderless autocomplete=off").classes("composer-input")
                    composer.props["aria-label"] = "Message"
                    composer.on("keydown.enter", send_message)
                    stop_button = ui.button(
                        icon="stop",
                        on_click=stop_generation,
                    ).props("round unelevated").classes("stop-button")
                    stop_button.props["aria-label"] = "Stop generation"
                    stop_button.disable()
                    stop_button.tooltip("Stop generation")
                    send_button = ui.button(
                        icon="arrow_upward",
                        on_click=send_message,
                    ).props("round unelevated").classes("send-button")
                    send_button.props["aria-label"] = "Send message"
                    send_button.tooltip("Send message")
                ui.label(
                    "Enter to send - responses and conversation history are stored locally"
                ).classes("w-full text-center text-xs text-slate-400")

    render_all()


def main() -> None:
    """Start the NiceGUI development server."""
    ui.run(title="Agent Lab", favicon="✨", host="127.0.0.1", port=8080, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
