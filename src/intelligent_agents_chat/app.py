"""NiceGUI chat interface backed by SQLite and selectable model profiles."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
import logging
from pathlib import Path
from time import monotonic
from uuid import uuid4

from nicegui import app as nicegui_app, ui

from intelligent_agents_chat.chat import (
    TextChunk,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    active_generations,
    format_tool_call_entry,
    format_tool_result_entry,
    poll_profile_status,
    profile_status,
    stream_reply,
)
from intelligent_agents_chat.context import (
    ContextOverflowError,
    ContextPlan,
    document_service,
    document_store,
    memory_store,
    prepare_conversation_context,
    rebuild_conversation_memory,
    repository,
)
from intelligent_agents_chat.database import (
    AppSettings,
    ContextRunInput,
    ContextSourceInput,
    DEFAULT_CONVERSATION_TITLE,
    DEFAULT_PROJECT_ID,
    Conversation,
    Message,
    Project,
)
from intelligent_agents_chat.llm import (
    LLMError,
    MAX_TOKENS,
    THINKING_MAX_TOKENS,
    ContextLengthExceededError,
)
from intelligent_agents_chat.logging_config import log_event
from intelligent_agents_chat.models import DEFAULT_PROFILE_KEY, get_profile, profile_options


STATIC_DIR = Path(__file__).resolve().parent / "static"
DOCUMENT_UPLOAD_ACCEPT = ".txt,.md,.markdown,.pdf"
DOCUMENT_UPLOAD_LABEL = "TXT, Markdown, or PDF"

logger = logging.getLogger(__name__)

# If the model server rejects a request because the live tool-call chain
# outgrew what ContextAssembler budgeted for at turn start (see
# ContextLengthExceededError), compact and retry the same turn this many
# times before giving up and surfacing it as a normal error.
MAX_CONTEXT_OVERFLOW_RETRIES = 1

CHAT_MARKDOWN_EXTRAS = [
    "break-on-newline",
    "cuddled-lists",
    "fenced-code-blocks",
    "strike",
    "tables",
    "task_list",
]


def _log_unhandled_application_exception(error: Exception) -> None:
    logger.error(
        "application.unhandled_exception",
        exc_info=(type(error), error, error.__traceback__),
        extra={
            "event": "application.unhandled_exception",
            "error_type": type(error).__name__,
        },
    )


nicegui_app.on_exception(_log_unhandled_application_exception)
nicegui_app.on_startup(poll_profile_status)


ui.add_css(STATIC_DIR / "app.css", shared=True)


@dataclass(slots=True)
class PageState:
    """State that must remain local to one connected browser page."""

    project_id: str
    conversation_id: str
    page_id: str = field(default_factory=lambda: str(uuid4()))
    generating: bool = False
    stop_event: asyncio.Event | None = None
    generation_task: asyncio.Task[None] | None = None
    generation_id: str | None = None


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
        return get_profile(profile_key).label
    except KeyError:
        return profile_key


def _chat_markdown(content: str):
    """Render content that's genuinely Markdown-authored (or at least
    benefits more from Markdown's structure -- tables, code blocks -- than it
    loses from the occasional misread underscore): the assistant's final
    answer, and a tool call's result (see _add_trace_step's markdown=True).
    Sanitized since it's still untrusted content.
    """
    return ui.markdown(
        content,
        extras=CHAT_MARKDOWN_EXTRAS,
        sanitize=True,
    ).classes("chat-markdown")


def _chat_plain_text(content: str):
    """Render chat content that was never meant to be Markdown -- the
    user's own message, or model reasoning: neither is Markdown *authored
    for display*, so parsing it as Markdown only risks misreading an
    ordinary underscore as emphasis. No sanitization needed either: NiceGUI's
    `ui.label` always renders its argument as plain text, never HTML.
    """
    return ui.label(content).classes("chat-plain-text")


@dataclass(slots=True)
class _TraceStep:
    """One row in a turn's trace timeline (see .trace-timeline in app.css):
    reasoning, a tool call+result, or a plain note, always in the order they
    actually happened. `title_label`/`preview`/`dialog_body` are kept around
    so a live-streaming step (reasoning still arriving, or a tool call still
    awaiting its result) can be updated in place via `update()` --
    render_assistant_turn (replay) just builds one with its final text and
    never calls update(). A tool step's `prefix` (its call header) is never
    part of this state: it's fixed the moment the call happens and rendered
    once, in _add_trace_step, without ever needing to change again.
    """

    title_label: object
    preview: object
    dialog_body: object
    markdown: bool

    def update(self, *, title: str | None = None, body: str | None = None) -> None:
        if title is not None:
            self.title_label.set_text(title)
        if body is not None:
            if self.markdown:
                self.preview.set_content(body)
                self.dialog_body.set_content(body)
            else:
                self.preview.set_text(body)
                self.dialog_body.set_text(body)


def _add_trace_step(
    timeline,
    *,
    dot_class: str,
    title: str,
    body: str,
    markdown: bool = False,
    prefix: str | None = None,
) -> _TraceStep:
    """A step with a fixed-height preview (see .trace-step-preview) that
    opens a popup with the full text on click. `markdown=True` for a tool
    call's result (real Markdown -- tables, code, ... -- worth rendering) or
    plain text (see _chat_plain_text) for reasoning, since that's model
    output, not Markdown the model authored for display. `prefix`, if given,
    is a fixed, always-plain-text header shown above `body` in both the
    preview and the dialog -- for a tool call's name+arguments (see
    format_tool_call_entry): unlike the result, this must never be parsed as
    Markdown (tool/argument names routinely contain underscores, which
    Markdown misreads as emphasis), and unlike `body` it never changes once
    the call has happened, so it isn't kept on the returned _TraceStep. Its
    place before `body` also means it's still visible in the clipped preview
    before a click reveals the rest. See _add_trace_note for the plain,
    non-expandable kind (compaction notices and the like).
    """
    render = _chat_markdown if markdown else _chat_plain_text
    with timeline, ui.row().classes("trace-step"):
        ui.element("span").classes(f"trace-step-dot {dot_class}")
        with ui.column().classes("gap-1 trace-step-body"):
            title_label = ui.label(title).classes("trace-step-title")
            with ui.dialog() as dialog, ui.card().classes("trace-step-dialog"):
                if prefix is not None:
                    _chat_plain_text(prefix)
                dialog_body = render(body)
            with ui.element("div").classes("trace-step-preview") as preview:
                preview.on("click", dialog.open)
                if prefix is not None:
                    _chat_plain_text(prefix)
                preview_content = render(body)
    return _TraceStep(title_label, preview_content, dialog_body, markdown)


def _add_trace_note(timeline, text: str) -> None:
    """A short, plain trace line -- no preview/popup, for things that are
    never worth expanding (e.g. a compaction notice), unlike _add_trace_step.
    """
    with timeline, ui.row().classes("trace-step"):
        ui.element("span").classes("trace-step-dot note")
        ui.label(text).classes("trace-step-note-text")


def _add_memory_trace_step(timeline, sources: list) -> None:
    """The trace's first step (see render_assistant_turn), listing every
    project-memory source included in this turn's context. Shown first
    since this context is fixed before generation even starts, unlike
    everything that happens after it (reasoning, tool calls, ...).

    Two-level, unlike _add_trace_step's single fixed-height preview + popup:
    each source's metadata (title, locator, rank, token estimate) is always
    visible once the "Steps" accordion is open, and only that one source's
    excerpt is hidden behind its own click (a nested ui.expansion) -- a list
    of independent items reads better expanded one at a time than one big
    popup with every excerpt already run together. Plain text throughout
    (see _chat_plain_text): titles/excerpts come from arbitrary chat
    history, not Markdown authored for display.

    Memory only -- never document passages. Those share the same
    MessageContextSource shape (source_kind="project_document", see
    database.py) since search_documents' retrieval reuses the same
    provenance machinery, but its results go straight into that tool's own
    output/trace step (see chat.py's format_tool_result_entry), not here:
    `sources` only ever comes from context_plan.included_sources (see
    send_message and context.py's prepare_conversation_context), which is
    populated solely by retrieve_project_memory -- documents are never
    fetched automatically into a turn's context, only on the model's own
    tool call.
    """
    title = f"Project memory · {len(sources)} source" + ("s" if len(sources) != 1 else "")
    with timeline, ui.row().classes("trace-step"):
        ui.element("span").classes("trace-step-dot memory")
        with ui.column().classes("gap-1 trace-step-body"):
            ui.label(title).classes("trace-step-title")
            with ui.column().classes("gap-1 memory-source-list"):
                for source in sources:
                    header = (
                        f"{source.source_title} — {source.source_locator} "
                        f"(rank {source.rank}, {source.token_estimate:,} tokens)"
                    )
                    if source.source_excerpt:
                        with ui.expansion(header).props("dense").classes(
                            "memory-source-expansion"
                        ):
                            _chat_plain_text(source.source_excerpt)
                    else:
                        ui.label(header).classes("memory-source-header")


@ui.page("/")
def index() -> None:
    """Render a client-local chat workspace backed by shared SQLite persistence."""
    projects = repository.list_projects()
    initial_project = next(
        (project for project in projects if project.id == DEFAULT_PROJECT_ID),
        projects[0],
    )
    conversations = repository.list_conversations(initial_project.id)
    initial = (
        conversations[0]
        if conversations
        else repository.create_conversation(project_id=initial_project.id)
    )
    state = PageState(project_id=initial_project.id, conversation_id=initial.id)

    def page_event(level: int, event: str, **fields: object) -> None:
        context: dict[str, object] = {
            "page_id": state.page_id,
            "project_id": state.project_id,
            "conversation_id": state.conversation_id,
        }
        if state.generation_id is not None:
            context["generation_id"] = state.generation_id
        context.update(fields)
        log_event(logger, level, event, **context)

    page_event(
        logging.INFO,
        "ui.page.opened",
        available_project_count=len(projects),
    )

    def current_settings() -> AppSettings:
        """The app-wide generation preferences (model, thinking, memory) --
        shared by every conversation rather than stored on one, so switching
        chats or starting a new one never resets them. Only change_model/
        change_thinking/change_memory below ever write this; every other
        read goes through here, normalized against what's actually available
        right now (profiles/capabilities can change between releases, same
        reason current_conversation() used to normalize per-conversation).
        """
        settings = repository.get_app_settings()
        model_profile = settings.model_profile if settings is not None else DEFAULT_PROFILE_KEY
        if model_profile not in profile_options():
            model_profile = DEFAULT_PROFILE_KEY
        profile = get_profile(model_profile)
        thinking_enabled = bool(
            settings is not None and settings.thinking_enabled and profile.supports_thinking
        )
        memory_enabled = bool(settings is not None and settings.memory_enabled)
        return AppSettings(
            model_profile=model_profile,
            thinking_enabled=thinking_enabled,
            memory_enabled=memory_enabled,
            updated_at=settings.updated_at if settings is not None else datetime.now(timezone.utc),
        )

    def current_project() -> Project:
        project = repository.get_project(state.project_id)
        if project is not None:
            return project

        available_projects = repository.list_projects()
        if not available_projects:
            page_event(logging.ERROR, "ui.project.recovery_failed")
            raise RuntimeError("No project is available")
        project = available_projects[0]
        page_event(
            logging.WARNING,
            "ui.project.recovered",
            missing_project_id=state.project_id,
            replacement_project_id=project.id,
        )
        state.project_id = project.id
        return project

    def current_conversation() -> Conversation:
        conversation = repository.get_conversation(state.conversation_id)
        if conversation is None or conversation.project_id != state.project_id:
            missing_conversation_id = state.conversation_id
            conversations = repository.list_conversations(state.project_id)
            conversation = (
                conversations[0]
                if conversations
                else repository.create_conversation(project_id=state.project_id)
            )
            state.conversation_id = conversation.id
            page_event(
                logging.WARNING,
                "ui.conversation.recovered",
                missing_conversation_id=missing_conversation_id,
                replacement_conversation_id=conversation.id,
            )
        return conversation

    def set_busy(is_busy: bool) -> None:
        state.generating = is_busy
        page_event(logging.DEBUG, "ui.busy_state.changed", is_busy=is_busy)
        if is_busy:
            composer.disable()
            send_button.disable()
            stop_button.enable()
            model_select.disable()
            thinking_toggle.disable()
            memory_toggle.disable()
        else:
            composer.enable()
            send_button.enable()
            stop_button.disable()
            model_select.enable()
            profile = get_profile(current_settings().model_profile)
            if profile.supports_thinking:
                thinking_toggle.enable()
            else:
                thinking_toggle.disable()
            memory_toggle.enable()
            composer.run_method("focus")

    def render_project_picker() -> None:
        project = current_project()
        project_select.set_options(
            {item.id: item.name for item in repository.list_projects()},
            value=project.id,
        )

    def render_conversation_list() -> None:
        conversation_list.clear()
        with conversation_list:
            for conversation in repository.list_conversations(state.project_id):
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
                    delete_button = (
                        ui.button(
                            icon="delete_outline",
                            on_click=partial(confirm_delete_conversation, conversation.id),
                        )
                        .props("flat round dense")
                        .classes("conversation-delete")
                    )
                    delete_button.props["aria-label"] = f"Delete {conversation.title}"
                    delete_button.tooltip("Delete conversation")

    def render_model_status(profile_key: str) -> None:
        available = profile_status.get(profile_key)
        if available is None:
            model_status_dot.classes(remove="online offline")
            model_status_label.set_text("Checking availability...")
        elif available:
            model_status_dot.classes(add="online", remove="offline")
            model_status_label.set_text("Model reachable")
        else:
            model_status_dot.classes(add="offline", remove="online")
            model_status_label.set_text("Model unreachable")

    def render_header() -> None:
        conversation = current_conversation()
        settings = current_settings()
        profile = get_profile(settings.model_profile)
        title_label.set_text(conversation.title)
        model_select.set_options(
            profile_options(),
            value=settings.model_profile,
        )
        render_model_status(settings.model_profile)
        thinking_toggle.set_value(settings.thinking_enabled)
        memory_toggle.set_value(settings.memory_enabled)
        if profile.supports_thinking and not state.generating:
            thinking_toggle.enable()
        else:
            thinking_toggle.disable()
        if state.generating:
            memory_toggle.disable()
        else:
            memory_toggle.enable()

    def render_assistant_turn(turn_messages: list[Message]) -> None:
        """Render one user turn's response as a single chat bubble: a
        collapsible trace (retrieved project memory first, since it's fixed
        before generation even starts -- see _add_memory_trace_step for why
        never document passages -- then reasoning, tool calls with their
        results, and notes, in the order they actually happened) followed by
        the final answer, if any. The reasoning/tool/note part of the trace
        is the same shape `send_message` builds live while streaming; memory
        sources only appear here, once persistence has run, since the live
        bubble is replaced by this very function (via render_all()) the
        moment generation finishes. Both trace and answer share one bubble
        because QChatMessage renders one bubble *per direct slot child*, so
        they must be wrapped together.
        """
        # Each step is {"kind": "reasoning" | "tool" | "note", "body": str}
        # for reasoning/note, or {"kind": "tool", "call": dict, "result":
        # str | None} for a tool step -- built as a plain dict (not a
        # dataclass) specifically so a later "tool" message can mutate its
        # "result" in place via `pending_by_call_id`, keeping call and
        # result as one step no matter how many other calls the same round
        # made in between (previously: call, call, call, result, result,
        # result -- see completion_messages' docstring in context.py for the
        # model-facing version of the same fix).
        steps: list[dict] = []
        pending_by_call_id: dict[str, dict] = {}
        final_content: str | None = None
        model_profile: str | None = None
        context_message_id: int | None = None
        for message in turn_messages:
            if message.role == "assistant":
                model_profile = message.model_profile
                context_message_id = message.id
                if message.reasoning:
                    steps.append({"kind": "reasoning", "body": message.reasoning})
                if message.tool_calls:
                    for call in message.tool_calls:
                        step = {"kind": "tool", "call": call, "result": None}
                        steps.append(step)
                        pending_by_call_id[call["id"]] = step
                    if message.content:
                        steps.append({"kind": "note", "body": message.content})
                elif message.content:
                    final_content = message.content
            elif message.role == "tool":
                step = pending_by_call_id.pop(message.tool_call_id or "", None)
                if step is not None:
                    step["result"] = message.content

        # Retrieved before rendering since it decides whether the accordion
        # is worth showing at all even when there were no reasoning/tool
        # steps -- e.g. a plain answer that still drew on project memory.
        # context_message_id is the turn's last assistant-role message -- the
        # same one memory sources/the context run get persisted against (see
        # send_message's finally block), since that's always the final
        # flush_pending() call's message once a turn completes normally.
        memory_sources = (
            repository.list_message_context_sources(context_message_id)
            if context_message_id is not None
            else []
        )
        if not steps and not memory_sources and not final_content:
            return
        with ui.chat_message(
            name=_profile_label(model_profile),
            stamp=_display_time(turn_messages[-1].created_at),
            sent=False,
        ).classes("chat-message"):
            with ui.column().classes("gap-0 w-full"):
                if steps or memory_sources:
                    with (
                        ui.expansion("Steps", value=False).props("dense").classes("tool-trace")
                    ):
                        timeline = ui.column().classes("trace-timeline")
                        if memory_sources:
                            # Always first: this context was fixed before
                            # generation even started, ahead of everything
                            # that happened during it.
                            _add_memory_trace_step(timeline, memory_sources)
                        for step in steps:
                            if step["kind"] == "reasoning":
                                _add_trace_step(
                                    timeline, dot_class="reasoning", title="Thinking",
                                    body=step["body"],
                                )
                            elif step["kind"] == "tool":
                                _add_trace_step(
                                    timeline, dot_class="tool", title=step["call"]["name"],
                                    body=format_tool_result_entry(step["result"]),
                                    markdown=True,
                                    prefix=format_tool_call_entry(step["call"]),
                                )
                            else:
                                _add_trace_note(timeline, step["body"])
                if final_content:
                    _chat_markdown(final_content)
                if context_message_id is not None:
                    context_run = repository.get_message_context_run(context_message_id)
                    if context_run is not None:
                        # Prefer the real peak (prompt+completion of the turn's
                        # last round -- the most the model ever held in context
                        # at once, see chat.py's UsageEvent) for the headline
                        # number; fall back to the chars/3 estimate if the
                        # server never reported usage.
                        if context_run.real_peak_total_tokens is not None:
                            headline = (
                                f"Context: {context_run.real_peak_total_tokens:,} / "
                                f"{context_run.context_window_tokens:,} tokens used at peak "
                                "this turn"
                            )
                        else:
                            headline = (
                                f"Context: ~{context_run.estimated_input_tokens:,} / "
                                f"{context_run.context_window_tokens:,} tokens (estimated -- "
                                "the model server didn't report real usage)"
                            )
                        tooltip_lines = [
                            headline,
                            # f"Estimated: {context_run.estimated_input_tokens:,} input tokens "
                            # f"of {context_run.input_budget_tokens:,} budget "
                            # f"({context_run.context_window_tokens:,} model window)",
                        ]
                        if context_run.real_prompt_tokens is not None:
                            tooltip_lines.append(
                                f"Input Token (initial request, without reasoning & tools): {context_run.real_prompt_tokens:,}"
                            )
                            tooltip_lines.append(
                                f"Output Tokens (total generated in this turn, without tools): {context_run.real_completion_tokens:,}"
                            )
                        ui.icon("data_usage", size="xs").classes("context-usage-icon").tooltip(
                            "\n".join(tooltip_lines)
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
                        f'Messages in "{current_project().name}" are kept separate and saved '
                        "locally in SQLite."
                    ).classes("text-sm text-slate-500 leading-relaxed")
            else:
                index = 0
                while index < len(messages):
                    message = messages[index]
                    if message.role == "user":
                        with ui.chat_message(
                            name="You",
                            stamp=_display_time(message.created_at),
                            sent=True,
                        ).classes("chat-message"):
                            _chat_plain_text(message.content)
                        index += 1
                        continue
                    turn_start = index
                    while index < len(messages) and messages[index].role in ("assistant", "tool"):
                        index += 1
                    render_assistant_turn(messages[turn_start:index])
        message_scroll.scroll_to(percent=1)

    def render_all() -> None:
        render_project_picker()
        render_conversation_list()
        render_header()
        render_messages()

    def change_project(event) -> None:
        project_id = str(event.value)
        if project_id == state.project_id:
            page_event(logging.DEBUG, "ui.project.switch_ignored", reason="already_selected")
            return
        if state.generating:
            page_event(
                logging.WARNING,
                "ui.project.switch_blocked",
                target_project_id=project_id,
                reason="generation_active",
            )
            ui.notify("Stop the current response before switching projects.", type="warning")
            render_project_picker()
            return
        if repository.get_project(project_id) is None:
            page_event(
                logging.WARNING,
                "ui.project.switch_blocked",
                target_project_id=project_id,
                reason="project_missing",
            )
            render_project_picker()
            return

        previous_project_id = state.project_id
        previous_conversation_id = state.conversation_id
        state.project_id = project_id
        conversations = repository.list_conversations(project_id)
        conversation = (
            conversations[0]
            if conversations
            else repository.create_conversation(project_id=project_id)
        )
        state.conversation_id = conversation.id
        page_event(
            logging.INFO,
            "ui.project.switched",
            previous_project_id=previous_project_id,
            previous_conversation_id=previous_conversation_id,
        )
        render_all()

    async def create_project() -> None:
        if state.generating:
            page_event(
                logging.WARNING,
                "ui.project.create_blocked",
                reason="generation_active",
            )
            ui.notify("Stop the current response before creating a project.", type="warning")
            return

        page_event(logging.DEBUG, "ui.project.create_dialog_opened")

        with ui.dialog() as dialog, ui.card().classes("w-96 max-w-full p-6 gap-5"):
            ui.label("Create project").classes("text-xl font-bold")
            ui.label("Conversations in different projects stay separate.").classes(
                "text-sm text-slate-500"
            )
            project_name_input = (
                ui.input(
                    label="Project name",
                    placeholder="e.g. Research",
                )
                .props("outlined autofocus")
                .classes("w-full")
            )
            project_name_input.on(
                "keydown.enter",
                lambda: dialog.submit(str(project_name_input.value or "")),
            )
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat no-caps")
                ui.button(
                    "Create",
                    on_click=lambda: dialog.submit(str(project_name_input.value or "")),
                ).props("unelevated no-caps")

        project_name = await dialog
        if project_name is None or project_name is False:
            page_event(logging.DEBUG, "ui.project.create_cancelled")
            return
        try:
            project = repository.create_project(str(project_name))
        except ValueError as error:
            page_event(
                logging.WARNING,
                "ui.project.create_rejected",
                reason=str(error),
                project_name_chars=len(str(project_name)),
            )
            ui.notify(str(error), type="warning")
            return

        conversation = repository.create_conversation(project_id=project.id)
        state.project_id = project.id
        state.conversation_id = conversation.id
        page_event(
            logging.INFO,
            "ui.project.created_and_selected",
            project_name_chars=len(project.name),
        )
        render_all()
        composer.run_method("focus")

    def select_conversation(conversation_id: str) -> None:
        if state.generating:
            page_event(
                logging.WARNING,
                "ui.conversation.switch_blocked",
                target_conversation_id=conversation_id,
                reason="generation_active",
            )
            ui.notify("Stop the current response before switching chats.", type="warning")
            return
        conversation = repository.get_conversation(conversation_id)
        if conversation is None or conversation.project_id != state.project_id:
            page_event(
                logging.WARNING,
                "ui.conversation.switch_blocked",
                target_conversation_id=conversation_id,
                reason=("conversation_missing" if conversation is None else "different_project"),
            )
            render_all()
            return
        previous_conversation_id = state.conversation_id
        state.conversation_id = conversation_id
        page_event(
            logging.INFO,
            "ui.conversation.switched",
            previous_conversation_id=previous_conversation_id,
        )
        render_all()

    def new_conversation() -> None:
        if state.generating:
            page_event(
                logging.WARNING,
                "ui.conversation.create_blocked",
                reason="generation_active",
            )
            ui.notify("Stop the current response before starting a new chat.", type="warning")
            return
        conversation = current_conversation()
        if conversation.title == DEFAULT_CONVERSATION_TITLE and not repository.list_messages(
            conversation.id
        ):
            page_event(logging.DEBUG, "ui.conversation.empty_reused")
            composer.run_method("focus")
            return
        created = repository.create_conversation(project_id=state.project_id)
        previous_conversation_id = state.conversation_id
        state.conversation_id = created.id
        page_event(
            logging.INFO,
            "ui.conversation.created_and_selected",
            previous_conversation_id=previous_conversation_id,
        )
        render_all()

    async def confirm_delete_conversation(conversation_id: str) -> None:
        if conversation_id in active_generations:
            page_event(
                logging.WARNING,
                "ui.conversation.delete_blocked",
                target_conversation_id=conversation_id,
                reason="generation_active",
            )
            ui.notify("Stop the response in this chat before deleting it.", type="warning")
            return
        conversation = repository.get_conversation(conversation_id)
        if conversation is None:
            page_event(
                logging.WARNING,
                "ui.conversation.delete_blocked",
                target_conversation_id=conversation_id,
                reason="conversation_missing",
            )
            render_all()
            return

        page_event(
            logging.DEBUG,
            "ui.conversation.delete_dialog_opened",
            target_conversation_id=conversation_id,
            title_chars=len(conversation.title),
        )

        with ui.dialog() as dialog, ui.card().classes("w-96 max-w-full p-6 gap-5"):
            ui.label("Delete conversation?").classes("text-xl font-bold")
            ui.label(f'"{conversation.title}" and all of its messages will be removed.').classes(
                "text-sm text-slate-500"
            )
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat no-caps")
                ui.button("Delete", on_click=lambda: dialog.submit(True)).props(
                    "unelevated no-caps color=negative"
                )

        if not await dialog:
            page_event(
                logging.DEBUG,
                "ui.conversation.delete_cancelled",
                target_conversation_id=conversation_id,
            )
            return
        if conversation_id in active_generations:
            page_event(
                logging.WARNING,
                "ui.conversation.delete_blocked",
                target_conversation_id=conversation_id,
                reason="generation_started_while_confirming",
            )
            ui.notify("This chat started generating and cannot be deleted yet.", type="warning")
            return
        deleted = repository.delete_conversation(conversation_id)
        was_selected = conversation_id == state.conversation_id
        replacement_conversation_id = None
        if was_selected:
            remaining = repository.list_conversations(state.project_id)
            replacement = (
                remaining[0]
                if remaining
                else repository.create_conversation(project_id=state.project_id)
            )
            state.conversation_id = replacement.id
            replacement_conversation_id = replacement.id
        page_event(
            logging.INFO,
            "ui.conversation.deleted",
            target_conversation_id=conversation_id,
            deleted=deleted,
            was_selected=was_selected,
            replacement_conversation_id=replacement_conversation_id,
        )
        render_all()

    def change_model(event) -> None:
        # These settings are global (see current_settings), not tied to any
        # one conversation -- so the guard is "is anything generating
        # anywhere", not "is the chat I'm looking at generating".
        if active_generations:
            page_event(
                logging.WARNING,
                "ui.model.change_blocked",
                target_model_profile=str(event.value),
                reason="generation_active",
            )
            ui.notify("Stop the current response before changing the model.", type="warning")
            render_header()
            return
        profile_key = str(event.value)
        try:
            profile = get_profile(profile_key)
        except KeyError:
            page_event(
                logging.WARNING,
                "ui.model.change_blocked",
                target_model_profile=profile_key,
                reason="unknown_profile",
            )
            render_header()
            return
        settings = current_settings()
        thinking_disabled = settings.thinking_enabled and not profile.supports_thinking
        repository.set_app_settings(
            model_profile=profile_key,
            thinking_enabled=False if thinking_disabled else settings.thinking_enabled,
            memory_enabled=settings.memory_enabled,
        )
        page_event(
            logging.INFO,
            "ui.model.changed",
            previous_model_profile=settings.model_profile,
            model_profile=profile_key,
            thinking_disabled=thinking_disabled,
        )
        render_conversation_list()
        render_header()

    def change_thinking(event) -> None:
        requested = bool(event.value)
        settings = current_settings()
        profile = get_profile(settings.model_profile)
        if active_generations:
            page_event(
                logging.WARNING,
                "ui.thinking.change_blocked",
                requested_thinking_enabled=requested,
                reason="generation_active",
            )
            ui.notify("Stop the current response before changing Thinking.", type="warning")
            render_header()
            return
        if requested == settings.thinking_enabled:
            page_event(
                logging.DEBUG,
                "ui.thinking.change_ignored",
                thinking_enabled=requested,
                reason="already_selected",
            )
            return
        if requested and not profile.supports_thinking:
            page_event(
                logging.WARNING,
                "ui.thinking.change_blocked",
                requested_thinking_enabled=requested,
                model_profile=profile.key,
                reason="profile_unsupported",
            )
            ui.notify("The selected model profile does not support Thinking.", type="warning")
            render_header()
            return

        repository.set_app_settings(
            model_profile=settings.model_profile,
            thinking_enabled=requested,
            memory_enabled=settings.memory_enabled,
        )
        page_event(
            logging.INFO,
            "ui.thinking.changed",
            model_profile=profile.key,
            thinking_enabled=requested,
            max_tokens=(THINKING_MAX_TOKENS if requested else MAX_TOKENS),
        )
        render_conversation_list()

    def change_memory(event) -> None:
        requested = bool(event.value)
        settings = current_settings()
        if active_generations:
            page_event(
                logging.WARNING,
                "ui.memory.change_blocked",
                requested_memory_enabled=requested,
                reason="generation_active",
            )
            ui.notify(
                "Stop the current response before changing project memory.", type="warning"
            )
            render_header()
            return
        if requested == settings.memory_enabled:
            page_event(
                logging.DEBUG,
                "ui.memory.change_ignored",
                memory_enabled=requested,
                reason="already_selected",
            )
            return

        repository.set_app_settings(
            model_profile=settings.model_profile,
            thinking_enabled=settings.thinking_enabled,
            memory_enabled=requested,
        )
        page_event(
            logging.INFO,
            "ui.memory.changed",
            memory_enabled=requested,
        )
        render_conversation_list()

    def manage_project_documents() -> None:
        project = current_project()
        page_event(
            logging.INFO,
            "ui.documents.management_opened",
            embedding_model=document_service.embedding_model,
        )

        with ui.dialog() as dialog, ui.card().classes("document-dialog"):
            with ui.row().classes("w-full items-start justify-between no-wrap"):
                with ui.column().classes("gap-1 min-w-0"):
                    ui.label(f"Project documents · {project.name}").classes("text-xl font-bold")
                    ui.label(
                        f"Upload {DOCUMENT_UPLOAD_LABEL} files. Indexed passages are available "
                        "only to chats in this project, via the search_documents tool."
                    ).classes("text-sm text-slate-500")
                document_dialog_close = ui.button(icon="close", on_click=dialog.close).props(
                    "flat round dense"
                )
                document_dialog_close.props["aria-label"] = "Close project documents"

            with ui.row().classes("document-ingestion-status"):
                ui.icon("hub", size="sm")
                ui.label(f"Semantic search · {document_service.embedding_model}").classes(
                    "text-xs text-slate-500"
                )

            document_count_label = ui.label().classes("text-xs text-slate-400")
            documents_container = ui.column().classes("document-entry-list")

            async def upload_document(event) -> None:
                file = event.file
                page_event(
                    logging.INFO,
                    "ui.documents.upload_started",
                    filename_chars=len(file.name),
                    media_type=file.content_type,
                    byte_size=file.size(),
                )
                try:
                    data = await file.read()
                    result = await document_service.upload(
                        project_id=project.id,
                        display_name=file.name,
                        media_type=file.content_type,
                        data=data,
                    )
                except Exception as error:
                    page_event(
                        logging.WARNING,
                        "ui.documents.upload_failed",
                        filename_chars=len(file.name),
                        media_type=file.content_type,
                        byte_size=file.size(),
                        error_type=type(error).__name__,
                    )
                    ui.notify(str(error), type="negative", multi_line=True, timeout=8000)
                else:
                    page_event(
                        logging.INFO,
                        "ui.documents.upload_completed",
                        document_id=result.document.id,
                        duplicate=result.duplicate,
                        chunk_count=result.document.chunk_count,
                        embedding_model=result.document.embedding_model,
                    )
                    ui.notify(
                        (
                            f'"{result.document.display_name}" was already indexed.'
                            if result.duplicate
                            else f'Indexed "{result.document.display_name}" '
                            f"({result.document.chunk_count} chunks)."
                        ),
                        type="info" if result.duplicate else "positive",
                    )
                render_documents()

            uploader = ui.upload(
                label="Add project documents",
                multiple=True,
                max_file_size=10 * 1024 * 1024,
                max_total_size=30 * 1024 * 1024,
                max_files=10,
                auto_upload=True,
                on_upload=upload_document,
                on_rejected=lambda: ui.notify(
                    f"Only up to 10 {DOCUMENT_UPLOAD_LABEL} files of at most 10 MiB are accepted.",
                    type="warning",
                ),
            ).props(f"accept={DOCUMENT_UPLOAD_ACCEPT} flat bordered")
            uploader.classes("document-uploader")

            async def reindex_document(document_id: str) -> None:
                page_event(
                    logging.INFO,
                    "ui.documents.reindex_started",
                    document_id=document_id,
                )
                try:
                    indexed = await document_service.reindex(document_id, project.id)
                except Exception as error:
                    page_event(
                        logging.WARNING,
                        "ui.documents.reindex_failed",
                        document_id=document_id,
                        error_type=type(error).__name__,
                    )
                    ui.notify(str(error), type="negative", multi_line=True, timeout=8000)
                else:
                    ui.notify(
                        f'Re-indexed "{indexed.display_name}" ({indexed.chunk_count} chunks).',
                        type="positive",
                    )
                render_documents()

            async def confirm_delete_document(document_id: str) -> None:
                document = document_store.get_document(document_id)
                if document is None or document.project_id != project.id:
                    ui.notify("This document no longer exists.", type="warning")
                    render_documents()
                    return
                with ui.dialog() as delete_dialog, ui.card().classes("w-96 max-w-full p-6 gap-5"):
                    ui.label("Delete project document?").classes("text-xl font-bold")
                    ui.label(
                        f'"{document.display_name}", its chunks, and embeddings will be removed.'
                    ).classes("text-sm text-slate-500")
                    with ui.row().classes("w-full justify-end gap-2"):
                        ui.button("Cancel", on_click=lambda: delete_dialog.submit(False)).props(
                            "flat no-caps"
                        )
                        ui.button("Delete", on_click=lambda: delete_dialog.submit(True)).props(
                            "unelevated no-caps color=negative"
                        )
                if not await delete_dialog:
                    return
                deleted = await document_service.delete(document.id, project.id)
                page_event(
                    logging.INFO,
                    "ui.documents.deleted",
                    document_id=document.id,
                    deleted=deleted,
                )
                render_documents()
                if deleted:
                    ui.notify(f'Deleted "{document.display_name}".', type="info")

            def render_documents() -> None:
                documents = document_store.list_documents(project.id)
                document_count_label.set_text(
                    f"{len(documents)} project document" + ("s" if len(documents) != 1 else "")
                )
                documents_container.clear()
                with documents_container:
                    if not documents:
                        ui.label(
                            "No documents yet. Upload a file to make it available for RAG."
                        ).classes("text-sm text-slate-500 py-6")
                    for document in documents:
                        status_icon = {
                            "indexed": "check_circle",
                            "processing": "hourglass_top",
                            "failed": "error",
                        }.get(document.status, "help")
                        with ui.expansion(
                            document.display_name,
                            caption=(
                                f"{document.status} · {document.chunk_count} chunks · "
                                f"{document.byte_size / 1024:.1f} KiB"
                            ),
                            icon=status_icon,
                        ).classes(f"document-entry status-{document.status}"):
                            with ui.column().classes("gap-2"):
                                ui.label(f"SHA-256 · {document.sha256[:16]}…").classes(
                                    "text-xs text-slate-400"
                                )
                                if document.embedding_model:
                                    ui.label(
                                        f"Embeddings · {document.embedding_model} · "
                                        f"{document.embedding_dimension} dimensions"
                                    ).classes("text-xs text-slate-400")
                                else:
                                    ui.label("Embeddings · not yet indexed").classes(
                                        "text-xs text-slate-400"
                                    )
                                if document.error_message:
                                    ui.label(document.error_message).classes(
                                        "document-error text-sm"
                                    )
                                with ui.row().classes("gap-2"):
                                    # Only for a document that actually failed:
                                    # re-running a successful one re-reads the
                                    # same file through the same deterministic
                                    # pipeline and the same pinned model, so it
                                    # can only ever produce what's already there.
                                    if document.status == "failed":
                                        ui.button(
                                            "Retry",
                                            icon="refresh",
                                            on_click=partial(reindex_document, document.id),
                                        ).props("flat dense no-caps")
                                    ui.button(
                                        "Delete",
                                        icon="delete_outline",
                                        on_click=partial(confirm_delete_document, document.id),
                                    ).props("flat dense no-caps color=negative")

            render_documents()

        dialog.open()

    def manage_project_memory() -> None:
        project = current_project()
        page_event(logging.INFO, "ui.memory.management_opened")

        with ui.dialog() as dialog, ui.card().classes("memory-dialog"):
            with ui.row().classes("w-full items-start justify-between no-wrap"):
                with ui.column().classes("gap-1 min-w-0"):
                    ui.label(f"Project memory · {project.name}").classes("text-xl font-bold")
                    ui.label(
                        "Review the chat turns available to other conversations in this project. "
                        "Disabled entries stay stored but are excluded from retrieval."
                    ).classes("text-sm text-slate-500")
                memory_dialog_close = ui.button(icon="close", on_click=dialog.close).props(
                    "flat round dense"
                )
                memory_dialog_close.props["aria-label"] = "Close project memory"

            memory_entries_container = ui.column().classes("memory-entry-list")

            def change_memory_entry(entry_id: int, event) -> None:
                enabled = bool(event.value)
                updated = memory_store.set_enabled(project.id, entry_id, enabled)
                page_event(
                    logging.INFO,
                    "ui.memory.entry_changed",
                    entry_id=entry_id,
                    enabled=enabled,
                    updated=updated,
                )
                if not updated:
                    ui.notify("This memory entry no longer exists.", type="warning")
                    render_memory_entries()

            def render_memory_entries() -> None:
                entries = memory_store.list_entries(project.id)
                memory_entries_container.clear()
                with memory_entries_container:
                    if not entries:
                        ui.label(
                            "No memory entries yet. They are derived automatically from chat turns."
                        ).classes("text-sm text-slate-500 py-6")
                    for entry in entries:
                        with ui.expansion(
                            entry.title,
                            caption=(
                                f"messages {entry.source_message_start_id}-"
                                f"{entry.source_message_end_id}"
                            ),
                            icon="chat_bubble_outline",
                        ).classes("memory-entry"):
                            ui.label(entry.content).classes("memory-entry-content")
                            ui.switch(
                                "Available for retrieval",
                                value=entry.enabled,
                                on_change=partial(change_memory_entry, entry.id),
                            ).props("dense color=deep-purple")

            def rebuild_project_memory() -> None:
                entry_count = memory_store.rebuild_project(project.id)
                page_event(
                    logging.INFO,
                    "ui.memory.rebuilt",
                    entry_count=entry_count,
                )
                render_memory_entries()
                ui.notify(f"Project memory rebuilt ({entry_count} entries).", type="positive")

            with ui.row().classes("w-full justify-between items-center"):
                ui.label("Memory is only retrieved inside this project.").classes(
                    "text-xs text-slate-400"
                )
                ui.button(
                    "Rebuild",
                    icon="refresh",
                    on_click=rebuild_project_memory,
                ).props("flat no-caps")

            render_memory_entries()

        dialog.open()

    def stop_generation() -> None:
        if state.stop_event is None:
            page_event(logging.DEBUG, "ui.generation.stop_ignored", reason="not_generating")
            return

        task_was_active = state.generation_task is not None and not state.generation_task.done()
        page_event(
            logging.INFO,
            "ui.generation.stop_requested",
            generation_task_active=task_was_active,
        )
        state.stop_event.set()
        stop_button.disable()
        if task_was_active:
            assert state.generation_task is not None
            state.generation_task.cancel()

    async def send_message() -> None:
        raw_text = str(composer.value or "")
        text = raw_text.strip()
        if not text:
            page_event(
                logging.DEBUG,
                "ui.generation.rejected",
                reason="empty_message",
                input_message_chars=len(raw_text),
            )
            ui.notify("Write a message first.", type="warning")
            return
        if state.generating:
            page_event(
                logging.WARNING,
                "ui.generation.rejected",
                reason="page_generation_active",
                input_message_chars=len(text),
            )
            return

        conversation = current_conversation()
        if conversation.id in active_generations:
            page_event(
                logging.WARNING,
                "ui.generation.rejected",
                reason="conversation_generation_active",
                input_message_chars=len(text),
            )
            ui.notify("This chat is already generating in another tab.", type="warning")
            return

        generation_id = str(uuid4())
        generation_started_at = monotonic()
        state.generation_id = generation_id
        active_generations.add(conversation.id)
        context_plan: ContextPlan | None = None
        usage_event: UsageEvent | None = None

        try:
            previous_messages = repository.list_messages(conversation.id)
            settings = current_settings()
            profile = get_profile(settings.model_profile)
            page_event(
                logging.INFO,
                "ui.generation.started",
                model_profile=profile.key,
                model_name=profile.model,
                thinking_enabled=settings.thinking_enabled,
                memory_enabled=settings.memory_enabled,
                max_tokens=(THINKING_MAX_TOKENS if settings.thinking_enabled else MAX_TOKENS),
                input_message_chars=len(text),
                previous_message_count=len(previous_messages),
            )
            repository.add_message(conversation.id, "user", text)
            if conversation.title == DEFAULT_CONVERSATION_TITLE and not previous_messages:
                repository.rename_conversation(conversation.id, _conversation_title(text))

            composer.value = ""
            render_all()
            set_busy(True)
            stop_event = asyncio.Event()
            state.stop_event = stop_event
            state.generation_task = asyncio.current_task()

            with messages_container:
                with ui.chat_message(name=profile.label, sent=False).classes("chat-message"):
                    # Both the trace accordion (created lazily on the first
                    # reasoning/tool event) and the answer share one wrapping
                    # column, so they end up in the same bubble -- QChatMessage
                    # renders one bubble *per direct slot child*, and we want
                    # only one for this whole turn.
                    with ui.column().classes("gap-0 w-full"):
                        trace_container = ui.column().classes("gap-0")
                        progress_text = f"Generating with {profile.label}..."
                        assistant_markdown = _chat_markdown(progress_text)
            message_scroll.scroll_to(percent=1)
        except Exception:
            logger.exception(
                "ui.generation.preparation_failed",
                extra={
                    "event": "ui.generation.preparation_failed",
                    "page_id": state.page_id,
                    "project_id": state.project_id,
                    "conversation_id": conversation.id,
                    "generation_id": generation_id,
                    "model_profile": current_settings().model_profile,
                    "input_message_chars": len(text),
                },
            )
            try:
                state.stop_event = None
                state.generation_task = None
                active_generations.discard(conversation.id)
                if state.generating:
                    set_busy(False)
            finally:
                state.generation_id = None
            raise

        # `pending_content`/`pending_reasoning` hold text since the last flush point
        # (round start, or the last tool call). `any_output`/`output_chars` drive the
        # "did we get anything at all" check and logging -- they see everything shown
        # live (reasoning, tool calls/results, final content), same as before.
        pending_content: list[str] = []
        pending_reasoning: list[str] = []
        any_output = False
        output_chars = 0
        event_count = 0
        messages_saved_count = 0
        stopped = False
        outcome = "streaming"
        last_paint = monotonic()
        trace_timeline = None
        current_reasoning_step: _TraceStep | None = None
        # tool_call_id -> its step, only while still waiting for a result --
        # popped once the matching ToolResultEvent arrives (see below), so
        # call and result always land in the same trace row no matter how
        # many other calls the same round made in between. Only the step is
        # kept (not the call dict): the call header is a fixed `prefix` set
        # once at creation (see _add_trace_step) and never needs recomputing;
        # only the result half (`body`) ever changes.
        pending_tool_steps: dict[str, _TraceStep] = {}
        # The last assistant message flush_pending actually persisted, of any
        # kind (tool-calling or final content) -- kept so the turn's context
        # sources/context run (see the finally block below) can still attach
        # to *some* message even if the turn's very last flush_pending() call
        # (right before persistence) has nothing new to flush (e.g. a tool
        # round produced no further reasoning/content of its own).
        context_message: Message | None = None

        def _ensure_trace_timeline():
            nonlocal trace_timeline
            if trace_timeline is None:
                with trace_container, ui.expansion("Steps", value=False).props(
                    "dense"
                ).classes("tool-trace"):
                    trace_timeline = ui.column().classes("trace-timeline")
            return trace_timeline

        def add_trace_step(
            *,
            dot_class: str,
            title: str,
            body: str,
            markdown: bool = False,
            prefix: str | None = None,
        ) -> _TraceStep:
            return _add_trace_step(
                _ensure_trace_timeline(),
                dot_class=dot_class,
                title=title,
                body=body,
                markdown=markdown,
                prefix=prefix,
            )

        def add_trace_note(text: str) -> None:
            _add_trace_note(_ensure_trace_timeline(), text)

        def flush_pending(tool_calls: tuple[dict, ...] | None = None) -> Message | None:
            nonlocal messages_saved_count, current_reasoning_step, context_message
            reasoning = "".join(pending_reasoning).strip() or None
            content = "".join(pending_content).strip()
            if current_reasoning_step is not None and reasoning:
                # Paint the final, complete text -- the last periodic repaint may
                # have landed slightly before the reasoning block actually closed.
                current_reasoning_step.update(body=reasoning)
            pending_reasoning.clear()
            pending_content.clear()
            current_reasoning_step = None
            if content or reasoning or tool_calls:
                context_message = repository.add_message(
                    conversation.id,
                    "assistant",
                    content,
                    model_profile=profile.key,
                    reasoning=reasoning,
                    tool_calls=tool_calls,
                )
                messages_saved_count += 1
                return context_message
            return None

        try:
            # A generation can be retried (bounded, see MAX_CONTEXT_OVERFLOW_RETRIES)
            # if the model server rejects it for having outgrown its context window
            # mid-turn -- ContextAssembler only ever budgets the turn's first
            # request, so a long tool-call chain can still exceed it later (see
            # ContextLengthExceededError). No new user message gets added between
            # attempts, so the turn is still "open": completion_messages() never
            # collapses it, and the retry's request naturally includes this turn's
            # own tool calls and results so far, in full -- nothing to re-thread by
            # hand. pending_content/pending_reasoning/the trace widgets etc. below
            # are already scoped outside this block, so a retry just keeps
            # appending to the same in-progress bubble, not starting a new one.
            force_compact = False
            for attempt in range(MAX_CONTEXT_OVERFLOW_RETRIES + 1):
                persisted_messages = repository.list_messages(conversation.id)
                context_plan = await prepare_conversation_context(
                    conversation,
                    profile,
                    persisted_messages,
                    text,
                    thinking_enabled=settings.thinking_enabled,
                    memory_enabled=settings.memory_enabled,
                    force_compact=force_compact,
                    request_id=generation_id,
                    on_compacting=lambda: add_trace_note(
                        "🗜️ Context limit reached -- compacting older history and retrying..."
                        if force_compact
                        else "🗜️ Compacting older conversation history..."
                    ),
                )
                request_messages = list(context_plan.messages)
                page_event(
                    logging.DEBUG,
                    "ui.generation.request_prepared",
                    model_profile=profile.key,
                    thinking_enabled=settings.thinking_enabled,
                    completion_message_count=len(request_messages),
                    completion_chars=sum(
                        len(message.get("content") or "") for message in request_messages
                    ),
                    memory_candidate_count=(
                        len(context_plan.included_sources) + len(context_plan.excluded_sources)
                    ),
                    memory_source_count=len(context_plan.included_sources),
                    memory_excluded_count=len(context_plan.excluded_sources),
                    estimated_input_tokens=context_plan.estimated_input_tokens,
                    input_budget_tokens=context_plan.input_budget_tokens,
                    omitted_history_messages=context_plan.omitted_history_messages,
                    retry_attempt=attempt,
                )
                try:
                    async with aclosing(
                        stream_reply(
                            profile,
                            request_messages,
                            request_id=generation_id,
                            thinking_enabled=settings.thinking_enabled,
                            project_id=conversation.project_id,
                            conversation_id=conversation.id,
                        )
                    ) as stream:
                        async for event in stream:
                            if stop_event.is_set():
                                stopped = True
                                break
                            any_output = True
                            if isinstance(event, TextChunk):
                                event_count += 1
                                output_chars += len(event.text)
                                if event.is_reasoning:
                                    pending_reasoning.append(event.text)
                                else:
                                    pending_content.append(event.text)
                            elif isinstance(event, ToolCallEvent):
                                event_count += len(event.tool_calls)
                                # Any content the model produced right before deciding to call a
                                # tool (rare, but possible) belongs in the trace, not the final
                                # answer bubble. it isn't the model's real answer yet.
                                leftover_content = "".join(pending_content).strip()
                                flush_pending(tool_calls=event.tool_calls)
                                if leftover_content:
                                    add_trace_note(leftover_content)
                                assistant_markdown.set_content(progress_text)
                                for call in event.tool_calls:
                                    output_chars += len(call["arguments"])
                                    pending_tool_steps[call["id"]] = add_trace_step(
                                        dot_class="tool",
                                        title=call["name"],
                                        body=format_tool_result_entry(None, still_running=True),
                                        markdown=True,
                                        prefix=format_tool_call_entry(call),
                                    )
                            elif isinstance(event, ToolResultEvent):
                                event_count += 1
                                repository.add_message(
                                    conversation.id,
                                    "tool",
                                    event.result,
                                    tool_call_id=event.tool_call_id,
                                )
                                messages_saved_count += 1
                                output_chars += len(event.result)
                                step = pending_tool_steps.pop(event.tool_call_id, None)
                                if step is not None:
                                    step.update(body=format_tool_result_entry(event.result))
                            elif isinstance(event, UsageEvent):
                                usage_event = event
                            now = monotonic()

                            # Actual UI updates (throttled to 40ms)
                            if now - last_paint >= 0.04:
                                if pending_reasoning:
                                    reasoning_text = "".join(pending_reasoning)
                                    if current_reasoning_step is None:
                                        current_reasoning_step = add_trace_step(
                                            dot_class="reasoning",
                                            title="Thinking",
                                            body=reasoning_text,
                                        )
                                    else:
                                        current_reasoning_step.update(body=reasoning_text)
                                if pending_content:
                                    assistant_markdown.set_content("".join(pending_content))
                                message_scroll.scroll_to(percent=1)
                                last_paint = now
                    break
                except ContextLengthExceededError:
                    if attempt >= MAX_CONTEXT_OVERFLOW_RETRIES:
                        raise
                    page_event(
                        logging.WARNING,
                        "ui.generation.context_overflow_retry",
                        model_profile=profile.key,
                        attempt=attempt,
                    )
                    force_compact = True

            if not any_output and not stopped:
                if settings.thinking_enabled:
                    raise LLMError(
                        f"{profile.label} produced no final answer. Its thinking token budget "
                        "may be exhausted; disable Thinking or increase "
                        "VLLM_THINKING_MAX_TOKENS."
                    )
                raise LLMError(f"{profile.label} returned an empty response.")
            outcome = "stopped" if stopped else "completed"
        except asyncio.CancelledError:
            if not stop_event.is_set():
                outcome = "unexpected_cancellation"
                page_event(
                    logging.ERROR,
                    "ui.generation.cancelled_unexpectedly",
                    model_profile=profile.key,
                )
                raise
            stopped = True
            outcome = "stopped"
        except (LLMError, ContextOverflowError) as error:
            outcome = (
                "context_budget_error" if isinstance(error, ContextOverflowError) else "model_error"
            )
            page_event(
                logging.WARNING,
                "ui.generation.model_error",
                model_profile=profile.key,
                thinking_enabled=settings.thinking_enabled,
                error_type=type(error).__name__,
            )
            ui.notify(str(error), type="negative", multi_line=True, timeout=8000)
        except Exception as error:
            outcome = "unexpected_error"
            logger.exception(
                "ui.generation.unexpected_error",
                extra={
                    "event": "ui.generation.unexpected_error",
                    "page_id": state.page_id,
                    "project_id": state.project_id,
                    "conversation_id": conversation.id,
                    "generation_id": generation_id,
                    "model_profile": profile.key,
                    "thinking_enabled": settings.thinking_enabled,
                    "error_type": type(error).__name__,
                },
            )
            raise
        finally:
            try:
                final_message = flush_pending() or context_message
                if final_message is not None and context_plan is not None:
                    if context_plan.included_sources:
                        repository.add_message_context_sources(
                            final_message.id,
                            [
                                ContextSourceInput(
                                    source_kind=source.candidate.source_kind,
                                    source_id=source.candidate.source_id,
                                    source_project_id=source.candidate.project_id,
                                    source_conversation_id=(
                                        source.candidate.source_conversation_id
                                    ),
                                    source_title=source.candidate.title,
                                    source_locator=source.candidate.locator,
                                    source_excerpt=source.candidate.text,
                                    rank=source.rank,
                                    score=source.candidate.score,
                                    token_estimate=source.token_estimate,
                                )
                                for source in context_plan.included_sources
                            ],
                        )
                    repository.add_message_context_run(
                        final_message.id,
                        ContextRunInput(
                            context_window_tokens=profile.context_window_tokens,
                            input_budget_tokens=context_plan.input_budget_tokens,
                            estimated_input_tokens=context_plan.estimated_input_tokens,
                            real_prompt_tokens=(
                                usage_event.prompt_tokens if usage_event is not None else None
                            ),
                            real_completion_tokens=(
                                usage_event.completion_tokens if usage_event is not None else None
                            ),
                            real_peak_total_tokens=(
                                usage_event.peak_total_tokens if usage_event is not None else None
                            ),
                        ),
                    )
                if messages_saved_count:
                    rebuild_conversation_memory(conversation.id)
                if stopped:
                    ui.notify(
                        "Generation stopped; the partial response was saved."
                        if messages_saved_count
                        else "Stopped.",
                        type="info",
                    )
            except Exception as error:
                outcome = "persistence_error"
                logger.exception(
                    "ui.generation.persistence_failed",
                    extra={
                        "event": "ui.generation.persistence_failed",
                        "page_id": state.page_id,
                        "project_id": state.project_id,
                        "conversation_id": conversation.id,
                        "generation_id": generation_id,
                        "model_profile": profile.key,
                        "thinking_enabled": settings.thinking_enabled,
                        "error_type": type(error).__name__,
                        "output_chars": output_chars,
                    },
                )
                raise
            finally:
                page_event(
                    logging.INFO,
                    "ui.generation.finished",
                    model_profile=profile.key,
                    thinking_enabled=settings.thinking_enabled,
                    outcome=outcome,
                    stopped=stopped,
                    stream_chunk_count=event_count,
                    output_chars=output_chars,
                    messages_saved_count=messages_saved_count,
                    duration_ms=round((monotonic() - generation_started_at) * 1000, 2),
                )
                state.stop_event = None
                state.generation_task = None
                active_generations.discard(conversation.id)
                try:
                    set_busy(False)
                    if repository.get_conversation(state.conversation_id) is not None:
                        render_all()
                finally:
                    state.generation_id = None

    with ui.element("main").classes("app-shell"):
        with ui.column().classes("sidebar"):
            with ui.row().classes("w-full items-center gap-3 px-1"):
                with ui.element("div").classes("brand-mark"):
                    ui.icon("auto_awesome", size="sm")
                with ui.column().classes("gap-0"):
                    ui.label("Agent Lab").classes("brand-name")
                    ui.label("Project workspace").classes("eyebrow")

            with ui.row().classes("project-picker"):
                project_select = (
                    ui.select(
                        {project.id: project.name for project in projects},
                        value=state.project_id,
                        label="Project",
                        on_change=change_project,
                    )
                    .props("outlined dense options-dense")
                    .classes("project-select")
                )
                project_create_button = (
                    ui.button(
                        icon="create_new_folder",
                        on_click=create_project,
                    )
                    .props("flat round dense")
                    .classes("project-create")
                )
                project_create_button.props["aria-label"] = "Create project"
                project_create_button.tooltip("Create project")

            ui.button(
                "New conversation",
                icon="add",
                on_click=new_conversation,
            ).props("unelevated no-caps").classes("new-chat-button")

            ui.label("Project chats").classes("eyebrow px-2 pt-1")
            conversation_list = ui.column().classes("conversation-list w-full")

            ui.button(
                "Manage project memory",
                icon="manage_search",
                on_click=manage_project_memory,
            ).props("flat no-caps align=left").classes("memory-manage-button")

            ui.button(
                "Manage project documents",
                icon="folder_open",
                on_click=manage_project_documents,
            ).props("flat no-caps align=left").classes("memory-manage-button")

        with ui.column().classes("main-panel"):
            initial_settings = current_settings()
            with ui.row().classes("chat-header items-center justify-between"):
                with ui.column().classes("min-w-0 gap-0"):
                    ui.label("Conversation").classes("eyebrow")
                    title_label = ui.label().classes("chat-title")
                with ui.row().classes("items-center gap-4 header-controls"):
                    model_select = (
                        ui.select(
                            profile_options(),
                            value=initial_settings.model_profile,
                            label="Model",
                            on_change=change_model,
                        )
                        .props("outlined dense options-dense")
                        .classes("model-select")
                    )
                    with ui.row().classes("items-center gap-2 no-wrap model-status"):
                        model_status_dot = ui.element("span").classes("model-status-dot")
                        model_status_label = ui.label().classes(
                            "status-copy text-xs text-slate-400"
                        )
                    thinking_toggle = (
                        ui.switch(
                            "Thinking",
                            value=initial_settings.thinking_enabled,
                            on_change=change_thinking,
                        )
                        .props("dense color=deep-purple")
                        .classes("thinking-toggle")
                    )
                    thinking_toggle.tooltip(
                        f"Off: up to {MAX_TOKENS:,} output tokens; "
                        f"on: up to {THINKING_MAX_TOKENS:,}"
                    )
                    memory_toggle = (
                        ui.switch(
                            "Use memory",
                            value=initial_settings.memory_enabled,
                            on_change=change_memory,
                        )
                        .props("dense color=deep-purple")
                        .classes("memory-toggle")
                    )
                    memory_toggle.tooltip(
                        "Retrieve relevant context from other chats in this project"
                    )

            message_scroll = ui.scroll_area().classes("message-scroll")
            with message_scroll:
                messages_container = ui.column().classes("message-column")

            with ui.column().classes("composer-area gap-2"):
                with ui.row().classes("composer-row items-end"):
                    composer = (
                        ui.textarea(
                            placeholder="Message the model...",
                        )
                        .props("borderless autocomplete=off autogrow")
                        .classes("composer-input")
                    )
                    composer.props["aria-label"] = "Message"
                    # Enter sends (and is prevented from also inserting a
                    # newline); Shift+Enter isn't matched by .exact, so it
                    # falls through to the textarea's own default behavior.
                    composer.on("keydown.enter.exact.prevent", send_message)
                    stop_button = (
                        ui.button(
                            icon="stop",
                            on_click=stop_generation,
                        )
                        .props("round unelevated")
                        .classes("stop-button")
                    )
                    stop_button.props["aria-label"] = "Stop generation"
                    stop_button.disable()
                    stop_button.tooltip("Stop generation")
                    send_button = (
                        ui.button(
                            icon="arrow_upward",
                            on_click=send_message,
                        )
                        .props("round unelevated")
                        .classes("send-button")
                    )
                    send_button.props["aria-label"] = "Send message"
                    send_button.tooltip("Send message")

    render_all()
    ui.timer(5.0, lambda: render_model_status(current_settings().model_profile))
    ui.context.client.on_disconnect(
        lambda: page_event(
            logging.INFO,
            "ui.page.disconnected",
            generation_active=state.generating,
        )
    )
