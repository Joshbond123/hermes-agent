import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ClipboardEvent, type DragEvent, type KeyboardEvent } from "react";
import { btApi } from "../api";
import { Icon, Spinner, StopIcon } from "./ui";
import { useToast } from "./toastContext";

const MAX_CHARS = 32000;
const THINK_KEY = "bt-thinking";

interface Attachment {
  id: number;
  name: string;
  status: "uploading" | "ready" | "error";
  path?: string;
}

interface Props {
  busy: boolean;
  stopping: boolean;
  onSend: (text: string, opts: { thinking: boolean; attachments: Array<{ path: string; name: string }> }) => void;
  onStop: () => void;
  /** Text to place in the box (suggestion chips); `nonce` makes repeats work. */
  prefill: { text: string; nonce: number } | null;
  /** Bumped when the box should take focus (new chat opened). */
  focusSignal: unknown;
}

let attachmentId = 0;

export function Composer({ busy, stopping, onSend, onStop, prefill, focusSignal }: Props) {
  const toast = useToast();
  const [value, setValue] = useState("");
  const [files, setFiles] = useState<Attachment[]>([]);
  const [dragging, setDragging] = useState(false);
  const [thinking, setThinking] = useState(() => {
    try {
      return localStorage.getItem(THINK_KEY) === "1";
    } catch {
      return false;
    }
  });
  const area = useRef<HTMLTextAreaElement>(null);
  const picker = useRef<HTMLInputElement>(null);
  const composing = useRef(false);

  useLayoutEffect(() => {
    const el = area.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, Math.round(window.innerHeight * 0.4))}px`;
  }, [value]);

  // A suggestion chip was picked: adopt its text once per pick (adjusting state while rendering).
  const [seenPrefill, setSeenPrefill] = useState(0);
  if (prefill && prefill.nonce !== seenPrefill) {
    setSeenPrefill(prefill.nonce);
    setValue(prefill.text);
  }
  useEffect(() => {
    if (!prefill) return;
    requestAnimationFrame(() => {
      area.current?.focus();
      area.current?.setSelectionRange(prefill.text.length, prefill.text.length);
    });
  }, [prefill]);

  useEffect(() => {
    if (!window.matchMedia?.("(pointer: coarse)").matches) area.current?.focus();
  }, [focusSignal]);

  const uploading = files.some((f) => f.status === "uploading");
  const ready = files.filter((f) => f.status === "ready" && f.path);
  const canSend = !busy && !uploading && (value.trim().length > 0 || ready.length > 0) && value.length <= MAX_CHARS;

  const submit = useCallback(() => {
    if (!canSend) return;
    onSend(value.trim() || "Please look at the attached file(s).", {
      thinking,
      attachments: ready.map((f) => ({ path: f.path as string, name: f.name })),
    });
    setValue("");
    setFiles([]);
  }, [canSend, onSend, ready, thinking, value]);

  const addFiles = useCallback(
    (list: FileList | File[] | null) => {
      const incoming = Array.from(list ?? []).slice(0, 8);
      for (const file of incoming) {
        attachmentId += 1;
        const id = attachmentId;
        setFiles((l) => [...l, { id, name: file.name, status: "uploading" }]);
        btApi
          .uploadAttachment(file)
          .then((res) => setFiles((l) => l.map((f) => (f.id === id ? { ...f, status: "ready", path: res.path } : f))))
          .catch((err) => {
            setFiles((l) => l.map((f) => (f.id === id ? { ...f, status: "error" } : f)));
            toast(`Could not attach ${file.name}: ${err instanceof Error ? err.message : "upload failed"}`, "error");
          });
      }
    },
    [toast],
  );

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key !== "Enter" || e.shiftKey || e.altKey || composing.current || e.nativeEvent.isComposing) return;
    // On touch devices Enter is a newline (there is a Send button); on keyboards it sends.
    if (window.matchMedia?.("(pointer: coarse)").matches) return;
    e.preventDefault();
    submit();
  };

  const onPaste = (e: ClipboardEvent<HTMLTextAreaElement>) => {
    if (e.clipboardData.files.length) {
      e.preventDefault();
      addFiles(e.clipboardData.files);
    }
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    addFiles(e.dataTransfer.files);
  };

  const toggleThinking = () => {
    const next = !thinking;
    setThinking(next);
    try {
      localStorage.setItem(THINK_KEY, next ? "1" : "0");
    } catch {
      /* private mode */
    }
  };

  return (
    <form
      className={`bt-composer${dragging ? " is-dragging" : ""}`}
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      {files.length ? (
        <ul className="bt-chips" aria-label="Attachments">
          {files.map((f) => (
            <li key={f.id} className={`bt-chip bt-chip-${f.status}`}>
              {f.status === "uploading" ? <Spinner size={11} /> : null}
              <span className="bt-chip-name">{f.name}</span>
              {f.status === "error" ? <span className="bt-chip-err">failed</span> : null}
              <button type="button" className="bt-chip-x" aria-label={`Remove ${f.name}`} onClick={() => setFiles((l) => l.filter((x) => x.id !== f.id))}>
                <Icon name="x" size={12} />
              </button>
            </li>
          ))}
        </ul>
      ) : null}
      <textarea
        ref={area}
        className="bt-input"
        rows={1}
        value={value}
        maxLength={MAX_CHARS}
        placeholder={busy ? "Blackthorn is answering — you can type the next message…" : "Message Blackthorn"}
        aria-label="Message"
        enterKeyHint="send"
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={onKeyDown}
        onPaste={onPaste}
        onCompositionStart={() => (composing.current = true)}
        onCompositionEnd={() => (composing.current = false)}
      />
      <div className="bt-composer-bar">
        <input ref={picker} type="file" multiple hidden onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }} />
        <button type="button" className="bt-icon-btn" aria-label="Attach files" title="Attach files" onClick={() => picker.current?.click()}>
          <Icon name="paperclip" />
        </button>
        <button
          type="button"
          className={`bt-pill${thinking ? " is-on" : ""}`}
          aria-pressed={thinking}
          title="Let the model reason before it answers. Slower, but can be better on hard problems."
          onClick={toggleThinking}
        >
          <Icon name="spark" size={14} />
          Think
        </button>
        <span className="bt-hint">Enter to send · Shift+Enter for a new line</span>
        <span className="bt-spacer" />
        {value.length > MAX_CHARS * 0.9 ? <span className={`bt-count-note${value.length >= MAX_CHARS ? " bt-warn" : ""}`}>{value.length}/{MAX_CHARS}</span> : null}
        {busy ? (
          <button type="button" className="bt-send bt-stop" onClick={onStop} disabled={stopping} aria-label="Stop generating" title="Stop generating">
            {stopping ? <Spinner size={14} /> : <StopIcon />}
          </button>
        ) : (
          <button type="submit" className="bt-send" disabled={!canSend} aria-label="Send message" title={uploading ? "Waiting for the upload to finish" : "Send"}>
            <Icon name="send" size={18} />
          </button>
        )}
      </div>
    </form>
  );
}
