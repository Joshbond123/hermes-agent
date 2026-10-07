import { useEffect, useState } from "react";
import { btApi } from "../api";
import { Dialog, Spinner } from "./ui";
import { useToast } from "./toastContext";

export function SystemPromptDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const toast = useToast();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState("");
  const [value, setValue] = useState("");
  const [fallback, setFallback] = useState("");
  const [max, setMax] = useState(8000);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    btApi
      .getSystemPrompt()
      .then((res) => {
        if (!alive) return;
        setSaved(res.prompt);
        setValue(res.prompt);
        setFallback(res.default);
        setMax(res.max_chars);
      })
      .catch((e) => alive && setError(e instanceof Error ? e.message : "Could not load the system prompt."))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [open]);

  const dirty = value.trim() !== saved.trim();
  const tooLong = value.length > max;

  const requestClose = () => {
    if (saving) return;
    if (dirty && !window.confirm("Discard your unsaved changes to the system prompt?")) return;
    onClose();
  };

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const res = await btApi.putSystemPrompt(value);
      setSaved(res.prompt);
      setValue(res.prompt);
      toast(res.prompt ? "System prompt saved. It applies to your next message." : "Using the default system prompt.", "success");
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Saving failed.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog
      open={open}
      onClose={requestClose}
      title="System prompt"
      wide
      busy={saving}
      footer={
        <>
          <button type="button" className="bt-btn bt-btn-ghost" onClick={() => setValue("")} disabled={saving || loading || !value}>
            Reset to default
          </button>
          <span className="bt-spacer" />
          <button type="button" className="bt-btn" onClick={requestClose} disabled={saving}>Cancel</button>
          <button type="button" className="bt-btn bt-btn-primary" onClick={() => void save()} disabled={saving || loading || !dirty || tooLong}>
            {saving ? <Spinner /> : null} Save
          </button>
        </>
      }
    >
      <p className="bt-muted">
        Instructions sent to the model at the start of every conversation. Leave it empty to use the default persona.
        Tool-use and honesty rules are always applied.
      </p>
      {loading ? (
        <p><Spinner /> Loading…</p>
      ) : (
        <>
          <textarea
            className="bt-textarea" value={value} onChange={(e) => setValue(e.target.value)} rows={10} data-autofocus
            placeholder={fallback} aria-label="System prompt" spellCheck={false}
          />
          <div className="bt-dialog-meta">
            <span className={tooLong ? "bt-warn" : "bt-muted"}>{value.length}/{max}</span>
            {!value ? <span className="bt-muted">Default persona in use</span> : null}
          </div>
        </>
      )}
      {error ? <p className="bt-error-line" role="alert">{error}</p> : null}
    </Dialog>
  );
}
