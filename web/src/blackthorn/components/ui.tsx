import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
  type RefObject,
} from "react";
import { createPortal } from "react-dom";
import { useMedia } from "../hooks/useMedia";
import { ToastContext, type ToastKind } from "./toastContext";

// --------------------------------------------------------------------------- //
// icons (functional controls only — never used decoratively inside messages)
// --------------------------------------------------------------------------- //
const PATHS = {
  menu: "M4 6h16M4 12h16M4 18h16",
  panel: "M4 5h16v14H4zM9 5v14",
  plus: "M12 5v14M5 12h14",
  x: "M6 6l12 12M18 6L6 18",
  send: "M12 19V5M5 12l7-7 7 7",
  paperclip: "M21.4 11.1l-9.2 9.2a6 6 0 0 1-8.5-8.5l8.6-8.6A4 4 0 1 1 18 8.8l-8.6 8.6a2 2 0 0 1-2.8-2.8l8.5-8.5",
  chip: "M7 7h10v10H7zM9 3v4M15 3v4M9 17v4M15 17v4M3 9h4M3 15h4M17 9h4M17 15h4",
  dots: "M5 12h.01M12 12h.01M19 12h.01",
  pencil: "M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z",
  pin: "M12 17v5M9 3h6l-1 6 3 3v2H7v-2l3-3z",
  archive: "M3 5h18v4H3zM5 9v10h14V9M10 13h4",
  trash: "M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3",
  search: "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM21 21l-4.3-4.3",
  download: "M12 4v11M7 11l5 5 5-5M5 20h14",
  sliders: "M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6",
  chevron: "M6 9l6 6 6-6",
  chevronRight: "M9 6l6 6-6 6",
  check: "M5 13l4 4L19 7",
  external: "M14 4h6v6M20 4l-9 9M18 14v6H4V6h6",
  moon: "M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z",
  sun: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4",
  refresh: "M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7",
  copy: "M9 9h11v11H9zM5 15V4h11",
  arrowDown: "M12 5v14M5 12l7 7 7-7",
  alert: "M12 9v4M12 17h.01M10.3 3.9L2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z",
  spark: "M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M18 6l-2.5 2.5M8.5 15.5L6 18",
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 18, className }: { name: IconName; size?: number; className?: string }) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d={PATHS[name]} />
    </svg>
  );
}

export function StopIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false">
      <rect x="6" y="6" width="12" height="12" rx="2.5" />
    </svg>
  );
}

export function Spinner({ size = 14, label }: { size?: number; label?: string }) {
  return <span className="bt-spinner" role={label ? "status" : undefined} aria-label={label} style={{ width: size, height: size }} />;
}

// --------------------------------------------------------------------------- //
// toasts
// --------------------------------------------------------------------------- //
interface ToastItem {
  id: number;
  text: string;
  kind: ToastKind;
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const next = useRef(0);
  const push = useCallback((text: string, kind: ToastKind = "info") => {
    next.current += 1;
    const id = next.current;
    setItems((l) => [...l.slice(-3), { id, text, kind }]);
    setTimeout(() => setItems((l) => l.filter((t) => t.id !== id)), kind === "error" ? 9000 : 4500);
  }, []);
  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="bt-toasts" role="region" aria-label="Notifications" aria-live="polite">
        {items.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`bt-toast bt-toast-${t.kind}`}
            onClick={() => setItems((l) => l.filter((x) => x.id !== t.id))}
            title="Dismiss"
          >
            {t.text}
          </button>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

/** Portals live inside the chat root so its theme variables apply to them. */
function portalTarget(): HTMLElement {
  return (typeof document !== "undefined" && document.querySelector<HTMLElement>(".bt-chat-root")) || document.body;
}

// --------------------------------------------------------------------------- //
// popover (portal, anchored, flips, closes on outside click / Escape)
// --------------------------------------------------------------------------- //
const FOCUSABLE = 'a[href],button:not([disabled]),textarea,input,select,[tabindex]:not([tabindex="-1"])';

export function Popover({
  open,
  onClose,
  anchorRef,
  children,
  align = "end",
  className = "",
  role = "dialog",
  label,
  width,
}: {
  open: boolean;
  onClose: () => void;
  anchorRef: RefObject<HTMLElement | null>;
  children: ReactNode;
  align?: "start" | "end";
  className?: string;
  role?: "dialog" | "menu";
  label: string;
  width?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const sheet = useMedia("(max-width: 639px)"); // bottom sheet on phones

  // Position imperatively (no state): below the anchor, flipped above when there is no room.
  const place = useCallback(() => {
    const anchor = anchorRef.current;
    const pop = ref.current;
    if (!anchor || !pop || sheet) return;
    const a = anchor.getBoundingClientRect();
    const margin = 8;
    pop.style.maxHeight = `${window.innerHeight - 2 * margin}px`;
    const p = pop.getBoundingClientRect();
    let left = align === "end" ? a.right - p.width : a.left;
    left = Math.max(margin, Math.min(left, window.innerWidth - p.width - margin));
    let top = a.bottom + 6;
    if (top + p.height > window.innerHeight - margin && a.top - p.height - 6 > margin) top = a.top - p.height - 6;
    top = Math.max(margin, Math.min(top, window.innerHeight - p.height - margin));
    pop.style.left = `${left}px`;
    pop.style.top = `${top}px`;
    pop.style.visibility = "visible";
  }, [align, anchorRef, sheet]);

  useLayoutEffect(() => {
    if (open) place();
  });

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node;
      if (ref.current?.contains(t) || anchorRef.current?.contains(t)) return;
      onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
        anchorRef.current?.focus();
      }
    };
    document.addEventListener("pointerdown", onDown, true);
    document.addEventListener("keydown", onKey, true);
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    const raf = requestAnimationFrame(() => {
      const first = ref.current?.querySelector<HTMLElement>(role === "menu" ? '[role="menuitem"]:not([disabled])' : FOCUSABLE);
      first?.focus({ preventScroll: true });
    });
    return () => {
      cancelAnimationFrame(raf);
      document.removeEventListener("pointerdown", onDown, true);
      document.removeEventListener("keydown", onKey, true);
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open, onClose, anchorRef, place, role]);

  if (!open) return null;
  return createPortal(
    <>
      {sheet ? <div className="bt-sheet-backdrop" onClick={onClose} aria-hidden="true" /> : null}
      <div
        ref={ref}
        role={role}
        aria-label={label}
        className={`bt-popover${sheet ? " bt-sheet" : ""} ${className}`}
        style={sheet ? undefined : { visibility: "hidden", width }}
        onKeyDown={role === "menu" ? menuKeys : undefined}
      >
        {children}
      </div>
    </>,
    portalTarget(),
  );
}

function menuKeys(e: React.KeyboardEvent<HTMLDivElement>) {
  const items = Array.from(e.currentTarget.querySelectorAll<HTMLElement>('[role="menuitem"]:not([disabled])'));
  if (!items.length) return;
  const i = items.indexOf(document.activeElement as HTMLElement);
  let target: HTMLElement | undefined;
  if (e.key === "ArrowDown") target = items[(i + 1) % items.length];
  else if (e.key === "ArrowUp") target = items[(i - 1 + items.length) % items.length];
  else if (e.key === "Home") target = items[0];
  else if (e.key === "End") target = items[items.length - 1];
  if (target) {
    e.preventDefault();
    target.focus();
  }
}

// --------------------------------------------------------------------------- //
// menu button
// --------------------------------------------------------------------------- //
export interface MenuItem {
  key: string;
  label: string;
  icon?: IconName;
  onSelect: () => void;
  danger?: boolean;
  disabled?: boolean;
  separatorBefore?: boolean;
  checked?: boolean;
}

export function MenuButton({
  items,
  label,
  icon = "dots",
  className = "",
  align = "end",
  children,
  onOpenChange,
}: {
  items: MenuItem[];
  label: string;
  icon?: IconName;
  className?: string;
  align?: "start" | "end";
  children?: ReactNode;
  onOpenChange?: (open: boolean) => void;
}) {
  const [open, setOpen] = useState(false);
  const btn = useRef<HTMLButtonElement>(null);
  const set = useCallback(
    (v: boolean) => {
      setOpen(v);
      onOpenChange?.(v);
    },
    [onOpenChange],
  );
  const close = useCallback(() => set(false), [set]);
  return (
    <>
      <button
        ref={btn}
        type="button"
        className={className}
        aria-label={label}
        title={label}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={(e) => {
          e.stopPropagation();
          set(!open);
        }}
      >
        {children ?? <Icon name={icon} />}
      </button>
      <Popover open={open} onClose={close} anchorRef={btn} align={align} role="menu" label={label} className="bt-menu">
        {items.map((item) => (
          <div key={item.key} role="none">
            {item.separatorBefore ? <div className="bt-menu-sep" role="separator" /> : null}
            <button
              type="button"
              role="menuitem"
              className={`bt-menu-item${item.danger ? " bt-danger" : ""}`}
              disabled={item.disabled}
              onClick={() => {
                close();
                item.onSelect();
              }}
            >
              {item.icon ? <Icon name={item.icon} size={16} /> : <span className="bt-menu-icon-space" />}
              <span className="bt-menu-label">{item.label}</span>
              {item.checked ? <Icon name="check" size={15} className="bt-menu-check" /> : null}
            </button>
          </div>
        ))}
      </Popover>
    </>
  );
}

// --------------------------------------------------------------------------- //
// dialog (modal, focus trap)
// --------------------------------------------------------------------------- //
export function Dialog({
  open,
  onClose,
  title,
  children,
  footer,
  busy = false,
  wide = false,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  footer?: ReactNode;
  busy?: boolean;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const restore = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    restore.current = document.activeElement as HTMLElement | null;
    const raf = requestAnimationFrame(() => {
      const target = ref.current?.querySelector<HTMLElement>("[data-autofocus]") ?? ref.current?.querySelector<HTMLElement>(FOCUSABLE);
      target?.focus();
    });
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !busy) {
        e.stopPropagation();
        onClose();
      } else if (e.key === "Tab" && ref.current) {
        const nodes = Array.from(ref.current.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((n) => n.offsetParent !== null);
        if (!nodes.length) return;
        const first = nodes[0];
        const last = nodes[nodes.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => {
      cancelAnimationFrame(raf);
      document.removeEventListener("keydown", onKey, true);
      restore.current?.focus?.();
    };
  }, [open, busy, onClose]);

  if (!open) return null;
  return createPortal(
    <div className="bt-dialog-backdrop" onMouseDown={(e) => e.target === e.currentTarget && !busy && onClose()}>
      <div ref={ref} className={`bt-dialog${wide ? " bt-dialog-wide" : ""}`} role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <div className="bt-dialog-head">
          <h2 id={titleId}>{title}</h2>
          <button type="button" className="bt-icon-btn" aria-label="Close" onClick={onClose} disabled={busy}>
            <Icon name="x" />
          </button>
        </div>
        <div className="bt-dialog-body">{children}</div>
        {footer ? <div className="bt-dialog-foot">{footer}</div> : null}
      </div>
    </div>,
    portalTarget(),
  );
}
