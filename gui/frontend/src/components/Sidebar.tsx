import React, { useEffect, useMemo, useState } from 'react';
import { CheckCircle2, Menu, X } from 'lucide-react';
import { NAV_ITEMS, type NavItem, type ViewId } from '../nav';

interface SidebarProps {
  active: ViewId;
  onSelect: (id: ViewId) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

interface SidebarCompletion {
  complete: boolean;
  title: string;
}

type CompletionMap = Partial<Record<ViewId, SidebarCompletion>>;

const fetchJsonWithTimeout = async <T,>(url: string, timeoutMs = 5500): Promise<T | null> => {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) return null;
    return (await response.json()) as T;
  } catch {
    return null;
  } finally {
    window.clearTimeout(timeout);
  }
};

export const Sidebar: React.FC<SidebarProps> = ({ active, onSelect, open, onOpenChange }) => {
  const [completion, setCompletion] = useState<CompletionMap>({});

  const grouped = useMemo(() => {
    const map = new Map<string, { label: string; items: NavItem[] }>();
    for (const item of NAV_ITEMS) {
      if (!map.has(item.group)) {
        map.set(item.group, { label: item.groupLabel, items: [] });
      }
      map.get(item.group)!.items.push(item);
    }
    return Array.from(map.values());
  }, []);

  useEffect(() => {
    let cancelled = false;
    const loadCompletion = async () => {
      const payload = await fetchJsonWithTimeout<{ items?: CompletionMap }>('/api/sidebar/completion', 8000);
      if (cancelled) return;
      setCompletion(payload?.items ?? {});
    };
    void loadCompletion();
    const interval = window.setInterval(loadCompletion, 60000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  // Close drawer on Escape (mobile)
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onOpenChange(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onOpenChange]);

  const handleSelect = (id: ViewId) => {
    onSelect(id);
    onOpenChange(false);
  };

  return (
    <>
      <button
        type="button"
        className="hamburger"
        aria-label={open ? 'Close navigation' : 'Open navigation'}
        aria-expanded={open}
        onClick={() => onOpenChange(!open)}
      >
        {open ? <X size={18} /> : <Menu size={18} />}
      </button>

      {open ? <div className="sidebar-scrim" onClick={() => onOpenChange(false)} aria-hidden /> : null}

      <aside className={`sidebar${open ? ' is-open' : ''}`} aria-label="Primary navigation">
        <div className="sidebar-brand">
          <div className="brand-mark" aria-hidden />
          <div>
            <div className="brand-title">CARA</div>
            <div className="brand-sub">Attribution Console</div>
          </div>
        </div>

        <nav className="sidebar-nav">
          {grouped.map((group) => (
            <div className="sidebar-group" key={group.label}>
              <div className="sidebar-group-label">{group.label}</div>
              {group.items.map((item) => {
                const Icon = item.icon;
                const isActive = item.id === active;
                const complete = completion[item.id]?.complete;
                return (
                  <button
                    key={item.id}
                    type="button"
                    className={`sidebar-link${isActive ? ' is-active' : ''}${complete ? ' is-complete' : ''}`}
                    onClick={() => handleSelect(item.id)}
                    aria-current={isActive ? 'page' : undefined}
                    title={complete ? completion[item.id]?.title : item.description}
                  >
                    <Icon size={16} />
                    <span>{item.label}</span>
                    {complete ? (
                      <CheckCircle2 className="sidebar-complete-icon" size={15} aria-label={completion[item.id]?.title} />
                    ) : null}
                  </button>
                );
              })}
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <span className="live-chip" id="connection-chip">
            <span className="live-dot" id="connection-dot" />
            <span id="connection-text">Live</span>
          </span>
        </div>
      </aside>
    </>
  );
};
