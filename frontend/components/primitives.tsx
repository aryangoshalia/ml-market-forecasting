import type { ReactNode } from "react";

/** Layout primitives. Sections and rules, not floating cards. */

export function Section({
  id,
  title,
  subtitle,
  aside,
  children,
}: {
  id?: string;
  title: string;
  subtitle?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section id={id} className="border-t border-line py-7">
      <header className="mb-4 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <div>
          <h2 className="text-[13px] font-semibold uppercase tracking-[0.08em] text-ink">{title}</h2>
          {subtitle ? <p className="mt-1 max-w-3xl text-2xs text-muted">{subtitle}</p> : null}
        </div>
        {aside ? <div className="text-2xs text-faint tabular">{aside}</div> : null}
      </header>
      {children}
    </section>
  );
}

export function Metric({
  label,
  value,
  hint,
  tone = "neutral",
  mono = true,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "neutral" | "up" | "down" | "muted";
  mono?: boolean;
}) {
  const toneClass =
    tone === "up" ? "text-up" : tone === "down" ? "text-down" : tone === "muted" ? "text-muted" : "text-ink";
  return (
    <div className="min-w-0">
      <div className="text-2xs uppercase tracking-[0.06em] text-faint">{label}</div>
      <div className={`mt-0.5 text-[15px] ${mono ? "font-mono" : ""} tabular ${toneClass}`}>{value}</div>
      {hint ? <div className="mt-0.5 text-2xs text-faint">{hint}</div> : null}
    </div>
  );
}

export function MetricRow({ children }: { children: ReactNode }) {
  return <div className="grid grid-cols-2 gap-x-8 gap-y-4 sm:grid-cols-3 lg:grid-cols-6">{children}</div>;
}

export function Chip({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "up" | "down" | "warn" | "accent";
}) {
  const tones = {
    neutral: "border-line text-muted",
    up: "border-up/40 text-up",
    down: "border-down/40 text-down",
    warn: "border-warn/40 text-warn",
    accent: "border-accent/40 text-accent",
  };
  return (
    <span
      className={`inline-flex items-center border px-1.5 py-0.5 text-2xs uppercase tracking-[0.06em] ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function Note({ children }: { children: ReactNode }) {
  return (
    <p className="mt-3 border-l-2 border-line pl-3 text-2xs leading-relaxed text-muted">{children}</p>
  );
}

export function Table({ children }: { children: ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] border-collapse text-[12px]">{children}</table>
    </div>
  );
}

export function Th({
  children,
  align = "right",
  title,
}: {
  children: ReactNode;
  align?: "left" | "right";
  title?: string;
}) {
  return (
    <th
      title={title}
      className={`border-b border-line pb-1.5 pr-3 text-2xs font-medium uppercase tracking-[0.06em] text-faint ${
        align === "left" ? "text-left" : "text-right"
      }`}
    >
      {children}
    </th>
  );
}

export function Td({
  children,
  align = "right",
  className = "",
}: {
  children: ReactNode;
  align?: "left" | "right";
  className?: string;
}) {
  return (
    <td
      className={`border-b border-line/60 py-1.5 pr-3 ${
        align === "left" ? "text-left" : "text-right font-mono tabular"
      } ${className}`}
    >
      {children}
    </td>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="py-8 text-center text-2xs text-faint">{children}</div>;
}

export function Bar({ value, max, tone }: { value: number; max: number; tone: "up" | "down" | "accent" }) {
  const width = max > 0 ? Math.min(100, (Math.abs(value) / max) * 100) : 0;
  const colour = tone === "up" ? "bg-up" : tone === "down" ? "bg-down" : "bg-accent";
  return (
    <span className="inline-block h-2 w-full max-w-[120px] bg-line/60 align-middle">
      <span className={`block h-full ${colour}`} style={{ width: `${width}%` }} />
    </span>
  );
}
