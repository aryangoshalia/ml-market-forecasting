"use client";

import type { RegimeTimeline } from "@/lib/schema";
import { shortDate } from "@/lib/format";
import { Empty, Note, Section, Table, Td, Th } from "./primitives";

const STATE_COLOURS = ["#1c5fd6", "#8a6410", "#a8231d", "#17683c", "#6b6b66"];

export function RegimePanel({ data }: { data?: RegimeTimeline }) {
  if (!data) {
    return (
      <Section id="regime" title="Market regime">
        <Empty>loading regimes</Empty>
      </Section>
    );
  }

  const first = Date.parse(`${data.episodes[0]?.start ?? "2012-01-01"}T00:00:00Z`);
  const last = Date.parse(`${data.episodes.at(-1)?.end ?? "2026-01-01"}T00:00:00Z`);
  const span = Math.max(last - first, 1);

  const totals = new Map<number, number>();
  for (const episode of data.episodes) {
    totals.set(episode.state, (totals.get(episode.state) ?? 0) + episode.sessions);
  }
  const allSessions = [...totals.values()].reduce((a, b) => a + b, 0);

  return (
    <Section
      id="regime"
      title="Market regime"
      subtitle="A hidden Markov model on market-level features, refitted inside every fold. Labels are generated from each state's fitted statistics after the fact, so no cluster was called bullish or bearish in advance."
      aside={`${data.model} · ${data.n_states} states · ${data.inference} inference`}
    >
      <div className="mb-2 flex h-8 w-full overflow-hidden border border-line">
        {data.episodes.map((episode, index) => {
          const start = Date.parse(`${episode.start}T00:00:00Z`);
          // Extend to where the next episode begins rather than to this one's last
          // session, so the segments tile the axis instead of leaving a gap per episode.
          const next = data.episodes[index + 1];
          const end = next
            ? Date.parse(`${next.start}T00:00:00Z`)
            : Date.parse(`${episode.end}T00:00:00Z`);
          const width = ((end - start) / span) * 100;
          return (
            <span
              key={`${episode.start}-${index}`}
              title={`${episode.name} · ${shortDate(episode.start)} to ${shortDate(episode.end)} · ${episode.sessions} sessions`}
              style={{
                width: `${Math.max(width, 0.08)}%`,
                background: STATE_COLOURS[episode.state % STATE_COLOURS.length],
              }}
            />
          );
        })}
      </div>
      <div className="mb-5 flex justify-between text-2xs text-faint tabular">
        <span>{shortDate(data.episodes[0]?.start ?? "")}</span>
        <span>{shortDate(data.episodes.at(-1)?.end ?? "")}</span>
      </div>

      <Table>
        <thead>
          <tr>
            <Th align="left">State</Th>
            <Th align="left">Description</Th>
            <Th>Sessions</Th>
            <Th>Share</Th>
            <Th>Episodes</Th>
          </tr>
        </thead>
        <tbody>
          {[...totals.entries()]
            .sort((a, b) => a[0] - b[0])
            .map(([state, sessions]) => (
              <tr key={state}>
                <Td align="left">
                  <span
                    className="mr-2 inline-block h-2 w-2 align-middle"
                    style={{ background: STATE_COLOURS[state % STATE_COLOURS.length] }}
                  />
                  <span className="font-mono">s{state}</span>
                </Td>
                <Td align="left" className="text-muted">
                  {data.names[String(state)] ?? `state ${state}`}
                </Td>
                <Td>{sessions.toLocaleString()}</Td>
                <Td>{((sessions / allSessions) * 100).toFixed(1)}%</Td>
                <Td>{data.episodes.filter((e) => e.state === state).length}</Td>
              </tr>
            ))}
        </tbody>
      </Table>

      <Note>
        Only filtered probabilities are used, meaning the regime shown for a session was
        inferable from data up to that session. Smoothed inference would read the whole sample
        and disagrees on about four percent of days, which would be look-ahead.
      </Note>
    </Section>
  );
}
