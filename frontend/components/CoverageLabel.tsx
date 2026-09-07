import type { Coverage } from "@/lib/schema";
import { shortDate } from "@/lib/format";

/** Every panel backed by recorded results says which period it covers. The same label
 *  renders whether the results shipped with the repository or were produced locally;
 *  only the dates move. */
export function CoverageLabel({ coverage }: { coverage: Coverage }) {
  return (
    <span className="tabular">
      walk-forward {shortDate(coverage.first_session)} to {shortDate(coverage.last_session)} ·{" "}
      {coverage.folds} folds
    </span>
  );
}
