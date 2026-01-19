import { Flame, CheckCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface DetectionResult {
  name: string;
  confidence: number;
  isHighRisk?: boolean;
}

interface DetectionResultsProps {
  polypDetected: boolean;
  confidenceScore: number;
  description: string;
  results: DetectionResult[];
  threshold: number;
}

export function DetectionResults({
  polypDetected,
  confidenceScore,
  description,
  results,
  threshold,
}: DetectionResultsProps) {
  return (
    <div
      className={cn(
        "mx-4 md:mx-8 mb-6 rounded-xl border-2 p-4 md:p-6 animate-fade-in",
        polypDetected
          ? "gradient-detection-alert border-detection-high/50"
          : "gradient-detection-normal border-primary/30"
      )}
    >
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 mb-4">
        <div className="flex items-center gap-3">
          {polypDetected ? (
            <>
              <div className="w-3 h-3 bg-detection-high rounded-full alert-pulse"></div>
              <h2 className="text-xl md:text-2xl font-bold text-foreground uppercase tracking-wide">
                Polyp Detected
              </h2>
            </>
          ) : (
            <>
              <CheckCircle className="w-6 h-6 text-primary" />
              <h2 className="text-xl md:text-2xl font-bold text-foreground">
                Detection Results
              </h2>
            </>
          )}
        </div>
        <div className="text-left sm:text-right">
          <div
            className={cn(
              "text-4xl md:text-5xl font-bold mb-1",
              polypDetected ? "text-foreground" : "text-primary"
            )}
          >
            {(confidenceScore * 100).toFixed(1)}%
          </div>
          <div
            className={cn(
              "text-xs uppercase tracking-wider px-2 py-1 rounded inline-block",
              polypDetected
                ? "bg-detection-high/20 text-detection-high"
                : "text-muted-foreground"
            )}
          >
            {polypDetected ? "Confidence Score" : "Probability Score"}
          </div>
        </div>
      </div>

      {/* Alert Message */}
      <p
        className={cn(
          "text-sm mb-6",
          polypDetected ? "text-foreground/80" : "text-muted-foreground"
        )}
      >
        {description}
      </p>

      {/* Detection Items */}
      <div className="space-y-4">
        {results.map((result, index) => (
          <div
            key={index}
            className={cn(
              "rounded-lg p-4 border",
              result.isHighRisk
                ? "bg-black/20 border-detection-high/30"
                : "bg-card/50 border-border"
            )}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                {result.isHighRisk ? (
                  <Flame className="w-5 h-5 text-detection-high" />
                ) : (
                  <div
                    className={cn(
                      "w-2 h-2 rounded-full",
                      polypDetected ? "bg-muted-foreground" : "bg-primary"
                    )}
                  ></div>
                )}
                <span
                  className={cn(
                    "font-semibold",
                    result.isHighRisk
                      ? "text-foreground"
                      : polypDetected
                      ? "text-muted-foreground"
                      : "text-foreground"
                  )}
                >
                  {result.name}
                </span>
              </div>
              <span
                className={cn(
                  "font-mono font-bold",
                  result.isHighRisk ? "text-detection-high" : "text-primary"
                )}
              >
                {result.confidence.toFixed(3)}
              </span>
            </div>

            {/* Progress Bar */}
            <div className="relative">
              <div
                className={cn(
                  "h-2 rounded-full overflow-hidden",
                  polypDetected ? "bg-black/30" : "bg-secondary"
                )}
              >
                <div
                  className={cn(
                    "h-full rounded-full transition-all duration-500",
                    result.isHighRisk
                      ? "bg-gradient-to-r from-detection-high/80 to-detection-high"
                      : "bg-gradient-to-r from-primary/80 to-primary"
                  )}
                  style={{ width: `${result.confidence * 100}%` }}
                ></div>
              </div>
              {result.isHighRisk && (
                <div
                  className="absolute top-1/2 -translate-y-1/2 w-0.5 h-4 bg-muted-foreground"
                  style={{ left: `${threshold * 100}%` }}
                >
                  <span className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs text-muted-foreground uppercase whitespace-nowrap">
                    Threshold
                  </span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
