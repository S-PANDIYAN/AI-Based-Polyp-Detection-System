import { Maximize2, Film, Clock, Target, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

interface Detection {
  label: string;
  confidence: number;
  x: number;
  y: number;
  isHighRisk: boolean;
}

interface ImageComparisonProps {
  originalImage: string;
  frameNumber: string;
  inferenceTime: string;
  detections: Detection[];
  polypDetected: boolean;
  showProbability: boolean;
}

export function ImageComparison({
  originalImage,
  frameNumber,
  inferenceTime,
  detections,
  polypDetected,
  showProbability,
}: ImageComparisonProps) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 md:gap-6 px-4 md:px-8 py-6">
      {/* Original Image */}
      <div className="bg-card rounded-xl border border-border overflow-hidden animate-fade-in">
        {/* Header */}
        <div className="bg-panel-header px-4 py-3 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-muted-foreground rounded-full"></div>
            <span className="text-foreground font-semibold text-sm uppercase tracking-wide">
              Original Input
            </span>
          </div>
          <button className="text-muted-foreground hover:text-foreground transition">
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>

        {/* Image Container */}
        <div className="relative bg-black aspect-square flex items-center justify-center">
          <img
            src={originalImage}
            alt="Original colonoscopy"
            className="max-w-full max-h-full object-contain"
          />
        </div>

        {/* Footer */}
        <div className="bg-panel-header px-4 py-2 border-t border-border">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Film className="w-3 h-3" />
            <span className="font-mono">FRAME: {frameNumber}</span>
          </div>
        </div>
      </div>

      {/* AI Inference Image */}
      <div
        className={cn(
          "bg-card rounded-xl border overflow-hidden animate-fade-in",
          polypDetected ? "border-detection-high/50" : "border-border"
        )}
      >
        {/* Header */}
        <div className="bg-panel-header px-4 py-3 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 bg-primary rounded-full animate-pulse"></div>
            <span className="text-primary font-semibold text-sm uppercase tracking-wide">
              AI Inference
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="bg-secondary text-muted-foreground px-2 py-1 rounded text-xs flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {inferenceTime}
            </span>
            {polypDetected && (
              <span className="bg-detection-high-bg text-detection-high px-2 py-1 rounded text-xs font-semibold uppercase flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" />
                Polyp Detected
              </span>
            )}
            {!polypDetected && (
              <span className="bg-primary/20 text-primary px-2 py-1 rounded text-xs font-semibold uppercase">
                YOLOv8
              </span>
            )}
          </div>
        </div>

        {/* Image Container with Overlays */}
        <div className="relative bg-black aspect-square flex items-center justify-center">
          <img
            src={originalImage}
            alt="AI detection"
            className="max-w-full max-h-full object-contain"
          />

          {/* Detection Overlays */}
          {showProbability &&
            detections.map((detection, index) => (
              <div
                key={index}
                className="absolute transform -translate-x-1/2"
                style={{ top: `${detection.y}%`, left: `${detection.x}%` }}
              >
                <div
                  className={cn(
                    "px-3 py-1.5 rounded-md text-xs font-bold flex items-center gap-2 shadow-lg whitespace-nowrap",
                    detection.isHighRisk
                      ? "bg-detection-high text-white"
                      : "bg-warning text-white"
                  )}
                >
                  {detection.isHighRisk && <AlertTriangle className="w-3 h-3" />}
                  {detection.label} {(detection.confidence * 100).toFixed(1)}%
                </div>
              </div>
            ))}

          {/* Bounding Box Simulation */}
          {polypDetected && (
            <div className="absolute top-1/4 left-1/3 w-1/3 h-1/3 border-2 border-primary rounded pointer-events-none" />
          )}
        </div>

        {/* Footer */}
        <div className="bg-panel-header px-4 py-2 border-t border-border">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Target className="w-3 h-3" />
            <span>{detections.length} detection(s) identified</span>
          </div>
        </div>
      </div>
    </div>
  );
}
