import { useState, useRef, useCallback } from "react";
import { Upload, Image as ImageIcon, AlertCircle, X, FolderOpen, Info, Cpu } from "lucide-react";
import { cn } from "@/lib/utils";

interface ImageUploadProps {
  onImageSelected: (file: File, preview: string) => void;
  onAnalyze: () => void;
  hasImage: boolean;
  isAnalyzing: boolean;
  uploadError: string | null;
  onClearError: () => void;
}

export function ImageUpload({
  onImageSelected,
  onAnalyze,
  hasImage,
  isAnalyzing,
  uploadError,
  onClearError,
}: ImageUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validFileTypes = ["image/png", "image/jpg", "image/jpeg", "image/bmp"];

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);

      const file = e.dataTransfer.files[0];
      if (file && validFileTypes.includes(file.type)) {
        const reader = new FileReader();
        reader.onload = (e) => {
          onImageSelected(file, e.target?.result as string);
        };
        reader.readAsDataURL(file);
      }
    },
    [onImageSelected]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file && validFileTypes.includes(file.type)) {
        const reader = new FileReader();
        reader.onload = (e) => {
          onImageSelected(file, e.target?.result as string);
        };
        reader.readAsDataURL(file);
      }
    },
    [onImageSelected]
  );

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="mx-4 md:mx-8 my-6">
      {/* Error Banner */}
      {uploadError && (
        <div className="mb-4 bg-detection-high-bg border-2 border-detection-high/50 rounded-xl p-4 flex items-start justify-between animate-fade-in">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-detection-high flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="text-detection-high font-semibold uppercase text-sm">Upload Failed</h4>
              <p className="text-foreground/80 text-sm mt-1">
                The file <code className="bg-secondary px-2 py-0.5 rounded text-xs">{uploadError}</code> could not be processed.
              </p>
            </div>
          </div>
          <button
            onClick={onClearError}
            className="text-muted-foreground hover:text-foreground transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      )}

      {/* Upload Card */}
      <div className="bg-card rounded-xl border border-border overflow-hidden">
        {/* Header */}
        <div className="bg-panel-header px-4 md:px-6 py-4 border-b border-border flex items-center justify-between">
          <div>
            <h3 className="text-foreground font-semibold text-lg">Image Upload Section</h3>
            <p className="text-muted-foreground text-sm">SUPPORTED: PNG, JPG, JPEG, BMP</p>
          </div>
          <span
            className={cn(
              "px-3 py-1 rounded-full text-xs font-semibold uppercase flex items-center gap-1",
              uploadError
                ? "bg-detection-high/20 text-detection-high"
                : "bg-primary/20 text-primary"
            )}
          >
            <span className={cn("w-2 h-2 rounded-full", uploadError ? "bg-detection-high" : "bg-primary")}></span>
            {uploadError ? "Error" : "Ready for Upload"}
          </span>
        </div>

        {/* Drop Zone */}
        <div className="p-4 md:p-6">
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={cn(
              "border-2 border-dashed rounded-xl p-8 md:p-12 text-center transition-all cursor-pointer",
              isDragOver
                ? "border-primary bg-primary/5"
                : uploadError
                ? "border-detection-high/50 bg-detection-high/5"
                : "border-border hover:border-primary/50"
            )}
            onClick={handleBrowseClick}
          >
            {uploadError ? (
              <>
                <div className="w-16 h-16 mx-auto mb-4 bg-detection-high/20 rounded-full flex items-center justify-center">
                  <ImageIcon className="w-8 h-8 text-detection-high" />
                  <X className="w-4 h-4 text-detection-high absolute" />
                </div>
                <h4 className="text-foreground font-semibold text-lg mb-2">Invalid file type</h4>
                <div className="bg-detection-high text-white px-4 py-2 rounded-lg inline-flex items-center gap-2 mb-4">
                  <AlertCircle className="w-4 h-4" />
                  Please upload PNG, JPG, JPEG, or BMP.
                </div>
                <p className="text-primary hover:underline cursor-pointer">Select a different file</p>
              </>
            ) : (
              <>
                <div className="w-16 h-16 mx-auto mb-4 bg-primary/20 rounded-full flex items-center justify-center">
                  <Upload className="w-8 h-8 text-primary" />
                </div>
                <h4 className="text-foreground font-semibold text-lg mb-2">
                  Drag & drop your scan here
                </h4>
                <p className="text-muted-foreground text-sm mb-6">
                  Supports high-resolution medical imaging
                </p>
                <div className="flex items-center justify-center gap-4">
                  <div className="h-px w-12 bg-border"></div>
                  <span className="text-muted-foreground text-sm">OR</span>
                  <div className="h-px w-12 bg-border"></div>
                </div>
                <button
                  type="button"
                  className="mt-6 bg-primary text-primary-foreground px-6 py-3 rounded-lg font-semibold hover:bg-primary/90 transition inline-flex items-center gap-2"
                >
                  <FolderOpen className="w-5 h-5" />
                  Browse Files
                </button>
              </>
            )}
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".png,.jpg,.jpeg,.bmp"
            onChange={handleFileSelect}
            className="hidden"
          />
        </div>

        {/* Footer */}
        <div className="bg-panel-header px-4 md:px-6 py-3 border-t border-border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-muted-foreground text-sm">
            <Info className="w-4 h-4" />
            Maximum file size: 50MB
          </div>
          <button
            onClick={onAnalyze}
            disabled={!hasImage || isAnalyzing}
            className={cn(
              "px-6 py-2 rounded-lg font-semibold flex items-center gap-2 transition",
              hasImage && !isAnalyzing
                ? "bg-primary text-primary-foreground hover:bg-primary/90"
                : "bg-muted text-muted-foreground cursor-not-allowed"
            )}
          >
            <Cpu className="w-5 h-5" />
            {isAnalyzing ? "Analyzing..." : "Analyze Image"}
          </button>
        </div>
      </div>
    </div>
  );
}
