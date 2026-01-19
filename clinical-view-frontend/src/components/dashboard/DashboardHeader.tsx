import { Home, ChevronRight, CloudUpload, Printer, Play, IdCard, Calendar } from "lucide-react";

interface DashboardHeaderProps {
  caseId: string;
  patientId: string;
  date: string;
  status: "pending" | "completed" | "error";
  onNewScan: () => void;
  onPrint: () => void;
  onReRun: () => void;
}

export function DashboardHeader({
  caseId,
  patientId,
  date,
  status,
  onNewScan,
  onPrint,
  onReRun,
}: DashboardHeaderProps) {
  const getStatusBadge = () => {
    switch (status) {
      case "completed":
        return (
          <span className="bg-success/20 text-success px-3 py-1 rounded-md text-sm font-semibold uppercase">
            Completed
          </span>
        );
      case "pending":
        return (
          <span className="bg-warning/20 text-warning px-3 py-1 rounded-md text-sm font-semibold uppercase">
            Pending
          </span>
        );
      case "error":
        return (
          <span className="bg-destructive/20 text-destructive px-3 py-1 rounded-md text-sm font-semibold uppercase">
            Error
          </span>
        );
    }
  };

  return (
    <header className="bg-panel-header border-b border-border px-4 md:px-8 py-6">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-muted-foreground mb-4">
        <Home className="w-4 h-4" />
        <ChevronRight className="w-4 h-4" />
        <span>Diagnostics</span>
        <ChevronRight className="w-4 h-4" />
        <span className="bg-secondary text-foreground px-3 py-1 rounded">{caseId}</span>
      </div>

      {/* Title Section */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-4">
        <div className="flex flex-wrap items-center gap-4">
          <h1 className="text-2xl md:text-3xl font-bold text-foreground">
            Colonoscopy Analysis
          </h1>
          {getStatusBadge()}
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={onNewScan}
            className="border-2 border-dashed border-border text-muted-foreground px-4 py-2 rounded-lg hover:border-primary hover:text-primary transition flex items-center gap-2"
          >
            <CloudUpload className="w-5 h-5" />
            <span className="hidden sm:inline">Drop new scan</span>
          </button>

          <button
            onClick={onPrint}
            className="bg-secondary text-foreground px-4 py-2 rounded-lg hover:bg-secondary/80 transition flex items-center gap-2"
          >
            <Printer className="w-5 h-5" />
            <span className="hidden sm:inline">Print</span>
          </button>

          <button
            onClick={onReRun}
            className="bg-primary text-primary-foreground px-4 md:px-6 py-2 rounded-lg hover:bg-primary/90 transition flex items-center gap-2 font-semibold"
          >
            <Play className="w-5 h-5" />
            Re-Run Analysis
          </button>
        </div>
      </div>

      {/* Meta Information */}
      <div className="flex flex-wrap items-center gap-4 md:gap-6 text-sm text-muted-foreground">
        <div className="flex items-center gap-2">
          <IdCard className="w-4 h-4" />
          <span>
            Patient ID: <span className="text-foreground">{patientId}</span>
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Calendar className="w-4 h-4" />
          <span>{date}</span>
        </div>
      </div>
    </header>
  );
}
