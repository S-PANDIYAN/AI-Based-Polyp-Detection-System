import { useState } from "react";
import {
  LayoutDashboard,
  History,
  FolderOpen,
  Settings,
  ChevronDown,
  Activity,
  Moon,
  Sun,
  Menu,
  X,
} from "lucide-react";
import { useTheme } from "@/contexts/ThemeContext";
import { cn } from "@/lib/utils";

interface SidebarProps {
  sensitivityThreshold: number;
  setSensitivityThreshold: (value: number) => void;
  showProbability: boolean;
  setShowProbability: (value: boolean) => void;
  autoExport: boolean;
  setAutoExport: (value: boolean) => void;
}

export function Sidebar({
  sensitivityThreshold,
  setSensitivityThreshold,
  showProbability,
  setShowProbability,
  autoExport,
  setAutoExport,
}: SidebarProps) {
  const { theme, toggleTheme } = useTheme();
  const [activeNav, setActiveNav] = useState("dashboard");
  const [mobileOpen, setMobileOpen] = useState(false);

  const navItems = [
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "history", label: "Case History", icon: History },
    { id: "datasets", label: "Datasets", icon: FolderOpen },
    { id: "settings", label: "Model Settings", icon: Settings },
  ];

  const SidebarContent = () => (
    <>
      {/* Logo Section */}
      <div className="p-6 border-b border-sidebar-border">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-primary rounded-lg flex items-center justify-center">
            <Activity className="w-7 h-7 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-foreground font-bold text-xl">PolypNet AI</h1>
            <p className="text-success text-xs flex items-center gap-1">
              <span className="w-2 h-2 bg-success rounded-full status-pulse"></span>
              v2.4 (YOLOv8)
            </p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 space-y-2 overflow-y-auto">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => setActiveNav(item.id)}
            className={cn(
              "w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all",
              activeNav === item.id
                ? "bg-sidebar-accent text-sidebar-accent-foreground border border-primary/30"
                : "text-sidebar-foreground hover:text-foreground hover:bg-secondary"
            )}
          >
            <item.icon className="w-5 h-5" />
            <span className="font-medium">{item.label}</span>
          </button>
        ))}

        {/* Inference Parameters */}
        <div className="pt-6">
          <h3 className="text-muted-foreground text-xs font-semibold uppercase tracking-wider px-4 mb-4">
            Inference Parameters
          </h3>

          {/* Sensitivity Slider */}
          <div className="px-4 mb-6">
            <div className="flex items-center justify-between mb-2">
              <label className="text-foreground text-sm">Sensitivity Threshold</label>
              <span className="bg-secondary text-primary font-mono text-sm px-2 py-0.5 rounded">
                {sensitivityThreshold.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={sensitivityThreshold}
              onChange={(e) => setSensitivityThreshold(parseFloat(e.target.value))}
              className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>0</span>
              <span>1</span>
            </div>
          </div>

          {/* Toggle Switches */}
          <div className="px-4 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-foreground text-sm">Show Probability</span>
              <button
                onClick={() => setShowProbability(!showProbability)}
                className={cn(
                  "w-11 h-6 rounded-full relative transition-colors",
                  showProbability ? "bg-primary" : "bg-muted"
                )}
              >
                <span
                  className={cn(
                    "absolute top-1 w-4 h-4 bg-white rounded-full transition-transform",
                    showProbability ? "right-1" : "left-1"
                  )}
                ></span>
              </button>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-foreground text-sm">Auto-Export Report</span>
              <button
                onClick={() => setAutoExport(!autoExport)}
                className={cn(
                  "w-11 h-6 rounded-full relative transition-colors",
                  autoExport ? "bg-primary" : "bg-muted"
                )}
              >
                <span
                  className={cn(
                    "absolute top-1 w-4 h-4 bg-white rounded-full transition-transform",
                    autoExport ? "right-1" : "left-1"
                  )}
                ></span>
              </button>
            </div>

            {/* Theme Toggle */}
            <div className="flex items-center justify-between pt-2">
              <span className="text-foreground text-sm">Theme</span>
              <button
                onClick={toggleTheme}
                className="w-11 h-6 bg-secondary rounded-full relative flex items-center justify-center"
              >
                {theme === "dark" ? (
                  <Moon className="w-4 h-4 text-primary" />
                ) : (
                  <Sun className="w-4 h-4 text-warning" />
                )}
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* System Status */}
      <div className="p-4 border-t border-sidebar-border">
        <div className="bg-card rounded-lg p-3 mb-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="w-2 h-2 bg-success rounded-full status-pulse"></span>
            <span className="text-success font-semibold text-sm">System Ready</span>
          </div>
          <p className="text-muted-foreground text-xs">Latency: 42ms • GPU: Active</p>
        </div>

        {/* User Profile */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center text-primary font-semibold">
            AS
          </div>
          <div className="flex-1">
            <p className="text-foreground text-sm font-medium">Dr. A. Smith</p>
            <p className="text-muted-foreground text-xs">Chief Radiologist</p>
          </div>
          <ChevronDown className="w-4 h-4 text-muted-foreground" />
        </div>
      </div>
    </>
  );

  return (
    <>
      {/* Mobile Menu Button */}
      <button
        onClick={() => setMobileOpen(!mobileOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-card rounded-lg border border-border"
      >
        {mobileOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
      </button>

      {/* Mobile Overlay */}
      {mobileOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/50 z-40"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar - Desktop */}
      <aside className="hidden lg:flex w-72 h-screen bg-sidebar border-r border-sidebar-border flex-col fixed left-0 top-0">
        <SidebarContent />
      </aside>

      {/* Sidebar - Mobile */}
      <aside
        className={cn(
          "lg:hidden fixed left-0 top-0 w-72 h-screen bg-sidebar border-r border-sidebar-border flex flex-col z-50 transition-transform duration-300",
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <SidebarContent />
      </aside>
    </>
  );
}
