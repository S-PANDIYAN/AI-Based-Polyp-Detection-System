import { useState, useCallback, useEffect } from "react";
import { Sidebar } from "@/components/dashboard/Sidebar";
import { DashboardHeader } from "@/components/dashboard/DashboardHeader";
import { ImageComparison } from "@/components/dashboard/ImageComparison";
import { DetectionResults } from "@/components/dashboard/DetectionResults";
import { ExportPanel } from "@/components/dashboard/ExportPanel";
import { ImageUpload } from "@/components/dashboard/ImageUpload";
import { analyzeImage, Detection, DetectionResponse } from "@/services/polypDetectionAPI";
import { useToast } from "@/hooks/use-toast";
import colonoscopyImage from "@/assets/colonoscopy-scan.jpg";

type ViewMode = "upload" | "results";

const Index = () => {
  const { toast } = useToast();
  
  // Sidebar state
  const [sensitivityThreshold, setSensitivityThreshold] = useState(0.75);
  const [showProbability, setShowProbability] = useState(true);
  const [autoExport, setAutoExport] = useState(false);

  // View mode
  const [viewMode, setViewMode] = useState<ViewMode>("upload");

  // Upload state
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Analysis state
  const [analysisStatus, setAnalysisStatus] = useState<"pending" | "completed" | "error">("completed");
  const [polypDetected, setPolypDetected] = useState(false);
  const [apiResponse, setApiResponse] = useState<DetectionResponse | null>(null);
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null);

  // Case data
  const caseData = {
    caseId: `Case #${Math.floor(Math.random() * 10000)}`,
    patientId: `${Math.floor(Math.random() * 900 + 100)}-${Math.floor(Math.random() * 90 + 10)}-${Math.floor(Math.random() * 90 + 10)}`,
    date: new Date().toLocaleString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true }),
  };

  // Convert API detections to UI format
  const detections = apiResponse?.detections.map((det: Detection) => ({
    label: det.class,
    confidence: det.confidence,
    x: (det.center.x / (apiResponse?.imageSize.width || 1)) * 100,
    y: (det.center.y / (apiResponse?.imageSize.height || 1)) * 100,
    isHighRisk: det.riskLevel === 'high' || det.riskLevel === 'critical',
  })) || [];

  const detectionResults = apiResponse?.detections
    .filter((det: Detection) => det.confidence >= sensitivityThreshold)
    .map((det: Detection) => ({
      name: det.class + (det.riskLevel === 'high' || det.riskLevel === 'critical' ? ' (High Risk)' : ''),
      confidence: det.confidence,
      isHighRisk: det.riskLevel === 'high' || det.riskLevel === 'critical',
    })) || [];

  const handleImageSelected = useCallback((file: File, preview: string) => {
    setSelectedImage(preview);
    setSelectedFile(file);
    setUploadError(null);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setAnalysisStatus("pending");

    try {
      const response = await analyzeImage(selectedFile);
      setApiResponse(response);
      setAnnotatedImage(response.annotatedImage);
      setPolypDetected(response.totalDetections > 0 && 
        response.detections.some((det: Detection) => det.riskLevel === 'high' || det.riskLevel === 'critical'));
      setAnalysisStatus("completed");
      setViewMode("results");
      
      toast({
        title: "Analysis Complete",
        description: `Found ${response.totalDetections} detection(s)`,
      });
    } catch (error) {
      console.error('Analysis error:', error);
      setAnalysisStatus("error");
      toast({
        variant: "destructive",
        title: "Analysis Failed",
        description: error instanceof Error ? error.message : "Failed to analyze image",
      });
    } finally {
      setIsAnalyzing(false);
    }
  }, [selectedFile, toast]);

  const handleNewScan = useCallback(() => {
    setViewMode("upload");
    setSelectedImage(null);
    setSelectedFile(null);
    setUploadError(null);
    setApiResponse(null);
    setAnnotatedImage(null);
    setPolypDetected(false);
  }, []);

  const handlePrint = useCallback(() => {
    window.print();
  }, []);

  const handleReRun = useCallback(async () => {
    if (!selectedFile) return;
    
    setIsAnalyzing(true);
    setAnalysisStatus("pending");

    try {
      const response = await analyzeImage(selectedFile);
      setApiResponse(response);
      setAnnotatedImage(response.annotatedImage);
      setPolypDetected(response.totalDetections > 0 && 
        response.detections.some((det: Detection) => det.riskLevel === 'high' || det.riskLevel === 'critical'));
      setAnalysisStatus("completed");
      
      toast({
        title: "Re-analysis Complete",
        description: `Found ${response.totalDetections} detection(s)`,
      });
    } catch (error) {
      console.error('Re-analysis error:', error);
      setAnalysisStatus("error");
      toast({
        variant: "destructive",
        title: "Re-analysis Failed",
        description: error instanceof Error ? error.message : "Failed to re-analyze image",
      });
    } finally {
      setIsAnalyzing(false);
    }
  }, [selectedFile, toast]);

  const handleClearError = useCallback(() => {
    setUploadError(null);
  }, []);

  // Get highest confidence score
  const highestConfidence = apiResponse?.detections.reduce((max: number, det: Detection) => 
    Math.max(max, det.confidence), 0) || 0;

  // Get inference time from API response
  const inferenceTime = apiResponse ? `${apiResponse.processingTime.toFixed(1)}s` : "0.4s";

  return (
    <div className="min-h-screen bg-background flex">
      {/* Sidebar */}
      <Sidebar
        sensitivityThreshold={sensitivityThreshold}
        setSensitivityThreshold={setSensitivityThreshold}
        showProbability={showProbability}
        setShowProbability={setShowProbability}
        autoExport={autoExport}
        setAutoExport={setAutoExport}
      />

      {/* Main Content */}
      <main className="flex-1 lg:ml-72 min-h-screen">
        {/* Header */}
        <DashboardHeader
          caseId={caseData.caseId}
          patientId={caseData.patientId}
          date={caseData.date}
          status={analysisStatus}
          onNewScan={handleNewScan}
          onPrint={handlePrint}
          onReRun={handleReRun}
        />

        {/* Content based on view mode */}
        {viewMode === "upload" ? (
          <ImageUpload
            onImageSelected={handleImageSelected}
            onAnalyze={handleAnalyze}
            hasImage={!!selectedImage}
            isAnalyzing={isAnalyzing}
            uploadError={uploadError}
            onClearError={handleClearError}
          />
        ) : (
          <>
            {/* Image Comparison */}
            <ImageComparison
              originalImage={annotatedImage || selectedImage || colonoscopyImage}
              frameNumber={`${Math.floor(Math.random() * 10000).toString().padStart(5, '0')}`}
              inferenceTime={inferenceTime}
              detections={detections}
              polypDetected={polypDetected}
              showProbability={showProbability}
            />

            {/* Detection Results */}
            <DetectionResults
              polypDetected={polypDetected}
              confidenceScore={highestConfidence}
              description={
                polypDetected
                  ? "Immediate attention required. High confidence anomaly detected in sigmoid region."
                  : detectionResults.length > 0 
                    ? "Primary finding identified with high confidence based on texture and morphology."
                    : "No significant polyps detected. Continue monitoring."
              }
              results={detectionResults.length > 0 ? detectionResults : [
                { name: "No High-Risk Polyps Detected", confidence: 0.0 },
              ]}
              threshold={sensitivityThreshold}
            />

            {/* Export Panel */}
            <ExportPanel
              caseId={caseData.caseId}
              patientId={caseData.patientId}
              date={caseData.date}
              confidenceScore={highestConfidence}
              detections={detectionResults}
              polypDetected={polypDetected}
            />
          </>
        )}
      </main>
    </div>
  );
};

export default Index;
