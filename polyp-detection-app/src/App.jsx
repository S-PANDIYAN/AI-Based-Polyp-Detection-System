import React, { useState, useRef } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { Bar } from 'react-chartjs-2';
import jsPDF from 'jspdf';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [fileName, setFileName] = useState('');
  const [fileSize, setFileSize] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [threshold, setThreshold] = useState(50);
  const fileInputRef = useRef(null);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFileName(file.name);
      setFileSize((file.size / 1024).toFixed(2) + ' KB');
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
      setResult(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const fakeEvent = { target: { files: [file] } };
      handleImageUpload(fakeEvent);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const analyzeImage = () => {
    if (!selectedImage) return;
    
    setAnalyzing(true);
    
    setTimeout(() => {
      // Mock AI prediction
      const polypConfidence = Math.random() * 100;
      const normalConfidence = 100 - polypConfidence;
      const isPolyp = polypConfidence > threshold;
      
      setResult({
        prediction: isPolyp ? 'Polyp Detected' : 'Normal',
        polypScore: polypConfidence.toFixed(2),
        normalScore: normalConfidence.toFixed(2),
        isPolyp: isPolyp,
        timestamp: new Date().toISOString(),
        confidenceLevel: polypConfidence > 90 ? 'Very High' : polypConfidence > 70 ? 'High' : polypConfidence > 50 ? 'Medium' : 'Low'
      });
      
      setAnalyzing(false);
    }, 2000);
  };

  const downloadAnalysis = () => {
    if (!result) return;
    
    const analysisData = {
      imageName: fileName,
      prediction: result.prediction,
      polypScore: result.polypScore + '%',
      normalScore: result.normalScore + '%',
      threshold: threshold + '%',
      confidenceLevel: result.confidenceLevel,
      timestamp: result.timestamp,
      modelInfo: {
        architecture: 'Custom CNN (4 blocks)',
        parameters: '~14M',
        framework: 'PyTorch 2.x',
        trainingData: {
          normal: '~700 images',
          polyp: '~880 images'
        }
      }
    };
    
    const dataStr = JSON.stringify(analysisData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `polyp_analysis_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const downloadReport = () => {
    if (!result || !imagePreview) return;
    
    const doc = new jsPDF();
    
    // Header
    doc.setFillColor(26, 32, 44);
    doc.rect(0, 0, 210, 40, 'F');
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(24);
    doc.text('AI Polyp Detection Report', 105, 20, { align: 'center' });
    doc.setFontSize(12);
    doc.text('Advanced Deep Learning for Colonoscopy Analysis', 105, 30, { align: 'center' });
    
    // Reset text color
    doc.setTextColor(0, 0, 0);
    
    // Patient/Image Info
    doc.setFontSize(14);
    doc.setFont(undefined, 'bold');
    doc.text('Image Information', 20, 55);
    doc.setFont(undefined, 'normal');
    doc.setFontSize(11);
    doc.text(`File Name: ${fileName}`, 20, 65);
    doc.text(`Analysis Date: ${new Date().toLocaleString()}`, 20, 72);
    
    // Prediction Results
    doc.setFontSize(14);
    doc.setFont(undefined, 'bold');
    doc.text('Analysis Results', 20, 90);
    doc.setFont(undefined, 'normal');
    doc.setFontSize(11);
    
    if (result.isPolyp) {
      doc.setTextColor(220, 38, 38);
      doc.text(`Prediction: ${result.prediction}`, 20, 100);
    } else {
      doc.setTextColor(34, 197, 94);
      doc.text(`Prediction: ${result.prediction}`, 20, 100);
    }
    
    doc.setTextColor(0, 0, 0);
    doc.text(`Polyp Confidence: ${result.polypScore}%`, 20, 110);
    doc.text(`Normal Confidence: ${result.normalScore}%`, 20, 117);
    doc.text(`Confidence Level: ${result.confidenceLevel}`, 20, 124);
    doc.text(`Detection Threshold: ${threshold}%`, 20, 131);
    
    // Model Information
    doc.setFontSize(14);
    doc.setFont(undefined, 'bold');
    doc.text('Model Information', 20, 150);
    doc.setFont(undefined, 'normal');
    doc.setFontSize(11);
    doc.text('Architecture: Custom CNN (4 blocks)', 20, 160);
    doc.text('Parameters: ~14M', 20, 167);
    doc.text('Framework: PyTorch 2.x', 20, 174);
    doc.text('Training Data: Normal (~700 images), Polyp (~880 images)', 20, 181);
    
    // Add image
    try {
      doc.addImage(imagePreview, 'JPEG', 20, 195, 80, 60);
    } catch (error) {
      console.error('Error adding image to PDF:', error);
    }
    
    // Disclaimer
    doc.setFillColor(255, 243, 224);
    doc.rect(15, 265, 180, 15, 'F');
    doc.setFontSize(9);
    doc.setTextColor(120, 53, 15);
    doc.text('DISCLAIMER: This AI system is for research and decision support only.', 20, 272);
    doc.text('Final diagnosis must be confirmed by a qualified medical professional.', 20, 277);
    
    doc.save(`polyp_report_${Date.now()}.pdf`);
  };

  const setPreset = (value) => {
    setThreshold(value);
  };

  const chartData = result ? {
    labels: ['Polyp', 'Normal'],
    datasets: [{
      label: 'Confidence Score',
      data: [parseFloat(result.polypScore), parseFloat(result.normalScore)],
      backgroundColor: [
        result.isPolyp ? 'rgba(220, 38, 38, 0.8)' : 'rgba(220, 38, 38, 0.3)',
        !result.isPolyp ? 'rgba(34, 197, 94, 0.8)' : 'rgba(34, 197, 94, 0.3)'
      ],
      borderColor: [
        'rgba(220, 38, 38, 1)',
        'rgba(34, 197, 94, 1)'
      ],
      borderWidth: 2
    }]
  } : null;

  const chartOptions = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: false
      }
    },
    scales: {
      x: {
        beginAtZero: true,
        max: 100,
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.7)',
          callback: function(value) {
            return value + '%';
          }
        }
      },
      y: {
        grid: {
          display: false
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.7)',
          font: {
            size: 14,
            weight: 'bold'
          }
        }
      }
    }
  };

  return (
    <div className="app">
      {/* Left Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-content">
          {/* Logo */}
          <div className="logo">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <h2>AI Polyp Detection</h2>
          </div>

          {/* Settings Section */}
          <div className="settings-section">
            <h3>Settings</h3>
            
            <div className="setting-group">
              <label>Detection Threshold</label>
              <div className="threshold-display">{threshold}%</div>
              <input 
                type="range" 
                min="0" 
                max="100" 
                value={threshold}
                onChange={(e) => setThreshold(parseInt(e.target.value))}
                className="threshold-slider"
              />
              
              <div className="presets">
                <button onClick={() => setPreset(50)} className={threshold === 50 ? 'active' : ''}>
                  50% Balanced
                </button>
                <button onClick={() => setPreset(70)} className={threshold === 70 ? 'active' : ''}>
                  70% Conservative
                </button>
                <button onClick={() => setPreset(90)} className={threshold === 90 ? 'active' : ''}>
                  90% High Specificity
                </button>
              </div>
            </div>
          </div>

          {/* Model Information */}
          <div className="model-info">
            <h3>Model Information</h3>
            <div className="info-item">
              <span className="label">Architecture:</span>
              <span className="value">Custom CNN (4 blocks)</span>
            </div>
            <div className="info-item">
              <span className="label">Parameters:</span>
              <span className="value">~14M</span>
            </div>
            <div className="info-item">
              <span className="label">Framework:</span>
              <span className="value">PyTorch 2.x</span>
            </div>
            
            <h4>Training Data</h4>
            <div className="info-item">
              <span className="label">Normal:</span>
              <span className="value">~700 images</span>
            </div>
            <div className="info-item">
              <span className="label">Polyp:</span>
              <span className="value">~880 images</span>
            </div>
            
            <h4>Anti-Overfitting</h4>
            <ul className="feature-list">
              <li>Dropout</li>
              <li>L2 Regularization</li>
              <li>Data Augmentation</li>
              <li>Early Stopping</li>
            </ul>
            
            <div className="status-badges">
              <span className="badge success">Model Loaded</span>
              <span className="badge info">Device: CPU</span>
              <span className="badge secondary">Last Trained: 2025-12-15</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        {/* Header */}
        <header className="header">
          <div>
            <h1>AI Polyp Detection System</h1>
            <p className="subtitle">Advanced Deep Learning for Colonoscopy Image Analysis</p>
          </div>
        </header>

        {/* Upload Section */}
        <div className="upload-section">
          <div 
            className="upload-area"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => fileInputRef.current?.click()}
          >
            {!imagePreview ? (
              <>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p>Drag & drop colonoscopy image here</p>
                <button className="browse-btn">Browse Files</button>
              </>
            ) : (
              <div className="image-preview">
                <img src={imagePreview} alt="Preview" />
                <div className="file-info">
                  <p className="file-name">{fileName}</p>
                  <p className="file-size">{fileSize}</p>
                </div>
              </div>
            )}
          </div>
          <input 
            ref={fileInputRef}
            type="file" 
            accept="image/*" 
            onChange={handleImageUpload}
            style={{ display: 'none' }}
          />
          
          {imagePreview && (
            <button 
              className="analyze-btn" 
              onClick={analyzeImage}
              disabled={analyzing}
            >
              {analyzing ? (
                <>
                  <span className="spinner"></span>
                  Analyzing...
                </>
              ) : (
                'Analyze Image'
              )}
            </button>
          )}
        </div>

        {/* Results Section */}
        {result && (
          <>
            {/* Result Banner */}
            <div className={`result-banner ${result.isPolyp ? 'danger' : 'success'}`}>
              <div className="banner-content">
                <div className="banner-icon">
                  {result.isPolyp ? (
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                  ) : (
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  )}
                </div>
                <div className="banner-text">
                  <h3>{result.prediction}</h3>
                  <p>Confidence: {result.polypScore}%</p>
                  <p className="advice">
                    {result.isPolyp 
                      ? 'Polyp detected. Recommend immediate medical consultation and biopsy for further analysis.'
                      : 'No polyp detected. Continue regular screening as per medical guidelines.'}
                  </p>
                </div>
              </div>
            </div>

            {/* Confidence Analysis */}
            <div className="analysis-panel">
              <h3>Confidence Analysis</h3>
              <div className="confidence-bars">
                {chartData && (
                  <div style={{ height: '150px' }}>
                    <Bar data={chartData} options={chartOptions} />
                  </div>
                )}
                <div className="confidence-details">
                  <div className="confidence-item">
                    <span className="label">Polyp Score:</span>
                    <span className="score polyp">{result.polypScore}%</span>
                  </div>
                  <div className="confidence-item">
                    <span className="label">Normal Score:</span>
                    <span className="score normal">{result.normalScore}%</span>
                  </div>
                  <div className="confidence-item">
                    <span className="label">Threshold:</span>
                    <span className="score">{threshold}%</span>
                  </div>
                  <div className="confidence-item">
                    <span className="label">Confidence Level:</span>
                    <span className={`badge ${result.confidenceLevel.toLowerCase().replace(' ', '-')}`}>
                      {result.confidenceLevel}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Visual Analysis */}
            <div className="visual-analysis">
              <h3>Visual Analysis</h3>
              <div className="image-comparison">
                <div className="comparison-card">
                  <div className="card-header">
                    <h4>Original Image</h4>
                    <span className="status-label">Uploaded</span>
                  </div>
                  <div className="card-image">
                    <img src={imagePreview} alt="Original" />
                  </div>
                </div>
                
                <div className="comparison-card">
                  <div className="card-header">
                    <h4>AI Result Image</h4>
                    <span className={`status-label ${result.isPolyp ? 'detected' : 'normal'}`}>
                      {result.isPolyp ? 'Polyp Detected' : 'Normal'}
                    </span>
                  </div>
                  <div className="card-image">
                    <img src={imagePreview} alt="Result" />
                    {result.isPolyp && (
                      <div className="bounding-box"></div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Export Results */}
            <div className="export-section">
              <h3>Export Results</h3>
              <div className="export-buttons">
                <button className="export-btn" onClick={downloadAnalysis}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download Analysis (JSON)
                </button>
                <button className="export-btn" onClick={downloadReport}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Download Report (PDF)
                </button>
              </div>
            </div>
          </>
        )}

        {/* Medical Disclaimer */}
        <div className="disclaimer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p>
            <strong>MEDICAL DISCLAIMER:</strong> This AI system is for research and decision support only.
            Final diagnosis must be confirmed by a qualified medical professional.
          </p>
        </div>
      </main>
    </div>
  );
}

export default App;
