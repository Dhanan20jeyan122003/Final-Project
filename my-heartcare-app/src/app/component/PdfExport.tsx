import React, { useState } from 'react';
import jsPDF from 'jspdf';

interface PdfExportProps {
  resultData: any;
  patientName?: string;
  patientAge?: string;
}

const PdfExport: React.FC<PdfExportProps> = ({ resultData, patientName = "Anonymous Patient", patientAge = "" }) => {
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Debug the received props
  console.log("PDF Export Props:", { patientName, patientAge });

  // Helper function to convert data URI to Image
  const dataUriToImageElement = async (dataUri: string): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = (e) => reject(e);
      // Setting crossOrigin attribute can help with some CORS issues
      img.crossOrigin = "Anonymous";
      img.src = dataUri;
    });
  };

  // Add this helper function to enhance image quality
  const enhanceImageQuality = async (dataUri: string): Promise<string> => {
    return new Promise((resolve, reject) => {
      try {
        const img = new Image();
        img.onload = () => {
          try {
            // Create a canvas with potentially slightly larger dimensions
            const canvas = document.createElement('canvas');
            // Make sure we have enough resolution
            canvas.width = Math.max(img.width, 800);
            canvas.height = Math.max(img.height, 600);
            
            const ctx = canvas.getContext('2d');
            if (!ctx) {
              resolve(dataUri); // Fall back to original if context creation fails
              return;
            }
            
            // Use better image rendering
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            
            // Draw the image onto the canvas
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            // Convert back to data URL with high quality
            const enhancedDataUri = canvas.toDataURL('image/png', 1.0);
            resolve(enhancedDataUri);
          } catch (err) {
            console.error("Error enhancing image:", err);
            resolve(dataUri); // Return original on error
          }
        };
        img.onerror = () => {
          console.error("Failed to load image for enhancement");
          resolve(dataUri); // Return original on error
        };
        img.src = dataUri;
      } catch (err) {
        console.error("Error in enhanceImageQuality:", err);
        resolve(dataUri); // Return original on error
      }
    });
  };

  const exportToPdf = async () => {
    try {
      setGenerating(true);
      setError(null);

      // Create a new PDF document
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 10;
      let yPos = margin;
      
      // Add header
      pdf.setTextColor(179, 39, 39); // Red header
      pdf.setFontSize(20);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Heart Health Analysis Report', pageWidth / 2, yPos, { align: 'center' });
      yPos += 10;
      
      // Ensure patientName has a value, using a fallback if it's empty
      const displayName = patientName && patientName.trim() !== "" 
        ? patientName 
        : "Anonymous Patient";
        
      // Add patient info
      pdf.setTextColor(0, 0, 0);
      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'normal');
      const patientInfo = `Patient: ${displayName}${patientAge ? ` | Age: ${patientAge}` : ''}`;
      pdf.text(patientInfo, pageWidth / 2, yPos, { align: 'center' });
      yPos += 5;
      pdf.text(`Date: ${new Date().toLocaleDateString()}`, pageWidth / 2, yPos, { align: 'center' });
      yPos += 10;

      // Add prediction section
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(14);
      pdf.setTextColor(179, 39, 39);
      pdf.text('Prediction Results', margin, yPos);
      yPos += 7;

      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(12);
      pdf.setTextColor(0, 0, 0);
      pdf.text(`Prediction: ${resultData?.prediction || "No prediction available"}`, margin, yPos);
      yPos += 6;
      pdf.text(`Confidence: ${(resultData.confidence * 100).toFixed(1)}%`, margin, yPos);
      yPos += 10;

      // Add model confidence scores
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(12);
      pdf.text('Model Confidence Scores:', margin, yPos);
      yPos += 6;

      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(10);
      
      // Clinical Model
      pdf.text(`• Clinical Model: ${(resultData.clinical * 100).toFixed(1)}%`, margin + 5, yPos);
      yPos += 5;
      
      // ECG Model
      if (resultData.ecg_analysis?.score !== undefined) {
        pdf.text(`• ECG Model: ${(resultData.ecg_analysis.score * 100).toFixed(1)}%`, margin + 5, yPos);
        yPos += 5;
      }
      
      // X-Ray Model
      if (resultData.xray_analysis?.confidence !== undefined) {
        pdf.text(`• X-Ray Model: ${(resultData.xray_analysis.confidence * 100).toFixed(1)}%`, margin + 5, yPos);
        yPos += 5;
      }
      
      // Echo Model
      if (resultData.echo_analysis?.score !== undefined) {
        pdf.text(`• Echo Model: ${(resultData.echo_analysis.score * 100).toFixed(1)}%`, margin + 5, yPos);
        yPos += 5;
      }
      
      yPos += 5;

      // Add imaging section with visualizations
      if (resultData.xray_analysis?.gradcam || resultData.echo_analysis?.gradcam) {
        pdf.setFont('helvetica', 'bold');
        pdf.setFontSize(14);
        pdf.setTextColor(179, 39, 39);
        pdf.text('Imaging Visualizations', margin, yPos);
        yPos += 7;

        pdf.setFont('helvetica', 'normal');
        pdf.setFontSize(10);
        pdf.setTextColor(0, 0, 0);

        // Add X-ray image if available
        if (resultData.xray_analysis?.gradcam) {
          try {
            pdf.text('X-Ray Analysis:', margin, yPos);
            yPos += 5;
            
            // Add a description of what the visualization shows
            if (resultData.xray_analysis?.abnormal_score > 70) {
              pdf.text('The image highlights areas of concern that require medical attention.', margin + 5, yPos);
            } else {
              pdf.text('The analyzed x-ray with potential areas of interest highlighted.', margin + 5, yPos);
            }
            yPos += 5;
            
            // Enhance image before adding to PDF
            const enhancedDataUri = await enhanceImageQuality(resultData.xray_analysis.gradcam);
            
            // Pre-load the enhanced image
            const img = await dataUriToImageElement(enhancedDataUri);
            const aspectRatio = img.height / img.width;
            
            // Use larger size for better quality - 85% of page width
            const imgWidth = (pageWidth - (margin * 2)) * 0.85;
            const imgHeight = imgWidth * aspectRatio;
            
            // Center the image horizontally
            const xPosition = margin + ((pageWidth - (margin * 2) - imgWidth) / 2);
            
            // Add image to PDF with maximum quality
            pdf.addImage(
              enhancedDataUri,
              'PNG',
              xPosition,
              yPos,
              imgWidth,
              imgHeight,
              undefined,
              'NONE'
            );
            
            // Update yPos to move below the image
            yPos += imgHeight + 10;
            
            // Check if we need a new page
            if (yPos > pageHeight - margin * 2) {
              pdf.addPage();
              yPos = margin;
            }
          } catch (e) {
            console.error("Failed to add X-ray image to PDF:", e);
            pdf.text('X-ray visualization could not be included.', margin + 5, yPos);
            yPos += 5;
          }
        }

        // Add Echo image if available
        if (resultData.echo_analysis?.gradcam) {
          try {
            // Check if we need to start on a new page based on remaining space
            // If less than 80mm of space left, start a new page
            if (pageHeight - yPos < 80) {
              pdf.addPage();
              yPos = margin;
            }
            
            pdf.text('Echocardiogram Analysis:', margin, yPos);
            yPos += 5;
            
            // Add a description of what the visualization shows
            if (resultData.echo_analysis?.score > 0.7) {
              pdf.text('The image highlights cardiac areas that may indicate abnormal function.', margin + 5, yPos);
            } else {
              pdf.text('The analyzed echocardiogram with regions of interest highlighted.', margin + 5, yPos);
            }
            yPos += 5;
            
            // Enhance image before adding to PDF
            const enhancedDataUri = await enhanceImageQuality(resultData.echo_analysis.gradcam);
            
            // Pre-load the enhanced image
            const img = await dataUriToImageElement(enhancedDataUri);
            const aspectRatio = img.height / img.width;
            
            // Use a more reasonable size - 70% of page width for better display
            const imgWidth = (pageWidth - (margin * 2)) * 0.7;
            const imgHeight = imgWidth * aspectRatio;
            
            // Calculate if the image will fit on the current page
            if (yPos + imgHeight > pageHeight - margin * 2) {
              // Not enough space, add a new page
              pdf.addPage();
              yPos = margin + 10; // Start a bit lower on new page
              
              // Add a header on the new page
              pdf.setFont('helvetica', 'bold');
              pdf.setFontSize(12);
              pdf.setTextColor(179, 39, 39);
              pdf.text('Echocardiogram Analysis (Continued)', margin, yPos);
              yPos += 10;
              
              pdf.setFont('helvetica', 'normal');
              pdf.setFontSize(10);
              pdf.setTextColor(0, 0, 0);
            }
            
            // Center the image horizontally
            const xPosition = margin + ((pageWidth - (margin * 2) - imgWidth) / 2);
            
            // Add image with standard quality to reduce PDF size while maintaining readability
            pdf.addImage(
              enhancedDataUri,
              'PNG',
              xPosition,
              yPos,
              imgWidth,
              imgHeight,
              undefined,
              'MEDIUM' // Use 'MEDIUM' for better file size vs quality balance
            );
            
            // Update yPos to move below the image
            yPos += imgHeight + 10;
          } catch (e) {
            console.error("Failed to add Echo image to PDF:", e);
            pdf.text('Echocardiogram visualization could not be included.', margin + 5, yPos);
            yPos += 5;
          }
        }
      }

      // Add explanations section
      if (resultData.explanations && resultData.explanations.length > 0) {
        pdf.setFont('helvetica', 'bold');
        pdf.setFontSize(14);
        pdf.setTextColor(179, 39, 39);
        pdf.text('Analysis Explanations', margin, yPos);
        yPos += 7;
        
        pdf.setFont('helvetica', 'normal');
        pdf.setFontSize(10);
        pdf.setTextColor(0, 0, 0);
        
        for (const explanation of resultData.explanations) {
          // Split long text into multiple lines to fit page width
          const maxCharsPerLine = 90;
          
          if (explanation.length <= maxCharsPerLine) {
            pdf.text(`• ${explanation}`, margin, yPos);
            yPos += 5;
          } else {
            // Split long explanation into multiple lines
            const words = explanation.split(' ');
            let currentLine = '• ';
            
            for (const word of words) {
              if ((currentLine + word).length <= maxCharsPerLine) {
                currentLine += word + ' ';
              } else {
                pdf.text(currentLine, margin, yPos);
                yPos += 5;
                currentLine = '  ' + word + ' '; // indent continuation lines
              }
            }
            
            if (currentLine.trim().length > 0) {
              pdf.text(currentLine, margin, yPos);
              yPos += 5;
            }
          }
          
          // Check if we need a new page
          if (yPos > pageHeight - margin) {
            pdf.addPage();
            yPos = margin;
          }
        }
      }
      
      yPos += 5;

      // Add imaging findings section if available
      if (resultData.xray_analysis?.explanations || resultData.echo_analysis?.explanations) {
        pdf.setFont('helvetica', 'bold');
        pdf.setFontSize(14);
        pdf.setTextColor(179, 39, 39);
        pdf.text('Detailed Imaging Findings', margin, yPos);
        yPos += 7;
        
        pdf.setFont('helvetica', 'normal');
        pdf.setFontSize(10);
        pdf.setTextColor(0, 0, 0);
        
        // X-ray findings
        if (resultData.xray_analysis?.explanations) {
          pdf.setFont('helvetica', 'bold');
          pdf.text('X-ray Analysis:', margin, yPos);
          yPos += 5;
          
          pdf.setFont('helvetica', 'normal');
          for (const exp of resultData.xray_analysis.explanations) {
            pdf.text(`• ${exp}`, margin + 5, yPos);
            yPos += 5;
            
            // Check if we need a new page
            if (yPos > pageHeight - margin) {
              pdf.addPage();
              yPos = margin;
            }
          }
          
          yPos += 2;
        }
        
        // Echo findings
        if (resultData.echo_analysis?.explanations) {
          pdf.setFont('helvetica', 'bold');
          pdf.text('Echo Analysis:', margin, yPos);
          yPos += 5;
          
          pdf.setFont('helvetica', 'normal');
          for (const exp of resultData.echo_analysis.explanations) {
            pdf.text(`• ${exp}`, margin + 5, yPos);
            yPos += 5;
            
            // Check if we need a new page
            if (yPos > pageHeight - margin) {
              pdf.addPage();
              yPos = margin;
            }
          }
          
          yPos += 2;
        }
      }

      // Add recommendations section
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(14);
      pdf.setTextColor(179, 39, 39);
      pdf.text('Recommendations', margin, yPos);
      yPos += 7;
      
      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(10);
      pdf.setTextColor(0, 0, 0);
      
      // Determine if a clinical consultation is recommended
      const hasHeartDisease = resultData.prediction === "Heart Disease";
      const hasImagingEmergency = resultData.imaging_emergency || 
        resultData.xray_analysis?.needs_attention || 
        resultData.echo_analysis?.needs_attention;
      
      if (hasHeartDisease || hasImagingEmergency) {
        pdf.setTextColor(179, 39, 39);
        pdf.text('• Based on your results, consulting with a doctor is recommended.', margin, yPos);
        yPos += 5;
        
        if (hasImagingEmergency) {
          pdf.text('• Your imaging studies show findings that require medical attention.', margin, yPos);
          yPos += 5;
        }
        
        pdf.setTextColor(0, 0, 0);
      } else {
        pdf.setTextColor(0, 100, 0);
        pdf.text('• Your results appear to be within normal limits.', margin, yPos);
        yPos += 5;
        pdf.setTextColor(0, 0, 0);
        pdf.text('• Continue with regular health check-ups.', margin, yPos);
        yPos += 5;
      }
      
      pdf.text('• Maintain a heart-healthy lifestyle with regular exercise and balanced diet.', margin, yPos);
      yPos += 5;
      pdf.text('• Monitor your blood pressure and cholesterol levels regularly.', margin, yPos);
      yPos += 15;

      // Add disclaimer
      pdf.setFontSize(8);
      pdf.setTextColor(100, 100, 100);
      const disclaimer = 'DISCLAIMER: This report was generated by an AI system and is not a substitute for professional medical advice.';
      pdf.text(disclaimer, pageWidth / 2, pageHeight - margin, { align: 'center' });
      
      // Save the PDF with formatted filename
      const cleanName = displayName.replace(/[^a-zA-Z0-9]/g, '_');
      const fileName = `Heart_Analysis_Report_${cleanName}_${new Date().toISOString().split('T')[0]}.pdf`;
      pdf.save(fileName);
      
      setGenerating(false);
    } catch (error) {
      console.error('Error generating PDF:', error);
      setError('Failed to generate PDF. Please try again.');
      setGenerating(false);
    }
  };

  return (
    <div className="mt-4 text-center">
      <button
        onClick={exportToPdf}
        disabled={generating}
        className="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition-all duration-200 flex items-center mx-auto disabled:bg-red-400 disabled:cursor-not-allowed"
      >
        {generating ? (
          <>
            <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Generating PDF...
          </>
        ) : (
          <>
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
            </svg>
            Export to PDF
          </>
        )}
      </button>
      
      {error && (
        <p className="text-red-500 text-sm mt-2">{error}</p>
      )}
      
      <p className="text-xs text-gray-500 mt-1">Save this report for your records or to share with your healthcare provider</p>
    </div>
  );
};

export default PdfExport;