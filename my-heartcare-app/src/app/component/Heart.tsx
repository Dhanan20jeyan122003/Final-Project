"use client";
import React, { useState, useEffect } from "react";

// Define types for form data and response
interface FormData {
    age: string;
    sex: string;
    cp: string;
    trestbps: string;
    chol: string;
    fbs: string;
    restecg: string;
    thalach: string;
    exang: string;
    oldpeak: string;
    slope: string;
    ca: string;
    thal: string;
}

interface FileData {
    ecgImage: File | null;
    xrayImage: File | null;
    echoVideo: File | null;
}

interface Previews {
    ecgImage: string;
    xrayImage: string;
    echoVideo: string;
}

interface ResultData {
    age: number;
    prediction: string;
    confidence: number;
    clinical: number;
    explanations: string[];
    imaging_emergency?: boolean; // Add this line
    ecg_analysis?: {
        score: number;
        explanations?: string[];
        gradcam?: string;
    } | null;
    xray_analysis?: {
        label?: string;
        confidence?: number;
        affected_percentage?: number;
        normal_score?: number;
        abnormal_score?: number;
        needs_attention?: boolean;
        explanations?: string[];
        gradcam?: string;
        doctor_recommendation?: string;
    } | null;
    echo_analysis?: {
        score: number;
        explanations?: string[];
        gradcam?: string;
        doctor_recommendation?: string;
        needs_attention?: boolean;
    } | null;
}

// Initialize empty form data
const initialFormData: FormData = {
    age: "",
    sex: "",
    cp: "",
    trestbps: "",
    chol: "",
    fbs: "",
    restecg: "",
    thalach: "",
    exang: "",
    oldpeak: "",
    slope: "",
    ca: "",
    thal: "",
};

// Initialize empty file data
const initialFileData: FileData = {
    ecgImage: null,
    xrayImage: null,
    echoVideo: null,
};

// Initialize empty previews
const initialPreviews: Previews = {
    ecgImage: "",
    xrayImage: "",
    echoVideo: "",
};

export default function Heart() {
    const [mounted, setMounted] = useState(false);
    const [resultData, setResultData] = useState<ResultData | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [formData, setFormData] = useState<FormData>(initialFormData);
    const [files, setFiles] = useState<FileData>(initialFileData);
    const [previews, setPreviews] = useState<Previews>(initialPreviews);

    useEffect(() => {
        setMounted(true);
    }, []);

    useEffect(() => {
        return () => {
            Object.values(previews).forEach((url) => {
                if (url) URL.revokeObjectURL(url);
            });
        };
    }, [previews]);

    // Show a loading screen until the component is hydrated on the client
    if (!mounted) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-pink-100 to-red-100">
                Loading...
            </div>
        );
    }

    // Maps frontend form values to backend expected values
    const mapFormValuesToBackend = (data: FormData) => {
        const mapValue = (value: string, mapping: Record<string, string>): string => {
            return mapping[value] || value;
        };

        // Define mappings
        const sexMap: Record<string, string> = {
            Male: "1",
            Female: "0",
        };

        const cpMap: Record<string, string> = {
            "Typical Angina": "0",
            "Atypical Angina": "1",
            "Non-anginal Pain": "2",
            Asymptomatic: "3",
        };

        const restecgMap: Record<string, string> = {
            Normal: "0",
            "ST-T Wave Abnormality": "1",
            "Left Ventricular Hypertrophy": "2",
        };

        const fbsMap: Record<string, string> = {
            True: "1",
            False: "0",
        };

        const exangMap: Record<string, string> = {
            Yes: "1",
            No: "0",
        };

        const slopeMap: Record<string, string> = {
            Upsloping: "0",
            Flat: "1",
            Downsloping: "2",
        };

        // Update thal mapping
        const thalMap: Record<string, string> = {
            Normal: "1",
            "Fixed Defect": "2",
            "Reversible Defect": "3"
        };

        // Ensure thal is always 1, 2, or 3
        let thalValue = "1"; // Default to Normal
        if (data.thal in thalMap) {
            thalValue = thalMap[data.thal];
        }

        return {
            age: data.age,
            sex: mapValue(data.sex, sexMap),
            cp: mapValue(data.cp, cpMap),
            trestbps: data.trestbps,
            chol: data.chol,
            fbs: mapValue(data.fbs, fbsMap),
            restecg: mapValue(data.restecg, restecgMap),
            thalach: data.thalach,
            exang: mapValue(data.exang, exangMap),
            oldpeak: data.oldpeak,
            slope: mapValue(data.slope, slopeMap),
            ca: data.ca,
            thal: thalValue,
        };
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value } = e.target;
        setFormData((prev) => ({ ...prev, [name]: value }));
    };

    // Add this helper function to check video format support
    function isVideoFormatSupported(mimeType: string): boolean {
        // Create a test video element
        const video = document.createElement('video');
        
        // Check if the browser can play this video type
        const canPlay = video.canPlayType(mimeType);
        
        // canPlayType returns "", "maybe" or "probably"
        return canPlay !== '';
    }

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, files: selected } = e.target;
        if (selected && selected[0]) {
            // Clean up previous preview URL for this file type
            if (previews[name as keyof Previews]) {
                URL.revokeObjectURL(previews[name as keyof Previews]);
            }

            // Update the file state
            setFiles((prev) => ({ ...prev, [name]: selected[0] }));

            // Create and set preview URL
            const fileUrl = URL.createObjectURL(selected[0]);
            console.log(`Created preview URL for ${name}:`, fileUrl);
            console.log(`File type: ${selected[0].type}`);
            
            // For debugging
            if (name === "echoVideo") {
                console.log("Echo Video file:", selected[0]);
                console.log("Echo Video size:", selected[0].size);
            }

            // Check video format support for echoVideo
            if (name === "echoVideo" && selected[0]) {
                const isSupported = isVideoFormatSupported(selected[0].type);
                console.log(`Video format ${selected[0].type} supported: ${isSupported}`);
                
                // If not supported, provide a warning
                if (!isSupported) {
                    setError(`Warning: Your browser might not support the video format ${selected[0].type}. Please use MP4 format if possible.`);
                }
            }
            
            setPreviews((prev) => ({ ...prev, [name]: fileUrl }));
        }
    };

    const resetForm = () => {
        setFormData(initialFormData);

        // Clean up existing preview URLs before resetting
        Object.values(previews).forEach(url => {
            if (url) URL.revokeObjectURL(url);
        });

        setFiles(initialFileData);
        setPreviews(initialPreviews);
    };

    const validateForm = (): boolean => {
        // Check if any required form field is empty
        const missingFields = Object.entries(formData)
            .filter(([_, value]) => value === "")
            .map(([key, _]) => key);

        if (missingFields.length > 0) {
            setError(`Please fill in all clinical fields before submitting. Missing: ${missingFields.join(", ")}.`);
            return false;
        }

        return true;
    };

    // Add form validation
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);
        setLoading(true);

        try {
            if (!validateForm()) {
                setLoading(false);
                return;
            }

            const form = new FormData();

            // Map categorical values to their numeric equivalents
            const sexMap = { 'Male': '1', 'Female': '0' };
            const chestPainMap = {
                'Typical Angina': '0',
                'Atypical Angina': '1',
                'Non-anginal Pain': '2',
                'Asymptomatic': '3'
            };
            const fbsMap = { 'True': '1', 'False': '0' };
            const restECGMap = {
                'Normal': '0',
                'ST-T Wave Abnormality': '1',
                'Left Ventricular Hypertrophy': '2'
            };
            const exangMap = { 'Yes': '1', 'No': '0' };
            const slopeMap = {
                'Upsloping': '0',
                'Flat': '1',
                'Downsloping': '2'
            };
            const thalMap = {
                'Normal': '1',
                'Fixed Defect': '2',
                'Reversible Defect': '3'
            };

            // Add clinical data with proper string conversion
            form.append("age", formData.age);
            form.append("sex", sexMap[formData.sex as keyof typeof sexMap]);
            form.append("chestPainType", chestPainMap[formData.cp as keyof typeof chestPainMap]);
            form.append("restingBP", formData.trestbps);
            form.append("cholesterol", formData.chol);
            form.append("fbs", fbsMap[formData.fbs as keyof typeof fbsMap]);
            form.append("restECG", restECGMap[formData.restecg as keyof typeof restECGMap]);
            form.append("maxHR", formData.thalach);
            form.append("exerciseAngina", exangMap[formData.exang as keyof typeof exangMap]);
            form.append("stDepression", formData.oldpeak);
            form.append("stSlope", slopeMap[formData.slope as keyof typeof slopeMap]);
            form.append("numVessels", formData.ca);
            form.append("thal", thalMap[formData.thal as keyof typeof thalMap]);

            // Add files if they exist
            if (files.xrayImage) {
                form.append("xrayImage", files.xrayImage);
            }
            if (files.ecgImage) {
                form.append("ecgImage", files.ecgImage);
            }
            if (files.echoVideo) {
                form.append("echoVideo", files.echoVideo);
            }

            const response = await fetch("http://localhost:8000/predict", {
                method: "POST",
                body: form
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Failed to get prediction");
            }

            const result = await response.json();
            setResultData(result);
            resetForm();

        } catch (error: any) {
            console.error("Submission error:", error);
            setError(error.message || "Failed to submit form. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-pink-100 to-red-100 py-8 px-4 space-y-6">
            <form onSubmit={handleSubmit} className="bg-white p-6 rounded-xl shadow-lg w-full max-w-5xl mx-auto">
                <h2 className="text-2xl font-bold text-red-500 text-center mb-4">Heart Disease Risk Assessment</h2>

                {/* Clinical Parameters Section */}
                <div className="mb-6">
                    <h3 className="text-lg font-semibold text-red-500 mb-4">Clinical Parameters</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <Input
                            label="Age"
                            name="age"
                            type="number"
                            value={formData.age}
                            onChange={handleChange}
                            loading={loading}
                        />
                        <Select
                            label="Sex"
                            name="sex"
                            value={formData.sex}
                            options={["Male", "Female"]}
                            onChange={handleChange}
                            loading={loading}
                        />
                        <Select
                            label="Chest Pain Type"
                            name="cp"
                            value={formData.cp}
                            options={["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]}
                            onChange={handleChange}
                            loading={loading}
                        />
                        <Input label="Resting BP (mmHg)" name="trestbps" type="number" value={formData.trestbps} onChange={handleChange} loading={loading} />
                        <Input label="Cholesterol (mg/dl)" name="chol" type="number" value={formData.chol} onChange={handleChange} loading={loading} />
                        <Select
                            label="Fasting Blood Sugar > 120 mg/dl"
                            name="fbs"
                            value={formData.fbs}
                            options={["True", "False"]}
                            onChange={handleChange}
                            loading={loading}
                        />
                        <Select
                            label="Resting ECG Results"
                            name="restecg"
                            value={formData.restecg}
                            options={["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]}
                            onChange={handleChange}
                            loading={loading}
                        />
                        <Input label="Max Heart Rate" name="thalach" type="number" value={formData.thalach} onChange={handleChange} loading={loading} />
                        <Select
                            label="Exercise-Induced Angina"
                            name="exang"
                            value={formData.exang}
                            options={["Yes", "No"]}
                            onChange={handleChange}
                            loading={loading}
                        />
                        <Input
                            label="ST Depression Induced by Exercise"
                            name="oldpeak"
                            type="number"
                            step="0.1"
                            value={formData.oldpeak}
                            onChange={handleChange}
                            loading={loading}
                        />
                        <Select
                            label="ST Slope"
                            name="slope"
                            value={formData.slope}
                            options={["Upsloping", "Flat", "Downsloping"]}
                            onChange={handleChange}
                            loading={loading}
                        />
                        <Select
                            label="Number of Major Vessels"
                            name="ca"
                            value={formData.ca}
                            options={["0", "1", "2", "3"]}
                            onChange={handleChange}
                            loading={loading}
                        />
                        <Select
                            label="Thalassemia"
                            name="thal"
                            value={formData.thal}
                            options={["Normal", "Fixed Defect", "Reversible Defect"]}
                            onChange={handleChange}
                            loading={loading}
                        />
                    </div>
                </div>

                {/* Medical Images Section */}
                <div className="bg-gray-50 p-6 rounded-xl border border-red-200 mb-6">
                    <h3 className="text-lg font-semibold text-red-500 mb-2">Upload Medical Images</h3>
                    <p className="text-gray-600 text-sm mb-4">Optional: Upload medical images to improve prediction accuracy</p>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <FileUpload
                            label="Upload ECG Image"
                            name="ecgImage"
                            accept="image/*"
                            onChange={handleFileUpload}
                            preview={previews.ecgImage}
                            previewType="image"
                            loading={loading}
                        />

                        <FileUpload
                            label="Upload Chest X-ray"
                            name="xrayImage"
                            accept="image/*"
                            onChange={handleFileUpload}
                            preview={previews.xrayImage}
                            previewType="image"
                        />

                        <FileUpload
                            label="Upload Echo Video"
                            name="echoVideo"
                            accept="video/*"
                            onChange={handleFileUpload}
                            preview={previews.echoVideo}
                            previewType="video"
                        />
                    </div>
                </div>

                {/* Submit Button */}
                <button
                    type="submit"
                    disabled={loading}
                    className={`w-full font-bold py-3 px-4 rounded-xl transition-all duration-200 ${
                        loading
                            ? "bg-gray-400 cursor-not-allowed flex items-center justify-center gap-2"
                            : "bg-red-600 hover:bg-red-700 text-white"
                    }`}
                >
                    {loading ? (
                        <>
                            <svg
                                className="animate-spin h-5 w-5 text-white"
                                xmlns="http://www.w3.org/2000/svg"
                                fill="none"
                                viewBox="0 0 24 24"
                            >
                                <circle
                                    className="opacity-25"
                                    cx="12"
                                    cy="12"
                                    r="10"
                                    stroke="currentColor"
                                    strokeWidth="4"
                                />
                                <path
                                    className="opacity-75"
                                    fill="currentColor"
                                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                />
                            </svg>
                            Processing...
                        </>
                    ) : (
                        "Submit Data for Analysis"
                    )}
                </button>

                {/* Error Display */}
                {error && (
                    <div className="bg-red-100 border border-red-400 text-red-800 font-semibold text-center p-4 rounded-xl mt-6 shadow-md">
                        ❌ {error}
                    </div>
                )}

                {/* Results Display */}
                {resultData && (
                    <div className="bg-white border border-red-300 p-6 rounded-xl mt-6 shadow-lg">
                        <h3 className="font-bold text-red-600 text-2xl mb-4">Analysis Results</h3>
                        
                        {/* Overall Prediction */}
                        <div className="bg-red-50 p-4 rounded-lg mb-4">
                            <div className="text-xl font-bold text-red-700">
                                Prediction: {resultData?.prediction || "No prediction available"}
                            </div>
                            <div className="text-lg font-medium text-red-600">
                                Overall Confidence: {(resultData.confidence * 100).toFixed(1)}%
                            </div>
                        </div>
                        
                        {/* Add the ModelConfidenceScores component */}
                        <ModelConfidenceScores resultData={resultData} />

                        {/* Add the updated Health Recommendations component */}
                        <HealthRecommendations resultData={resultData} />

                        {/* Visualization Display Section - Only show when abnormal */}
                        {resultData && (
                            <ImageAnalysisSection resultData={resultData} />
                        )}

                        {/* Add a message for normal imaging results */}
                        {resultData && 
                         !resultData.xray_analysis?.needs_attention && 
                         !resultData.echo_analysis?.needs_attention && 
                         (resultData.xray_analysis?.gradcam || resultData.echo_analysis?.gradcam) && (
                            <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                                <div className="flex items-center text-green-700">
                                    <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                                    </svg>
                                    <span className="font-medium">Good news! No areas of concern were detected in your imaging studies.</span>
                                </div>
                            </div>
                        )}

                        {/* Rest of your results display... */}
                    </div>
                )}
            </form>

            {/* Add this component to display explanations */}
            {resultData && (
                <ExplanationsSection resultData={resultData} />
            )}
        </div>
    );
}

interface InputProps {
    label: string;
    name: string;
    type: string;
    value: string | number;
    step?: string;
    loading?: boolean;
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

function Input({ label, name, type, value, step, loading, onChange }: InputProps) {
    return (
        <div className="flex flex-col">
            <label htmlFor={name} className="text-red-600 font-semibold mb-1 text-sm">
                {label}:
            </label>
            <input
                id={name}
                type={type}
                name={name}
                value={value}
                step={step}
                onChange={onChange}
                disabled={loading}
                className={`border border-red-300 p-2 rounded-md text-black focus:outline-none focus:ring-2 focus:ring-red-400 ${
                    loading ? 'bg-gray-100 cursor-not-allowed' : ''
                }`}
                required
            />
        </div>
    );
}

interface SelectProps {
    label: string;
    name: string;
    options: string[];
    value: string;
    loading?: boolean;
    onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
}

function Select({ label, name, options, value, loading, onChange }: SelectProps) {
    return (
        <div className="flex flex-col">
            <label htmlFor={name} className="text-red-600 font-semibold mb-1 text-sm">
                {label}:
            </label>
            <select
                id={name}
                name={name}
                value={value}
                onChange={onChange}
                disabled={loading}
                className={`border border-red-300 p-2 rounded-md text-black focus:outline-none focus:ring-2 focus:ring-red-400 ${
                    loading ? 'bg-gray-100 cursor-not-allowed' : ''
                }`}
                required
            >
                <option value="">Select</option>
                {options.map((opt, index) => (
                    <option key={index} value={opt}>
                        {opt}
                    </option>
                ))}
            </select>
        </div>
    );
}

interface FileUploadProps {
    label: string;
    name: string;
    accept: string;
    preview: string;
    previewType: "image" | "video";
    loading?: boolean;
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

// Fix for the FileUpload component - update this component in Heart.tsx
function FileUpload({ label, name, accept, preview, previewType, loading, onChange }: FileUploadProps) {
    return (
        <div className="flex flex-col">
            <label className="text-red-600 font-semibold mb-1 text-sm">{label}</label>
            <input
                type="file"
                name={name}
                accept={accept}
                onChange={onChange}
                disabled={loading}
                className={`p-2 border border-red-300 rounded-md text-black text-sm focus:outline-none focus:ring-2 focus:ring-red-400 ${
                    loading ? 'bg-gray-100 cursor-not-allowed' : ''
                }`}
            />
            {preview && previewType === "image" && (
                <img
                    src={preview}
                    alt={`${name} Preview`}
                    className="mt-2 max-h-40 rounded border object-contain bg-gray-50"
                    onError={(e) => {
                        console.error(`Failed to load ${name} preview`);
                        e.currentTarget.src = "https://via.placeholder.com/200x150?text=Preview+Error";
                    }}
                />
            )}
            {preview && previewType === "video" && (
                <div className="mt-2 max-h-40 rounded border bg-gray-50 relative">
                    <video 
                        className="w-full h-40 object-contain" 
                        controls
                        onError={(e) => {
                            console.error(`Failed to load ${name} video preview`);
                            const target = e.currentTarget;
                            const parent = target.parentElement;
                            if (parent) {
                                const errorMsg = document.createElement('div');
                                errorMsg.textContent = 'Video preview unavailable';
                                errorMsg.className = 'absolute inset-0 flex items-center justify-center bg-gray-200 text-gray-600';
                                parent.appendChild(errorMsg);
                                target.style.display = 'none';
                            }
                        }}
                    >
                        <source src={preview} type="video/mp4" />
                        Your browser does not support the video tag.
                    </video>
                </div>
            )}
        </div>
    );
}

interface ResultItemProps {
    label: string;
    value: string;
}

function ResultItem({ label, value }: ResultItemProps) {
    return (
        <div className="mb-2">
            <span className="font-semibold text-red-600">{label}:</span>
            <span className="text-black ml-2">{value}</span>
        </div>
    );
}

const getThalValue = (thalOption: string): string => {
    switch (thalOption) {
        case "Normal":
            return "1";
        case "Fixed Defect":
            return "2";
        case "Reversible Defect":
            return "3";
        default:
            return "0"; // Default fallback
    }
}

interface HeatmapVisualizationProps {
    label: string;
    gradcam?: string;
    score: number;
    analysis?: {
        normal_score?: number;
        abnormal_score?: number;
        affected_percentage?: number;
        needs_attention?: boolean;
    };
}

function HeatmapVisualization({ label, gradcam, score, analysis }: HeatmapVisualizationProps) {
    if (!gradcam) return null;

    const normalScore = analysis?.normal_score ?? 0;
    const abnormalScore = analysis?.abnormal_score ?? 0;
    const affectedPercentage = analysis?.affected_percentage ?? 0;
    const needsAttention = analysis?.needs_attention ?? false;

    return (
        <div className={`mt-4 p-4 border ${needsAttention ? 'border-red-400 bg-red-50' : 'border-red-200 bg-white'} rounded-lg shadow-md`}>
            <h4 className={`font-semibold ${needsAttention ? 'text-red-700' : 'text-red-600'} mb-2 flex items-center`}>
                {label} Analysis {needsAttention && <span className="ml-2 text-red-600 text-lg">⚠️</span>}
            </h4>
            <div className="flex flex-col items-center">
                <div className="relative w-full max-w-md">
                    {/* Enhanced Image with higher quality rendering */}
                    <img 
                        src={`data:image/png;base64,${gradcam}`} 
                        alt={`${label} Heatmap`}
                        className={`w-full h-auto rounded-lg ${needsAttention ? 'ring-2 ring-red-500' : 'shadow-lg'}`}
                        style={{ imageRendering: 'auto' }}
                    />
                    {/* Information overlay for X-Ray */}
                    {label === "X-Ray" && (
                        <div className="absolute bottom-2 left-2 right-2 flex flex-col bg-black bg-opacity-70 text-white px-3 py-2 rounded text-sm">
                            <div className="flex justify-between">
                                <span>Normal: {normalScore.toFixed(1)}%</span>
                                <span>Abnormal: {abnormalScore.toFixed(1)}%</span>
                            </div>
                            {needsAttention && (
                                <span className="text-yellow-300 mt-1 font-bold flex items-center">
                                    <svg className="w-4 h-4 mr-1 inline" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                                        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                                    </svg>
                                    Areas requiring medical attention detected
                                </span>
                            )}
                        </div>
                    )}
                    
                    {/* Information overlay for Echocardiogram */}
                    {label === "Echocardiogram" && (
                        <div className="absolute bottom-2 left-2 right-2 flex flex-col bg-black bg-opacity-70 text-white px-3 py-2 rounded text-sm">
                            <div className="flex justify-between">
                                <span>Normal: {(100 - score * 100).toFixed(1)}%</span>
                                <span>Abnormal: {(score * 100).toFixed(1)}%</span>
                            </div>
                            {needsAttention && (
                                <span className="text-yellow-300 mt-1 font-bold flex items-center">
                                    <svg className="w-4 h-4 mr-1 inline" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                                        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                                    </svg>
                                    Cardiac abnormality detected
                                </span>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

function ImageAnalysisSection({ resultData }: { resultData: ResultData }) {
    const hasXrayResults = resultData.xray_analysis && resultData.xray_analysis.gradcam;
    const hasEchoResults = resultData.echo_analysis && resultData.echo_analysis.gradcam;

    // More stringent threshold for determining if there are real abnormalities
    const hasRealAbnormalities = 
        ((resultData.xray_analysis?.abnormal_score ?? 0) > 75) || 
        ((resultData.echo_analysis?.score ?? 0) > 0.75);

    // Show a positive "normal" message for patients with no abnormalities
    if (!hasRealAbnormalities && (hasXrayResults || hasEchoResults)) {
        return (
            <div className="mt-8">
                <h3 className="text-xl font-semibold text-green-700 mb-4">Image Analysis Results:</h3>
                <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                    <div className="flex items-center text-green-700 mb-3">
                        <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                        </svg>
                        <span className="font-medium">
                            Good news! No significant abnormalities were detected in your imaging studies.
                        </span>
                    </div>
                    {hasXrayResults && (
                        <div className="mb-3">
                            <p className="text-sm text-green-800 font-medium">X-ray Analysis:</p>
                            <p className="text-sm text-gray-700">
                                Normal cardiac silhouette with clear lung fields. No significant abnormalities detected.
                            </p>
                        </div>
                    )}
                    {hasEchoResults && (
                        <div>
                            <p className="text-sm text-green-800 font-medium">Echo Analysis:</p>
                            <p className="text-sm text-gray-700">
                                Normal cardiac function and wall motion observed. No significant abnormalities detected.
                            </p>
                        </div>
                    )}
                </div>
            </div>
        );
    }

    // Original display for abnormal results remains the same...
    return (
        <div className="mt-8">
            <h3 className="text-xl font-semibold text-red-700 mb-4">Image Analysis Areas of Concern:</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {hasXrayResults && (
                    <div className="border-2 border-red-200 rounded-lg p-4 bg-red-50">
                        <h4 className="font-bold text-lg mb-3 text-red-800 flex items-center">
                            X-Ray Analysis
                            {resultData.xray_analysis?.needs_attention && (
                                <span className="ml-2 text-red-600">⚠️</span>
                            )}
                        </h4>
                        <div className="rounded-lg overflow-hidden shadow-md">
                            <div className="bg-black p-1">
                                <div className="relative">
                                    <img
                                        src={resultData.xray_analysis?.gradcam}
                                        alt="X-ray analysis with highlighted areas"
                                        className="w-full h-auto max-h-[400px] object-contain mx-auto"
                                        onError={(e) => {
                                            console.error("X-ray image failed to load");
                                            e.currentTarget.src = "https://via.placeholder.com/400x300?text=X-ray+Image+Failed";
                                        }}
                                    />
                                </div>
                            </div>
                        </div>

                        {resultData.xray_analysis?.explanations && resultData.xray_analysis.explanations.length > 0 && (
                            <div className="mt-3 text-sm">
                                <p className="font-semibold text-red-700">Key Findings:</p>
                                <ul className="list-disc pl-5 text-gray-700 space-y-1">
                                    {resultData.xray_analysis.explanations.slice(0, 2).map((exp, idx) => (
                                        <li key={`xray-exp-${idx}`}>{exp}</li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                )}

                {hasEchoResults && (
                    <div className="border-2 border-red-200 rounded-lg p-4 bg-red-50">
                        <h4 className="font-bold text-lg mb-3 text-red-800 flex items-center">
                            Echo Analysis
                            {resultData.echo_analysis?.needs_attention && (
                                <span className="ml-2 text-red-600">⚠️</span>
                            )}
                        </h4>
                        <div className="rounded-lg overflow-hidden shadow-md">
                            <div className="bg-black p-1">
                                <div className="relative">
                                    <img
                                        src={resultData.echo_analysis?.gradcam}
                                        alt="Echo analysis with highlighted areas"
                                        className="w-full h-auto max-h-[400px] object-contain mx-auto"
                                        onError={(e) => {
                                            console.error("Echo image failed to load");
                                            e.currentTarget.src = "https://via.placeholder.com/400x300?text=Echo+Image+Failed";
                                        }}
                                    />
                                </div>
                            </div>
                        </div>

                        {resultData.echo_analysis?.explanations && resultData.echo_analysis.explanations.length > 0 && (
                            <div className="mt-3 text-sm">
                                <p className="font-semibold text-red-700">Key Findings:</p>
                                <ul className="list-disc pl-5 text-gray-700 space-y-1">
                                    {resultData.echo_analysis.explanations.slice(0, 2).map((exp, idx) => (
                                        <li key={`echo-exp-${idx}`}>{exp}</li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Legend explaining the highlighted areas */}
            <div className="mt-4 p-3 bg-white border border-gray-200 rounded-lg shadow-sm">
                <p className="text-sm font-medium text-gray-700 mb-2">Visualization Legend:</p>
                <div className="grid grid-cols-2 gap-2">
                    <div className="flex items-center">
                        <div className="w-4 h-4 bg-red-500 mr-2"></div>
                        <span className="text-xs text-gray-700">High-concern areas requiring attention</span>
                    </div>
                    <div className="flex items-center">
                        <div className="w-4 h-4 bg-yellow-400 mr-2"></div>
                        <span className="text-xs text-gray-700">Moderate-concern areas</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

interface ModelConfidenceScoresProps {
    resultData: ResultData;
}

function ModelConfidenceScores({ resultData }: ModelConfidenceScoresProps) {
  return (
    <div className="mb-4">
      <h4 className="font-semibold text-red-600 mb-2">Model Confidence Scores:</h4>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        {/* Clinical Model */}
        <div className="bg-blue-50 p-3 rounded-lg">
          <div className="font-medium text-blue-800">Clinical Model</div>
          <div className="text-lg font-bold text-blue-700">
            {(resultData.clinical * 100).toFixed(1)}%
          </div>
        </div>
        
        {/* ECG Model - Add this block to show ECG confidence */}
        {resultData.ecg_analysis?.score !== undefined && (
          <div className="bg-yellow-50 p-3 rounded-lg">
            <div className="font-medium text-yellow-800">ECG Model</div>
            <div className="text-lg font-bold text-yellow-700">
              {(resultData.ecg_analysis.score * 100).toFixed(1)}%
            </div>
          </div>
        )}
        
        {/* X-Ray Model */}
        {resultData.xray_analysis?.confidence !== undefined && (
          <div className="bg-green-50 p-3 rounded-lg">
            <div className="font-medium text-green-800">X-Ray Model</div>
            <div className="text-lg font-bold text-green-700">
              {(resultData.xray_analysis.confidence * 100).toFixed(1)}%
            </div>
          </div>
        )}
        
        {/* Echo Model */}
        {resultData.echo_analysis?.score !== undefined && (
          <div className="bg-purple-50 p-3 rounded-lg">
            <div className="font-medium text-purple-800">Echo Model</div>
            <div className="text-lg font-bold text-purple-700">
              {(resultData.echo_analysis.score * 100).toFixed(1)}%
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

interface HealthRecommendationsProps {
    resultData: ResultData;
}

// Update the HealthRecommendations component
function HealthRecommendations({ resultData }: HealthRecommendationsProps) {
  const hasHeartDisease = resultData.prediction === "Heart Disease";
  // Important change: use imaging_emergency flag if available
  const hasImagingEmergency = resultData.imaging_emergency || 
    resultData.xray_analysis?.needs_attention || 
    resultData.echo_analysis?.needs_attention;
    
  // A patient needs consultation if either prediction indicates heart disease OR imaging shows emergency
  const needsConsultation = hasHeartDisease || hasImagingEmergency;
  
  return (
    <div className={`p-4 mt-4 rounded-lg ${needsConsultation ? 'bg-red-50 border border-red-200' : 'bg-green-50 border border-green-200'}`}>
      {/* If there's a contradiction between prediction and imaging, explain it */}
      {!hasHeartDisease && hasImagingEmergency && (
        <div className="mb-3 p-3 bg-yellow-100 border border-yellow-300 rounded">
          <p className="text-orange-700 font-bold">⚠️ Important Notice</p>
          <p className="text-orange-700">
            While your overall risk score is low, your imaging results show concerning findings
            that require medical attention. Please consult with a healthcare provider.
          </p>
        </div>
      )}
      
      {/* Rest of component */}
      {needsConsultation ? (
        <div className="space-y-3">
          <div className="flex items-start">
            <svg className="w-5 h-5 text-red-600 mr-2 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
            </svg>
            <span className="text-red-700 font-medium">
              Based on your results, we recommend consulting with a doctor for a comprehensive evaluation of your cardiac health.
            </span>
          </div>
          
          {/* Specific medical recommendations based on test results */}
          <div className="mt-2 ml-7 space-y-2">
            {resultData.clinical > 0.65 && (
              <div className="text-red-600">
                • Your clinical risk factors suggest elevated risk of heart disease that warrants medical evaluation.
              </div>
            )}
            
            {resultData.ecg_analysis?.score && resultData.ecg_analysis.score > 0.6 && (
              <div className="text-red-600">
                • Your ECG shows abnormal patterns that should be reviewed by a cardiologist.
              </div>
            )}
            
            {resultData.xray_analysis?.needs_attention && (
              <div className="text-red-600">
                • Your chest X-ray reveals areas of concern that require medical attention.
                {resultData.xray_analysis?.doctor_recommendation && (
                  <div className="ml-3 mt-1 text-red-500">
                    {resultData.xray_analysis.doctor_recommendation}
                  </div>
                )}
              </div>
            )}
            
            {resultData.echo_analysis?.needs_attention && (
              <div className="text-red-600">
                • Your echocardiogram indicates potential cardiac abnormalities that should be evaluated.
                {resultData.echo_analysis?.doctor_recommendation && (
                  <div className="ml-3 mt-1 text-red-500">
                    {resultData.echo_analysis.doctor_recommendation}
                  </div>
                )}
              </div>
            )}
          </div>
          
          <div className="mt-2 pt-2 border-t border-red-200 text-red-600 ml-7">
            Do not delay seeking medical attention as early intervention can significantly improve outcomes for heart-related conditions.
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="flex items-start">
            <svg className="w-5 h-5 text-green-600 mr-2 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <span className="text-green-700 font-medium">
              Your results appear to be within normal limits. To maintain your heart health:
            </span>
          </div>
          
          {/* Provide personalized recommendations even for normal results */}
          <ul className="ml-7 space-y-1 text-green-600 list-disc">
            <li>Continue with regular health check-ups, including blood pressure and cholesterol screening</li>
            <li>Maintain a balanced diet rich in fruits, vegetables, whole grains, and lean proteins</li>
            <li>Engage in regular physical activity (aim for at least 150 minutes of moderate exercise weekly)</li>
            <li>Manage stress through techniques like meditation, yoga, or deep breathing exercises</li>
            <li>Ensure you get 7-9 hours of quality sleep each night</li>
            <li>Avoid tobacco products and limit alcohol consumption</li>
          </ul>
          
          <div className="mt-2 pt-2 border-t border-green-200 text-green-600 ml-7">
            Even with good results, maintaining a heart-healthy lifestyle remains important for long-term cardiovascular health.
          </div>
        </div>
      )}
    </div>
  );
}

interface ExplanationsSectionProps {
    resultData: ResultData;
}

function ExplanationsSection({ resultData }: ExplanationsSectionProps) {
    if (!resultData) return null;
    
    const isNormalPrediction = resultData.prediction === "No Heart Disease";

    if (isNormalPrediction) {
        // Show positive health information for normal predictions
        return (
            <div className="mt-6 p-4 bg-white border border-green-200 rounded-lg shadow">
                <h3 className="text-green-600 font-bold text-lg mb-3">Healthy Heart Information</h3>
                
                <div className="space-y-4">
                    <div>
                        <h4 className="font-medium text-green-500 mb-2">Understanding Your Healthy Results</h4>
                        <p className="text-gray-700">
                            Your assessment indicates a low risk for heart disease. This means your clinical indicators 
                            and any provided imaging show patterns consistent with healthy cardiovascular function.
                        </p>
                    </div>
                    
                    <div>
                        <h4 className="font-medium text-green-500 mb-2">Benefits of Heart-Healthy Living</h4>
                        <ul className="list-disc pl-5 space-y-1 text-gray-700">
                            <li>Lower risk of heart attacks and stroke</li>
                            <li>Reduced chance of developing type 2 diabetes</li>
                            <li>Improved energy levels and quality of life</li>
                            <li>Better mental health and cognitive function</li>
                            <li>Longer life expectancy with more healthy years</li>
                        </ul>
                    </div>
                    
                    <div>
                        <h4 className="font-medium text-green-500 mb-2">Prevention Is Key</h4>
                        <p className="text-gray-700">
                            Even with healthy results, regular check-ups and screenings are important. The American Heart 
                            Association recommends adults have their cardiovascular risk factors checked starting at age 
                            20, including blood pressure, cholesterol, blood glucose, and body weight.
                        </p>
                    </div>
                    
                    <div className="bg-green-50 p-3 rounded-lg border border-green-100 mt-2">
                        <p className="text-green-700 font-medium flex items-center">
                            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                            </svg>
                            Remember: Heart health can change over time. Continue with regular check-ups and maintain a heart-healthy lifestyle.
                        </p>
                    </div>
                </div>
            </div>
        );
    }

    // Show detailed explanations for abnormal predictions
    if (!resultData.explanations || resultData.explanations.length === 0) {
        return null;
    }

    return (
        <div className="mt-6 p-4 bg-white border border-red-200 rounded-lg shadow">
            <h3 className="text-red-600 font-bold text-lg mb-3">Analysis Explanations</h3>
            <ul className="list-disc pl-5 space-y-2">
                {resultData.explanations.map((explanation, index) => (
                    <li key={index} className="text-gray-700">{explanation}</li>
                ))}
            </ul>

            {/* Display model-specific explanations if available */}
            {resultData.xray_analysis?.explanations && resultData.xray_analysis.explanations.length > 0 && (
                <div className="mt-4">
                    <h4 className="font-medium text-red-500">X-Ray Analysis Details:</h4>
                    <ul className="list-disc pl-5 space-y-1">
                        {resultData.xray_analysis.explanations.map((explanation, index) => (
                            <li key={`xray-${index}`} className="text-gray-600">{explanation}</li>
                        ))}
                    </ul>
                </div>
            )}

            {resultData.echo_analysis?.explanations && resultData.echo_analysis.explanations.length > 0 && (
                <div className="mt-4">
                    <h4 className="font-medium text-red-500">Echocardiogram Analysis Details:</h4>
                    <ul className="list-disc pl-5 space-y-1">
                        {resultData.echo_analysis.explanations.map((explanation, index) => (
                            <li key={`echo-${index}`} className="text-gray-600">{explanation}</li>
                        ))}
                    </ul>
                </div>
            )}

            {resultData.ecg_analysis?.explanations && resultData.ecg_analysis.explanations.length > 0 && (
                <div className="mt-4">
                    <h4 className="font-medium text-red-500">ECG Analysis Details:</h4>
                    <ul className="list-disc pl-5 space-y-1">
                        {resultData.ecg_analysis.explanations.map((explanation, index) => (
                            <li key={`ecg-${index}`} className="text-gray-600">{explanation}</li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}