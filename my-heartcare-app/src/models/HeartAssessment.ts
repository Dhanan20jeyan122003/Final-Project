import mongoose from 'mongoose';

// Schema for clinical parameters
const ClinicalDataSchema = new mongoose.Schema({
  age: Number,
  sex: String,
  cp: String,
  trestbps: Number,
  chol: Number,
  fbs: String,
  restecg: String,
  thalach: Number,
  exang: String,
  oldpeak: Number,
  slope: String,
  ca: String,
  thal: String,
});

// Schema for model predictions
const PredictionResultSchema = new mongoose.Schema({
  prediction: String,
  confidence: Number,
  clinical: Number,
  explanations: [String],
  imaging_emergency: Boolean,
  ecg_analysis: {
    score: Number,
    explanations: [String],
    gradcam: String,
  },
  xray_analysis: {
    label: String,
    confidence: Number,
    affected_percentage: Number,
    normal_score: Number,
    abnormal_score: Number,
    needs_attention: Boolean,
    explanations: [String],
    gradcam: String,
    doctor_recommendation: String,
  },
  echo_analysis: {
    score: Number,
    explanations: [String],
    gradcam: String,
    doctor_recommendation: String,
    needs_attention: Boolean,
  },
});

// Main heart assessment schema
const HeartAssessmentSchema = new mongoose.Schema({
  patient: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
  },
  patientName: {
    type: String,
    required: true,
  },
  clinicalData: ClinicalDataSchema,
  files: {
    ecgImage: String, // Store file paths
    xrayImage: String,
    echoVideo: String,
  },
  results: PredictionResultSchema,
  notes: String,
  reviewedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
  },
  reviewNotes: String,
  dateCreated: {
    type: Date,
    default: Date.now,
  },
  pdfReport: String, // Path to stored PDF report
}, {
  timestamps: true,
});

const HeartAssessment = mongoose.models.HeartAssessment || 
  mongoose.model('HeartAssessment', HeartAssessmentSchema);

export default HeartAssessment;