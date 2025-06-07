import mongoose from 'mongoose';

const PatientSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
  },
  medicalHistory: {
    heartDisease: Boolean,
    diabetes: Boolean,
    hypertension: Boolean,
    stroke: Boolean,
    otherConditions: String,
  },
  medications: [{
    name: String,
    dosage: String,
    frequency: String,
    startDate: Date,
  }],
  emergencyContact: {
    name: String,
    relationship: String,
    phone: String,
  },
}, {
  timestamps: true,
});

const Patient = mongoose.models.Patient || mongoose.model('Patient', PatientSchema);

export default Patient;