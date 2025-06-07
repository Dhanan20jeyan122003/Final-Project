import mongoose from 'mongoose';

const VisitSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
  },
  visitDate: {
    type: Date,
    default: Date.now,
  },
  ipAddress: String,
  userAgent: String,
  pages: [String],
  interactionTime: Number, // in seconds
  assessmentsViewed: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'HeartAssessment',
  }],
  assessmentCreated: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'HeartAssessment',
  },
  sessionId: String,
}, {
  timestamps: true,
});

const Visit = mongoose.models.Visit || mongoose.model('Visit', VisitSchema);

export default Visit;