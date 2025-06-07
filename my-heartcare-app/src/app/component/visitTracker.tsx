import { NextResponse } from 'next/server';
import dbConnect from '@/lib/mongodb';
import HeartAssessment from '@/models/HeartAssessment';
import { getAuthenticatedUser } from '@/lib/auth';

// Get all assessments for a user
export async function GET(req: Request) {
  try {
    await dbConnect();
    
    const user = await getAuthenticatedUser(req);
    if (!user) {
      return NextResponse.json({ success: false, message: 'Unauthorized' }, { status: 401 });
    }
    
    const { searchParams } = new URL(req.url);
    const patientId = searchParams.get('patientId') || user.id;
    
    // Admins and doctors can view all assessments or filter by patient
    // Patients can only view their own assessments
    const query = user.role === 'patient' 
      ? { patient: user.id }
      : patientId ? { patient: patientId } : {};
    
    const assessments = await HeartAssessment.find(query)
      .sort({ createdAt: -1 })
      .populate('patient', 'name email');
    
    return NextResponse.json({ 
      success: true,
      assessments 
    });
  } catch (error) {
    console.error('Error fetching assessments:', error);
    return NextResponse.json({ 
      success: false, 
      message: 'Failed to fetch assessments' 
    }, { status: 500 });
  }
}

// Save a new assessment
export async function POST(req: Request) {
  try {
    await dbConnect();
    
    const user = await getAuthenticatedUser(req);
    if (!user) {
      return NextResponse.json({ success: false, message: 'Unauthorized' }, { status: 401 });
    }
    
    const data = await req.json();
    
    // Create the assessment record
    const assessment = await HeartAssessment.create({
      patient: user.id,
      patientName: data.patientName || user.name,
      clinicalData: {
        age: data.age,
        sex: data.sex,
        cp: data.cp,
        trestbps: data.trestbps,
        chol: data.chol,
        fbs: data.fbs,
        restecg: data.restecg,
        thalach: data.thalach,
        exang: data.exang,
        oldpeak: data.oldpeak,
        slope: data.slope,
        ca: data.ca,
        thal: data.thal,
      },
      files: {
        ecgImage: data.ecgImagePath,
        xrayImage: data.xrayImagePath,
        echoVideo: data.echoVideoPath,
      },
      results: data.resultData,
      notes: data.notes || '',
    });
    
    return NextResponse.json({ 
      success: true, 
      message: 'Assessment saved successfully',
      assessmentId: assessment._id
    }, { status: 201 });
  } catch (error) {
    console.error('Error saving assessment:', error);
    return NextResponse.json({ 
      success: false, 
      message: 'Failed to save assessment' 
    }, { status: 500 });
  }
}