import { NextResponse } from 'next/server';
import dbConnect from '@/lib/mongodb';
import Visit from '@/models/Visit';
import { cookies } from 'next/headers';
import { getAuthenticatedUser } from '@/lib/auth';

export async function POST(req: Request) {
  try {
    await dbConnect();
    
    const user = await getAuthenticatedUser(req);
    if (!user) {
      return NextResponse.json({ success: false, message: 'Unauthorized' }, { status: 401 });
    }
    
    const { page, interactionTime, assessmentId } = await req.json();
    const sessionCookies = await cookies();
    const sessionId = sessionCookies.get('sessionId')?.value || 'unknown';
    
    // Find existing visit for this session or create new one
    let visit = await Visit.findOne({ 
      user: user.id,
      sessionId,
      createdAt: { $gte: new Date(Date.now() - 24 * 60 * 60 * 1000) } // within last 24 hours
    });
    
    if (visit) {
      // Update existing visit
      if (page && !visit.pages.includes(page)) {
        visit.pages.push(page);
      }
      
      if (interactionTime) {
        visit.interactionTime = (visit.interactionTime || 0) + interactionTime;
      }
      
      if (assessmentId) {
        if (assessmentId.viewed && !visit.assessmentsViewed.includes(assessmentId.viewed)) {
          visit.assessmentsViewed.push(assessmentId.viewed);
        }
        
        if (assessmentId.created) {
          visit.assessmentCreated = assessmentId.created;
        }
      }
      
      await visit.save();
    } else {
      // Create new visit
      visit = await Visit.create({
        user: user.id,
        sessionId,
        pages: page ? [page] : [],
        interactionTime: interactionTime || 0,
        assessmentsViewed: assessmentId?.viewed ? [assessmentId.viewed] : [],
        assessmentCreated: assessmentId?.created || null,
      });
    }
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error logging visit:', error);
    return NextResponse.json({ success: false }, { status: 500 });
  }
}