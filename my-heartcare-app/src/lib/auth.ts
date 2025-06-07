import { cookies } from 'next/headers';
import dbConnect from './mongodb';
import User from '@/models/User';
import Visit from '@/models/Visit';

export async function getAuthenticatedUser(req: Request) {
  try {
    await dbConnect();
    
    // Get session ID from cookies
    const cookieStore = await cookies();
    const sessionId = cookieStore.get('sessionId')?.value;
    
    if (!sessionId) {
      return null;
    }
    
    // Find latest visit with this session ID
    const visit = await Visit.findOne({ sessionId })
      .sort({ createdAt: -1 })
      .populate('user');
      
    if (!visit || !visit.user) {
      return null;
    }
    
    // Return user information
    return {
      id: visit.user._id,
      name: visit.user.name,
      email: visit.user.email,
      role: visit.user.role
    };
  } catch (error) {
    console.error('Auth error:', error);
    return null;
  }
}