import { NextResponse } from 'next/server';
import dbConnect from '@/lib/mongodb';
import User from '@/models/User';
import Patient from '@/models/Patient';

export async function POST(req: Request) {
  try {
    await dbConnect();
    const { name, email, password, dateOfBirth, gender } = await req.json();

    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return NextResponse.json({ success: false, message: 'Email already registered' }, { status: 400 });
    }

    // Create user
    const user = await User.create({
      name,
      email,
      password,
      dateOfBirth,
      gender,
      lastLogin: new Date(),
    });

    // Create patient record
    await Patient.create({
      user: user._id,
    });

    return NextResponse.json({ 
      success: true, 
      message: 'Registration successful',
      userId: user._id 
    }, { status: 201 });
  } catch (error) {
    console.error('Registration error:', error);
    return NextResponse.json({ 
      success: false, 
      message: 'Registration failed' 
    }, { status: 500 });
  }
}