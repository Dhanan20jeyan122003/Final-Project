"use client";
import React from 'react';
import Link from 'next/link';

const About: React.FC = () => {
  return (
    // Fix top spaces with more negative margin and relative positioning
    <section className="bg-red-500 w-screen -mx-4 -mt-8 pb-16 relative top-0" id="about">
      <div className="container mx-auto px-4">
        <div className="flex justify-center">
          <div className="w-full max-w-5xl">
            {/* Adjust top padding to compensate for the negative margin */}
            <div className="text-center mb-12 pt-20">
              <h2 className="text-4xl md:text-5xl font-bold text-white inline-flex items-center justify-center">
                <span className="text-green-300 mr-3">
                  <svg 
                    xmlns="http://www.w3.org/2000/svg" 
                    className="h-10 w-10" 
                    fill="none" 
                    viewBox="0 0 24 24" 
                    stroke="currentColor"
                  >
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth={2} 
                      d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" 
                    />
                  </svg>
                </span>
                About the Project
              </h2>
              <div className="w-24 h-1 bg-white mx-auto my-6"></div>
            </div>

            {/* Text content with left alignment */}
            <div className="text-white text-left px-4 md:px-6 lg:px-12 text-base md:text-lg">
              <p className="mb-6 text-center">
                <strong className="flex items-center justify-center flex-wrap">
                  Welcome to 
                  <span className="inline-flex items-center mx-2">
                    <svg 
                      xmlns="http://www.w3.org/2000/svg" 
                      className="h-5 w-5 text-green-300" 
                      fill="none" 
                      viewBox="0 0 24 24" 
                      stroke="currentColor"
                    >
                      <path 
                        strokeLinecap="round" 
                        strokeLinejoin="round" 
                        strokeWidth={2} 
                        d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" 
                      />
                    </svg>
                  </span>
                  Heart.io
                </strong>
              </p>

              <p className="mb-6 text-center">Dear Visitor,</p>

              {/* Change from justify-center to items-start for left alignment */}
              <p className="mb-6 flex items-start">
                <span className="mr-3 text-blue-300 mt-1">
                  <svg 
                    xmlns="http://www.w3.org/2000/svg" 
                    className="h-6 w-6" 
                    fill="none" 
                    viewBox="0 0 24 24" 
                    stroke="currentColor"
                  >
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth={2} 
                      d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064" 
                    />
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth={2} 
                      d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" 
                    />
                  </svg>
                </span>
                <span>
                  Heart disease is the world's leading cause of death, often going undiagnosed. At <strong>Heart.io</strong>, we use AI to detect heart disease <strong>faster, smarter, and more accurately</strong>.
                </span>
              </p>

              {/* Change all flex containers from justify-center to items-start */}
              <p className="mb-6 flex items-start">
                <span className="mr-3 text-yellow-300 mt-1">
                  <svg 
                    xmlns="http://www.w3.org/2000/svg" 
                    className="h-6 w-6" 
                    fill="none" 
                    viewBox="0 0 24 24" 
                    stroke="currentColor"
                  >
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth={2} 
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" 
                    />
                  </svg>
                </span>
                <span>
                  Our platform blends <strong>Hybrid Machine Learning + Deep Learning</strong> to assess both clinical and image data for more reliable risk assessment.
                </span>
              </p>

              {/* Keep section headers centered */}
              <p className="mt-8 mb-4 text-center">
                <strong className="flex items-center justify-center">
                  <span className="mr-3 text-teal-300">
                    <svg 
                      xmlns="http://www.w3.org/2000/svg" 
                      className="h-6 w-6" 
                      fill="none" 
                      viewBox="0 0 24 24" 
                      stroke="currentColor"
                    >
                      <path 
                        strokeLinecap="round" 
                        strokeLinejoin="round" 
                        strokeWidth={2} 
                        d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" 
                      />
                      <path 
                        strokeLinecap="round" 
                        strokeLinejoin="round" 
                        strokeWidth={2} 
                        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" 
                      />
                    </svg>
                  </span>
                  How Does Heart.io Work?
                </strong>
              </p>

              {/* Left align process steps */}
              <p className="mb-4 flex items-start">
                <span className="mr-3 text-orange-300 mt-1">
                  <svg 
                    xmlns="http://www.w3.org/2000/svg" 
                    className="h-6 w-6" 
                    fill="none" 
                    viewBox="0 0 24 24" 
                    stroke="currentColor"
                  >
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth={2} 
                      d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" 
                    />
                  </svg>
                </span>
                <span>
                  <strong>1. Patient Clinical Details</strong><br />
                  Enter data like age, cholesterol, and blood pressure. ML algorithms analyze these to assess risk.
                </span>
              </p>

              {/* Continue updating each flex element with items-start */}
              
            </div>
          </div>
        </div>

        {/* Keep CTA section centered */}
        <div className="text-center mt-12 pb-8">
          <div className="mx-auto max-w-2xl">
            <h3 className="text-3xl font-bold text-white mb-4 inline-flex items-center justify-center">
              <span className="text-blue-300 mr-2">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  className="h-6 w-6" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    strokeWidth={2} 
                    d="M13 10V3L4 14h7v7l9-11h-7z" 
                  />
                </svg>
              </span>
              Let's Go!
            </h3>
            <div className="h-1 w-12 bg-white mx-auto mb-6"></div>
            <Link 
              href="/services" 
              className="bg-white text-red-500 hover:bg-gray-100 px-8 py-3 rounded-lg font-bold transition duration-300 inline-flex items-center"
            >
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                className="h-5 w-5 mr-2 text-blue-500" 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" 
                />
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" 
                />
              </svg>
              Get Started!
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;