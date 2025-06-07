import React from 'react';
import Head from 'next/head';
import Image from 'next/image';

const Header: React.FC = () => {
  return (
    <>
      <Head>
        <title>heart.io - Saving Lives One jpg at a Time</title>
        <meta name="description" content="heart.io - Medical imaging and diagnosis solutions" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      {/* Use negative margin to eliminate any top space and extend background up */}
      <main className="w-screen overflow-x-hidden -mt-[64px]">
        {/* Hero section with the geometric background and heart logo */}
        <div className="relative w-full h-screen flex flex-col items-center justify-center">
          {/* Geometric background - Fixed to cover top edge completely */}
          <div className="absolute top-0 left-0 w-screen h-screen z-0">
            <div 
                className="absolute top-0 left-0 w-full h-full" 
                style={{ 
                  backgroundImage: "url('/img/header.png')",
                  backgroundSize: "cover",
                  backgroundPosition: "center top", /* Position from top */
                  backgroundRepeat: "no-repeat",
                  width: "100vw", 
                  height: "calc(100% + 64px)", /* Extend height to cover bottom gap */
                  margin: 0,
                  padding: 0,
                  transform: "translateY(-64px)", /* Pull background up to fill navbar gap */
                }}>
            </div>
          </div>
          
          {/* Logo and content */}
          <div className="z-10 text-center w-full max-w-4xl mx-auto px-4">
            <div className="mx-auto w-64 h-64 relative mb-8">
              <Image 
                src="/img/logo2.png" 
                alt="heart.io logo" 
                fill
                style={{objectFit: "contain"}}
                priority
              />
            </div>
            
            <h1 className="text-6xl md:text-7xl font-bold text-gray-800 mb-8">heart.io</h1>
            
            <div className="w-24 h-1 bg-red-500 mx-auto mb-8"></div>
            
            <p className="text-2xl md:text-3xl text-gray-700 italic mb-16">Saving Lives One jpg at a Time</p>
            
            {/* Scroll down button */}
            <button className="w-16 h-16 rounded-full bg-red-500 flex items-center justify-center text-white shadow-lg hover:bg-red-600 transition-colors duration-300 mx-auto animate-bounce">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
              </svg>
            </button>
          </div>
        </div>
        
        {/* Enhanced bottom extension to ensure no gap with About section */}
        <div className="bg-[url('/img/header.png')] bg-cover bg-center h-32 -mt-16 w-screen mb-0"></div>
        
        {/* Additional page content would go here */}
      </main>
    </>
  );
};

export default Header;