import React from 'react';

interface HeartHealthBannerProps {
  onSubscribe?: () => void;
}

const Footer: React.FC<HeartHealthBannerProps> = ({ onSubscribe }) => {
  return (
    // Extended footer to fill bottom spaces
    <footer className="w-screen bg-red-500 text-white py-16 -mx-4 mt-0 -mb-4 pb-8">
      <div className="max-w-4xl mx-auto px-4">
        <h2 className="text-3xl md:text-4xl font-bold mb-6 text-center">
          Stay Connected with Our Heart Health Project
        </h2>
        
        <div className="w-20 h-1 bg-white mx-auto mb-8"></div>
        
        <p className="text-lg md:text-xl mb-4 text-center">
          We're constantly working to make heart disease prediction smarter and more accessible.
        </p>

        <p className="mt-10 text-xl font-medium text-center">
          Your health, our mission.
        </p>
        
        {/* Copyright section with extended bottom padding */}
        <div className="mt-16 pb-12 text-center">
          <div className="w-16 h-1 bg-white/40 mx-auto mb-6"></div>
          <p className="text-sm text-white/80">
            © {new Date().getFullYear()} heart.io. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;