"use client";

import React, { useState, useEffect } from 'react';
import Image from 'next/image';

interface GalleryItemProps {
  id: string;
  category: string;
  name: string;
  thumbnailSrc: string;
  fullSizeSrc: string;
}

const Portfolio: React.FC = () => {
  const [isClient, setIsClient] = useState(false);
  
  // Use useEffect to confirm we're on client side
  useEffect(() => {
    setIsClient(true);
  }, []);

  const galleryItems: GalleryItemProps[] = [
    {
      id: '1',
      category: 'ECG PATTERN',
      name: 'Normal Heartbeat',
      thumbnailSrc: '/img/portfolio/thumbnails/1.jpg',
      fullSizeSrc: '/img/portfolio/fullsize/1.jpg',
    },
    {
      id: '2',
      category: 'ECG PATTERN',
      name: 'Myocardial Infarction',
      thumbnailSrc: '/img/portfolio/thumbnails/2.jpg',
      fullSizeSrc: '/img/portfolio/fullsize/2.jpg',
    },
    {
      id: '3',
      category: 'ECG PATTERN',
      name: 'Bundle Branch Block',
      thumbnailSrc: '/img/portfolio/thumbnails/3.jpg',
      fullSizeSrc: '/img/portfolio/fullsize/3.jpg',
    },
    {
      id: '4',
      category: 'HEART CONDITION',
      name: 'Atrial Fibrillation',
      thumbnailSrc: '/img/portfolio/thumbnails/4.jpg',
      fullSizeSrc: '/img/portfolio/fullsize/4.jpg',
    },
    {
      id: '5',
      category: 'RISK FACTOR',
      name: 'High Blood Pressure',
      thumbnailSrc: '/img/portfolio/thumbnails/5.jpg',
      fullSizeSrc: '/img/portfolio/fullsize/5.jpg',
    },
    {
      id: '6',
      category: 'RISK FACTOR',
      name: 'Cholesterol Imbalance',
      thumbnailSrc: '/img/portfolio/thumbnails/6.jpg',
      fullSizeSrc: '/img/portfolio/fullsize/6.jpg',
    },
  ];

  // State for lightbox functionality
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  const openLightbox = (src: string) => {
    setSelectedImage(src);
  };

  const closeLightbox = () => {
    setSelectedImage(null);
  };

  // Add title and section heading
  return (
    <section className="w-screen bg-white overflow-hidden" id="portfolio" style={{ margin: 0, padding: 0 }}>
      {/* Header - Changed text color to red */}
      <div className="w-full px-0 pt-16 pb-10">
        <div className="text-center max-w-4xl mx-auto px-4">
          <h2 className="text-4xl font-bold mb-4 text-red-500">Heart Patterns</h2>
          <p className="text-lg text-gray-600 mb-8">Common ECG patterns associated with various heart conditions.</p>
          <div className="w-24 h-1 bg-red-500 mx-auto"></div>
        </div>
      </div>

      {/* Gallery - Full Width with no side spaces */}
      {isClient && (
        <div className="w-screen mx-0 p-0">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-0">
            {galleryItems.map((item) => (
              <GalleryItem 
                key={item.id} 
                {...item} 
                onImageClick={openLightbox}
              />
            ))}
          </div>
        </div>
      )}

      {/* Lightbox - Full Screen */}
      {isClient && selectedImage && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-90 z-50 flex items-center justify-center"
          onClick={closeLightbox}
          style={{ width: '100vw', height: '100vh' }}
        >
          <div className="relative w-full h-full flex items-center justify-center p-4">
            <button 
              className="absolute top-4 right-4 text-white bg-red-500 rounded-full p-2 z-10"
              onClick={(e) => {
                e.stopPropagation();
                closeLightbox();
              }}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            <div className="relative w-full h-full max-w-5xl max-h-[90vh] flex items-center justify-center">
              <Image
                src={selectedImage}
                alt="Enlarged view"
                fill
                className="object-contain"
                onClick={(e) => e.stopPropagation()}
                sizes="100vw"
                priority
              />
            </div>
          </div>
        </div>
      )}
    </section>
  );
};

// Gallery Item Component - Full Size
interface GalleryItemWithClickProps extends GalleryItemProps {
  onImageClick: (src: string) => void;
}

const GalleryItem: React.FC<GalleryItemWithClickProps> = ({
  id,
  category,
  name,
  thumbnailSrc,
  fullSizeSrc,
  onImageClick
}) => {
  const [imgError, setImgError] = useState(false);

  return (
    <div className="w-full h-80 sm:h-96 relative group overflow-hidden cursor-pointer">
      <div 
        className="w-full h-full"
        onClick={() => onImageClick(fullSizeSrc)}
      >
        {/* Image with better error handling */}
        <div className="w-full h-full relative">
          {imgError ? (
            <div className="w-full h-full bg-gray-200 flex items-center justify-center">
              <span className="text-gray-500">Image not available</span>
            </div>
          ) : (
            <Image
              src={thumbnailSrc}
              alt={name}
              fill
              sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
              className="object-cover w-full h-full"
              onError={() => setImgError(true)}
              priority
            />
          )}
          
          {/* Red overlay */}
          <div className="absolute inset-0 bg-red-500 opacity-70"></div>
        </div>

        {/* Text overlay */}
        <div className="absolute inset-0 flex items-center justify-center text-center text-white p-4">
          <div>
            <div className="text-sm font-light uppercase tracking-wider mb-2">
              {category}
            </div>
            <div className="text-2xl font-bold">
              {name}
            </div>
          </div>
        </div>

        {/* Hover effect */}
        <div className="absolute inset-0 bg-red-600 opacity-0 group-hover:opacity-90 transition-opacity duration-300 flex items-center justify-center">
          <div className="text-white text-center p-4">
            <div className="text-sm font-light uppercase tracking-wider mb-2">
              {category}
            </div>
            <div className="text-2xl font-bold mb-2">
              {name}
            </div>
            <div className="w-8 h-1 bg-white mx-auto mt-2 mb-3"></div>
            <div className="text-sm">Click to view</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Portfolio;