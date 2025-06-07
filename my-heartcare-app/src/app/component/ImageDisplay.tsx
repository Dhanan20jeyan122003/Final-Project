import { useState, useEffect } from 'react';

interface ImageDisplayProps {
  src: string | undefined;
  alt: string;
  className?: string;
  fallbackText?: string;
}

export default function ImageDisplay({ 
  src, 
  alt, 
  className = "w-full h-auto max-h-[400px] object-contain mx-auto", 
  fallbackText = "Image unavailable"
}: ImageDisplayProps) {
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | undefined>(undefined);

  useEffect(() => {
    console.log(`${alt} src provided:`, !!src);
    if (src) {
      // Log the first part of the string to debug
      console.log(`${alt} source starts with:`, src.substring(0, 50));
      console.log(`${alt} source length:`, src.length);
      setImageUrl(src);
    } else {
      setError(true);
    }
  }, [src, alt]);

  if (!imageUrl) {
    return (
      <div className="bg-gray-200 w-full h-[300px] flex items-center justify-center">
        <p className="text-gray-500">{fallbackText}</p>
      </div>
    );
  }

  return (
    <div className="bg-black p-1 relative min-h-[200px]">
      {!loaded && !error && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-200">
          <p className="text-gray-500">Loading image...</p>
        </div>
      )}
      
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-200">
          <p className="text-gray-500">{fallbackText}</p>
        </div>
      )}
      
      <img
        src={imageUrl}
        alt={alt}
        className={`${className} ${loaded ? 'block' : 'hidden'}`}
        onLoad={() => {
          console.log(`${alt} image loaded successfully`);
          setLoaded(true);
        }}
        onError={(e) => {
          console.error(`${alt} image failed to load`, e);
          setError(true);
        }}
      />
    </div>
  );
}