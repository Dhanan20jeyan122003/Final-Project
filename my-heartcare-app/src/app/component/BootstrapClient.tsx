"use client";

import { useEffect } from 'react';

export default function BootstrapClient() {
  useEffect(() => {
    // Add Bootstrap 4 scripts
    const jquery = document.createElement('script');
    jquery.src = 'https://code.jquery.com/jquery-3.5.1.slim.min.js';
    jquery.integrity = "sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj";
    jquery.crossOrigin = "anonymous";
    
    const popper = document.createElement('script');
    popper.src = 'https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js';
    popper.integrity = "sha384-9/reFTGAW83EW2RDu2S0VKaIzap3H66lZH81PoYlFhbGU+6BZp6G7niu735Sk7lN";
    popper.crossOrigin = "anonymous";
    
    const bootstrap = document.createElement('script');
    bootstrap.src = 'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js';
    bootstrap.integrity = "sha384-B4gt1jrGC7Jh4AgTPSdUtOBvfO8shuf57BaghqFfPlYxofvL8/KUEfYiJOMMV+rV";
    bootstrap.crossOrigin = "anonymous";
    
    // Append scripts in order with proper dependency sequence
    const appendScript = (script: HTMLScriptElement): Promise<void> => {
      return new Promise<void>((resolve) => {
        script.onload = () => resolve();
        document.body.appendChild(script);
      });
    };
    
    // Append scripts in sequence to ensure dependencies are loaded properly
    appendScript(jquery)
      .then(() => appendScript(popper))
      .then(() => appendScript(bootstrap))
      .then(() => console.log("Bootstrap loaded successfully"));
    
    // Clean up on unmount
    return () => {
      try {
        if (document.body.contains(jquery)) document.body.removeChild(jquery);
        if (document.body.contains(popper)) document.body.removeChild(popper);
        if (document.body.contains(bootstrap)) document.body.removeChild(bootstrap);
      } catch (e) {
        console.warn("Error removing bootstrap scripts:", e);
      }
    };
  }, []);
  
  return null;
}