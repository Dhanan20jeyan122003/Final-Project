/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
      "./pages/**/*.{js,ts,jsx,tsx}",
      "./components/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
      extend: {
        colors: {
          'red': {
            500: '#FF3B41', // Custom red color matching the navbar in the image
            600: '#E52F35', // Slightly darker for hover states
          },
          'green': {
            300: '#1de9b6', // Teal/green color used for icons
          },
          'blue': {
            300: '#00bcd4', // Light blue color used for globe icon
          },
          'yellow': {
            300: '#ffd54f', // Yellow color used for brain icon
          },
          'teal': {
            300: '#4db6ac', // Teal color used for cogs icon
          },
          'orange': {
            300: '#ff8a65', // Orange color used for user icon
          },
          'purple': {
            300: '#ba68c8', // Purple color used for image icon
          }
        },
      },
    },
    plugins: [],
  }