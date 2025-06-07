import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Navigation from './component/Navigation'; // Import Navigation
import "./globals.css";

// Configure fonts
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: 'swap',
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: 'swap',
});

// Metadata is properly exported from a server component
export const metadata: Metadata = {
  title: "Heart.io - Heart Disease Diagnosis",
  description: "AI-powered heart disease diagnosis and analysis",
  keywords: "heart disease, diagnosis, AI, cardiac health, echo analysis",
  icons: {
    icon: '/img/logo2.png',
    apple: '/img/logo2.png',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen flex flex-col`}
      >
        <Navigation />
        <main className="flex-grow relative overflow-visible">
          {children}
        </main>
      </body>
    </html>
  );
}
