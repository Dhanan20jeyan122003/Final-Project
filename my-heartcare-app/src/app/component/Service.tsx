import React from 'react';
import { Brain, Accessibility, LineChart, Heart } from 'lucide-react';

interface ServiceItemProps {
  icon: React.ReactNode;
  title: string;
  description: string;
}

const ServiceItem: React.FC<ServiceItemProps> = ({ icon, title, description }) => {
  return (
    <div className="flex flex-col items-center text-center">
      <div className="text-red-500 mb-4">
        {icon}
      </div>
      <h3 className="text-2xl font-semibold mb-2 text-black">{title}</h3>
      <p className="text-gray-600 max-w-xs">{description}</p>
    </div>
  );
};

const Service: React.FC = () => {
  const services = [
    {
      icon: <Brain size={48} />,
      title: "Hybrid Intelligence",
      description: "Combining CNN with classic ML to enhance heart disease prediction accuracy."
    },
    {
      icon: <Accessibility size={48} />,
      title: "Accessible for All",
      description: "Just an ECG image or a few inputs is all it takes — anytime, anywhere."
    },
    {
      icon: <LineChart size={48} />,
      title: "Data-Driven",
      description: "Trained on diverse datasets for robust and scalable performance."
    },
    {
      icon: <Heart size={48} />,
      title: "Heart-Centric",
      description: "Crafted with care to save lives through proactive heart health monitoring."
    }
  ];

  return (
    <section className="py-16 px-4">
      <div className="container mx-auto">
        <h2 className="text-4xl font-bold text-center mb-2 text-black">At Your Service</h2>
        <div className="flex justify-center mb-12">
          <div className="w-16 h-1 bg-red-500"></div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {services.map((service, index) => (
            <ServiceItem
              key={index}
              icon={service.icon}
              title={service.title}
              description={service.description}
            />
          ))}
        </div>
      </div>
    </section>
  );
};

export default Service;