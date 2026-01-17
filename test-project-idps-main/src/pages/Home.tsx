import React from 'react';
import { Link } from 'react-router-dom';
import { Shield, Zap, Users, BarChart3, ChevronRight, AlertTriangle } from 'lucide-react';

export default function Home() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative pt-20 pb-32 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center">
            <div className="flex justify-center mb-8">
              <div className="p-4 bg-blue-600/20 backdrop-blur-sm rounded-2xl border border-blue-500/30">
                <Shield className="h-16 w-16 text-blue-400" />
              </div>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
              AI Enhanced
              <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent block">
                Security Shield
              </span>
            </h1>
            
            <p className="text-xl text-gray-300 mb-12 max-w-3xl mx-auto leading-relaxed">
              Advanced AI-powered intrusion detection and prevention system that monitors your network 
              in real-time, identifying threats before they can cause damage.
            </p>

            <Link
              to="/check-intrusion"
              className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white font-semibold rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
            >
              <AlertTriangle className="h-5 w-5 mr-2" />
              Check Network Intrusion
              <ChevronRight className="h-5 w-5 ml-2" />
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 px-4 sm:px-6 lg:px-8 bg-slate-800/30 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">
              Comprehensive Security Features
            </h2>
            <p className="text-lg text-gray-400 max-w-2xl mx-auto">
              Our AI-powered system provides multi-layered protection with advanced analytics
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <FeatureCard
              icon={<Zap className="h-8 w-8" />}
              title="Real-time Detection"
              description="Monitor network traffic in real-time with AI-powered threat detection algorithms"
              color="blue"
            />
            <FeatureCard
              icon={<Shield className="h-8 w-8" />}
              title="Advanced Prevention"
              description="Automatically block suspicious activities before they can compromise your system"
              color="cyan"
            />
            <FeatureCard
              icon={<BarChart3 className="h-8 w-8" />}
              title="Detailed Analytics"
              description="Comprehensive reports and analytics to understand your security landscape"
              color="blue"
            />
            <FeatureCard
              icon={<Users className="h-8 w-8" />}
              title="Expert Support"
              description="Chat with security experts for guidance and threat analysis"
              color="cyan"
            />
            <FeatureCard
              icon={<AlertTriangle className="h-8 w-8" />}
              title="Log Analysis"
              description="Upload and analyze security logs using advanced machine learning models"
              color="blue"
            />
            <FeatureCard
              icon={<Shield className="h-8 w-8" />}
              title="User Privileges"
              description="Registered users get access to history, analytics, and premium features"
              color="cyan"
            />
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">
            Ready to Secure Your Network?
          </h2>
          <p className="text-lg text-gray-400 mb-12">
            Start protecting your infrastructure today with our AI-powered security system
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/signup"
              className="px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-xl transition-colors"
            >
              Get Started Free
            </Link>
            <Link
              to="/check-intrusion"
              className="px-8 py-4 bg-slate-700/50 hover:bg-slate-700 text-white font-semibold rounded-xl border border-slate-600 transition-colors"
            >
              Try Detection Now
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}

interface FeatureCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  color: 'blue' | 'cyan';
}

function FeatureCard({ icon, title, description, color }: FeatureCardProps) {
  const colorClasses = {
    blue: 'from-blue-600/20 to-blue-800/20 border-blue-500/30 text-blue-400',
    cyan: 'from-cyan-600/20 to-cyan-800/20 border-cyan-500/30 text-cyan-400'
  };

  return (
    <div className="group">
      <div className="h-full p-8 bg-slate-800/40 backdrop-blur-sm rounded-2xl border border-slate-700/50 hover:border-slate-600/50 transition-all duration-300 hover:transform hover:-translate-y-2">
        <div className={`inline-flex p-3 rounded-xl bg-gradient-to-br ${colorClasses[color]} mb-6`}>
          {icon}
        </div>
        <h3 className="text-xl font-semibold text-white mb-4">{title}</h3>
        <p className="text-gray-400 leading-relaxed">{description}</p>
      </div>
    </div>
  );
}