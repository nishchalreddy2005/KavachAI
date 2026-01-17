import React from 'react';
import { BarChart3, Shield, AlertTriangle, Clock, TrendingUp, Eye } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

export default function Dashboard() {
  const { user } = useAuth();

  const mockLogs = [
    {
      id: 1,
      timestamp: '2025-01-07T10:30:00Z',
      status: 'safe',
      source: 'Manual Input',
      confidence: 85
    },
    {
      id: 2,
      timestamp: '2025-01-07T09:15:00Z',
      status: 'warning',
      source: 'CSV Upload',
      confidence: 78
    },
    {
      id: 3,
      timestamp: '2025-01-06T16:45:00Z',
      status: 'danger',
      source: 'Manual Input',
      confidence: 92
    },
    {
      id: 4,
      timestamp: '2025-01-06T14:20:00Z',
      status: 'safe',
      source: 'CSV Upload',
      confidence: 88
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'safe':
        return 'text-green-400 bg-green-600/20';
      case 'warning':
        return 'text-yellow-400 bg-yellow-600/20';
      case 'danger':
        return 'text-red-400 bg-red-600/20';
      default:
        return 'text-gray-400 bg-gray-600/20';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'safe':
        return <Shield className="h-4 w-4" />;
      case 'warning':
        return <AlertTriangle className="h-4 w-4" />;
      case 'danger':
        return <AlertTriangle className="h-4 w-4" />;
      default:
        return <Shield className="h-4 w-4" />;
    }
  };

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4">
        <div className="text-center">
          <Shield className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-white mb-4">Access Restricted</h2>
          <p className="text-gray-400">Please log in to access your dashboard</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen px-4 sm:px-6 lg:px-8 py-12">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Security Dashboard</h1>
          <p className="text-gray-400">Welcome back, {user.email}</p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          <StatCard
            title="Total Scans"
            value="24"
            change="+12%"
            icon={<BarChart3 className="h-6 w-6" />}
            color="blue"
          />
          <StatCard
            title="Threats Blocked"
            value="3"
            change="+1"
            icon={<Shield className="h-6 w-6" />}
            color="green"
          />
          <StatCard
            title="High Risk Alerts"
            value="1"
            change="-2"
            icon={<AlertTriangle className="h-6 w-6" />}
            color="red"
          />
          <StatCard
            title="Avg Response Time"
            value="2.3s"
            change="-0.5s"
            icon={<Clock className="h-6 w-6" />}
            color="cyan"
          />
        </div>

        {/* Recent Intrusion Logs */}
        <div className="bg-slate-800/40 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-white flex items-center">
              <Eye className="h-6 w-6 mr-2" />
              Recent Intrusion Logs
            </h2>
            <button className="text-blue-400 hover:text-blue-300 font-medium">
              View All
            </button>
          </div>

          <div className="space-y-4">
            {mockLogs.map((log) => (
              <div
                key={log.id}
                className="flex items-center justify-between p-4 bg-slate-700/30 rounded-lg border border-slate-600/50 hover:border-slate-600 transition-colors"
              >
                <div className="flex items-center space-x-4">
                  <div className={`p-2 rounded-lg ${getStatusColor(log.status)}`}>
                    {getStatusIcon(log.status)}
                  </div>
                  <div>
                    <p className="text-white font-medium">
                      {log.status === 'safe' && 'No Threat Detected'}
                      {log.status === 'warning' && 'Potential Threat'}
                      {log.status === 'danger' && 'Critical Threat'}
                    </p>
                    <p className="text-gray-400 text-sm">{log.source}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-white text-sm">{log.confidence}% confidence</p>
                  <p className="text-gray-400 text-xs">
                    {new Date(log.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

interface StatCardProps {
  title: string;
  value: string;
  change: string;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'red' | 'cyan';
}

function StatCard({ title, value, change, icon, color }: StatCardProps) {
  const colorClasses = {
    blue: 'from-blue-600/20 to-blue-800/20 border-blue-500/30 text-blue-400',
    green: 'from-green-600/20 to-green-800/20 border-green-500/30 text-green-400',
    red: 'from-red-600/20 to-red-800/20 border-red-500/30 text-red-400',
    cyan: 'from-cyan-600/20 to-cyan-800/20 border-cyan-500/30 text-cyan-400'
  };

  return (
    <div className="bg-slate-800/40 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6">
      <div className="flex items-center justify-between mb-4">
        <div className={`p-2 rounded-lg bg-gradient-to-br ${colorClasses[color]}`}>
          {icon}
        </div>
        <div className="flex items-center text-sm">
          <TrendingUp className="h-4 w-4 text-green-400 mr-1" />
          <span className="text-green-400">{change}</span>
        </div>
      </div>
      <h3 className="text-2xl font-bold text-white mb-1">{value}</h3>
      <p className="text-gray-400 text-sm">{title}</p>
    </div>
  );
}