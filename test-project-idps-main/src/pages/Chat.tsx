import React, { useState } from 'react';
import { Send, MessageSquare, User, Bot, Shield } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

interface Message {
  id: number;
  text: string;
  sender: 'user' | 'expert';
  timestamp: string;
}

export default function Chat() {
  const { user } = useAuth();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Hello! I'm a security expert. How can I help you today?",
      sender: 'expert',
      timestamp: new Date().toISOString()
    }
  ]);
  const [newMessage, setNewMessage] = useState('');

  const sendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newMessage.trim()) return;

    const userMessage: Message = {
      id: messages.length + 1,
      text: newMessage,
      sender: 'user',
      timestamp: new Date().toISOString()
    };

    setMessages([...messages, userMessage]);
    setNewMessage('');

    // Simulate expert response
    setTimeout(() => {
      const expertResponse: Message = {
        id: messages.length + 2,
        text: "Thanks for your question. Based on the information provided, I recommend implementing additional monitoring on those specific network segments. Would you like me to provide a detailed security assessment?",
        sender: 'expert',
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, expertResponse]);
    }, 1500);
  };

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4">
        <div className="text-center">
          <MessageSquare className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-white mb-4">Access Restricted</h2>
          <p className="text-gray-400">Please log in to access the chat with security experts</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen px-4 sm:px-6 lg:px-8 py-12">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <div className="flex justify-center mb-4">
            <div className="p-3 bg-cyan-600/20 backdrop-blur-sm rounded-2xl border border-cyan-500/30">
              <MessageSquare className="h-8 w-8 text-cyan-400" />
            </div>
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">Security Expert Chat</h1>
          <p className="text-gray-400">Get real-time guidance from cybersecurity professionals</p>
        </div>

        <div className="bg-slate-800/40 backdrop-blur-sm rounded-2xl border border-slate-700/50 overflow-hidden">
          {/* Chat Header */}
          <div className="p-6 border-b border-slate-700/50 bg-slate-700/30">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-green-600/20 rounded-full">
                <Shield className="h-5 w-5 text-green-400" />
              </div>
              <div>
                <h3 className="text-white font-semibold">Security Expert</h3>
                <p className="text-green-400 text-sm">Online</p>
              </div>
            </div>
          </div>

          {/* Messages */}
          <div className="h-96 overflow-y-auto p-6 space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`flex items-start space-x-3 max-w-xs lg:max-w-md ${
                  message.sender === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                }`}>
                  <div className={`p-2 rounded-full ${
                    message.sender === 'user' 
                      ? 'bg-blue-600/20 text-blue-400' 
                      : 'bg-cyan-600/20 text-cyan-400'
                  }`}>
                    {message.sender === 'user' ? (
                      <User className="h-4 w-4" />
                    ) : (
                      <Bot className="h-4 w-4" />
                    )}
                  </div>
                  <div className={`rounded-xl p-4 ${
                    message.sender === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-700/50 text-gray-300'
                  }`}>
                    <p className="text-sm leading-relaxed">{message.text}</p>
                    <p className={`text-xs mt-2 ${
                      message.sender === 'user' ? 'text-blue-100' : 'text-gray-400'
                    }`}>
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Message Input */}
          <div className="p-6 border-t border-slate-700/50 bg-slate-700/30">
            <form onSubmit={sendMessage} className="flex space-x-4">
              <input
                type="text"
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                placeholder="Type your security question..."
                className="flex-1 px-4 py-3 bg-slate-600/50 border border-slate-500 rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent text-white placeholder-gray-400"
              />
              <button
                type="submit"
                className="px-6 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 text-white font-medium rounded-lg transition-all duration-300 flex items-center space-x-2"
              >
                <Send className="h-4 w-4" />
                <span>Send</span>
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}