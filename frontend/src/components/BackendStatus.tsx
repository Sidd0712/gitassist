import { useEffect, useState } from 'react';
import { healthCheck } from '../services/api';

export function BackendStatus() {
  const [status, setStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  useEffect(() => {
    const check = async () => {
      try {
        await healthCheck();
        setStatus('online');
      } catch {
        setStatus('offline');
      }
    };

    check();

    const interval = setInterval(check, 10000);
    return () => clearInterval(interval);
  }, []);

  const color =
    status === 'online'
      ? 'bg-green-500'
      : status === 'offline'
      ? 'bg-red-500'
      : 'bg-yellow-500';

  return (
    <div className="fixed top-4 right-4 z-50 flex items-center gap-2 glass px-3 py-2 rounded-full text-xs text-white">
      <span className={`w-2 h-2 rounded-full ${color}`} />
      {status === 'online' && 'Backend Online'}
      {status === 'offline' && 'Backend Offline'}
      {status === 'checking' && 'Checking...'}
    </div>
  );
}