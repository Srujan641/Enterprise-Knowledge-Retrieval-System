import { useRef, useState } from 'react';
import axios from 'axios';
import { LogOut, Send, Upload, User, Loader2 } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const SESSION_KEY = 'rag-session';

const readSession = () => {
  try {
    return JSON.parse(localStorage.getItem(SESSION_KEY));
  } catch {
    return null;
  }
};

const writeSession = (session) => {
  try {
    if (session) localStorage.setItem(SESSION_KEY, JSON.stringify(session));
    else localStorage.removeItem(SESSION_KEY);
  } catch {
    // The session then lasts until the page is reloaded.
  }
};

const errorText = (err, fallback) => err.response?.data?.detail || fallback;

function Login({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError('');
    try {
      const res = await axios.post(`${API_URL}/api/auth/login`, { username, password });
      onLogin(res.data);
    } catch (err) {
      setError(errorText(err, 'Could not reach the server. Is the backend running?'));
    }
    setBusy(false);
  };

  const field = 'w-full border border-gray-300 rounded-md px-3 py-2 outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500';

  return (
    <div className="flex h-screen items-center justify-center bg-gray-100 p-4 font-sans">
      <form onSubmit={submit} className="w-full max-w-sm bg-white rounded-lg shadow-sm p-6 flex flex-col gap-4">
        <h1 className="font-bold text-xl text-blue-600">Enterprise RAG</h1>
        <label className="text-sm font-medium text-gray-700">
          Username
          <input className={`${field} mt-1`} value={username} onChange={e => setUsername(e.target.value)} autoComplete="username" required />
        </label>
        <label className="text-sm font-medium text-gray-700">
          Password
          <input className={`${field} mt-1`} type="password" value={password} onChange={e => setPassword(e.target.value)} autoComplete="current-password" required />
        </label>
        {error && <p className="text-sm text-red-600">{error}</p>}
        <button disabled={busy} className="bg-blue-600 hover:bg-blue-700 text-white rounded-md py-2 disabled:opacity-50 transition-colors">
          {busy ? 'Signing in...' : 'Sign in'}
        </button>
      </form>
    </div>
  );
}

function App() {
  const [session, setSession] = useState(readSession);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [visibility, setVisibility] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const fileInput = useRef(null);

  const signIn = (next) => {
    writeSession(next);
    setSession(next);
  };

  const signOut = () => {
    writeSession(null);
    setSession(null);
    setMessages([]);
    setUploadStatus(null);
    setVisibility('');
  };

  if (!session) return <Login onLogin={signIn} />;

  const authHeaders = { headers: { Authorization: `Bearer ${session.token}` } };
  const uploadRoles = session.role === 'Admin'
    ? ['general', 'HR', 'Admin']
    : session.role === 'general' ? ['general'] : [session.role, 'general'];
  const uploadRole = visibility || uploadRoles[0];

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const history = messages.filter(m => !m.failed).slice(-6).map(({ role, content }) => ({ role, content }));
      const res = await axios.post(`${API_URL}/api/query`, { query: userMsg.content, history }, authHeaders);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: res.data.answer,
        sources: res.data.sources,
        cached: res.data.cached
      }]);
    } catch (err) {
      if (err.response?.status === 401) signOut();
      else setMessages(prev => [...prev, { role: 'assistant', failed: true, content: errorText(err, 'Error fetching response. Is the backend running?') }]);
    }
    setLoading(false);
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setUploadStatus(null);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('role', uploadRole);

    try {
      const res = await axios.post(`${API_URL}/api/ingest`, formData, authHeaders);
      setUploadStatus({
        ok: true,
        text: res.data.duplicate ? `${file.name} is already indexed.` : `${file.name}: ${res.data.chunks} chunks indexed.`
      });
    } catch (err) {
      if (err.response?.status === 401) signOut();
      else setUploadStatus({ ok: false, text: errorText(err, 'Upload failed.') });
    }
    setUploading(false);
    if (fileInput.current) fileInput.current.value = '';
  };

  return (
    <div className="flex h-screen bg-gray-100 font-sans">
      <div className="w-64 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200 font-bold text-xl text-blue-600">Enterprise RAG</div>

        <div className="p-4 flex flex-col gap-4">
          <div className="flex items-center gap-2 text-sm text-gray-700">
            <User size={16} className="text-gray-500" />
            <span className="flex-1 truncate">{session.username} ({session.role})</span>
            <button onClick={signOut} title="Sign out" aria-label="Sign out" className="text-gray-500 hover:text-gray-800">
              <LogOut size={16} />
            </button>
          </div>

          <div>
            <label htmlFor="visibility" className="block text-sm font-medium text-gray-700 mb-1">Who can read uploads</label>
            <select
              id="visibility"
              value={uploadRole}
              onChange={e => setVisibility(e.target.value)}
              className="w-full border border-gray-300 rounded-md p-2 bg-transparent outline-none text-sm mb-3"
            >
              {uploadRoles.map(r => <option key={r} value={r}>{r === 'general' ? 'Everyone' : r}</option>)}
            </select>

            <label className="flex items-center justify-center gap-2 border-2 border-dashed border-gray-300 rounded-lg p-4 cursor-pointer hover:bg-gray-50 transition-colors">
              {uploading ? <Loader2 className="animate-spin text-blue-500" /> : <Upload className="text-blue-500" />}
              <span className="text-sm text-gray-600">{uploading ? 'Uploading...' : 'Click to Upload'}</span>
              <input ref={fileInput} type="file" className="hidden" onChange={handleUpload} accept=".pdf,.txt,.docx" disabled={uploading} />
            </label>
            {uploadStatus && (
              <p className={`text-xs mt-2 break-words ${uploadStatus.ok ? 'text-green-700' : 'text-red-600'}`}>{uploadStatus.text}</p>
            )}
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col min-w-0">
        <div className="flex-1 p-6 overflow-y-auto flex flex-col gap-4">
          {messages.length === 0 && (
            <div className="m-auto text-gray-400 text-center">
              <p className="text-lg mb-2">Welcome to Enterprise RAG Assistant</p>
              <p className="text-sm">Upload documents on the left, then ask questions below.</p>
            </div>
          )}

          {messages.map((m, idx) => (
            <div key={idx} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[70%] rounded-2xl p-4 shadow-sm ${m.role === 'user' ? 'bg-blue-600 text-white' : 'bg-white text-gray-800'}`}>
                <p className="whitespace-pre-wrap">{m.content}</p>
                {m.sources && m.sources.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <p className="text-xs font-semibold text-gray-500 mb-1">Cited Sources:</p>
                    <ul className="text-xs text-blue-500 list-disc list-inside">
                      {m.sources.map((s, i) => <li key={i}>{s}</li>)}
                    </ul>
                  </div>
                )}
                {m.cached && (
                  <span className="text-[10px] text-yellow-600 mt-2 block font-semibold">⚡ Cached Response</span>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-white rounded-2xl p-4 shadow-sm text-gray-800 flex items-center gap-2">
                <Loader2 size={16} className="animate-spin text-blue-500" /> Thinking...
              </div>
            </div>
          )}
        </div>

        <div className="p-4 bg-white border-t border-gray-200">
          <form onSubmit={handleSend} className="max-w-4xl mx-auto flex gap-2">
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Ask a question about your documents..."
              className="flex-1 border border-gray-300 rounded-full px-4 py-2 outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              aria-label="Send"
              className="bg-blue-600 hover:bg-blue-700 text-white rounded-full p-2 w-10 h-10 flex items-center justify-center disabled:opacity-50 transition-colors"
            >
              <Send size={18} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
