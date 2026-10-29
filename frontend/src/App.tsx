import React, { useState, useEffect, useRef } from 'react';
import {
  Database,
  Server,
  ShieldCheck,
  FileText,
  Layers,
  Cpu,
  RefreshCw,
  CheckCircle2,
  ExternalLink,
  Search,
  UploadCloud,
  Trash2,
  Download,
  LogOut,
  LogIn,
  UserPlus,
  Lock,
  Mail,
  User as UserIcon,
  X,
  FileCheck2,
  AlertTriangle,
  Sparkles,
  Quote,
} from 'lucide-react';

interface DatabaseHealth {
  status: 'connected' | 'disconnected';
  pgvector_available: boolean;
  message: string;
}

interface HealthResponse {
  status: 'healthy' | 'degraded';
  timestamp: string;
  database: DatabaseHealth;
  environment: {
    service: string;
    version: string;
    embedding_model?: string;
    embedding_dimension?: number;
  };
}

interface UserProfile {
  id: number;
  email: string;
  username: string;
  created_at: string;
}

interface DocumentItem {
  id: string;
  title: string;
  filename: string;
  file_size_bytes: number;
  page_count: number;
  content_type: string;
  status: 'queued' | 'processing' | 'completed' | 'failed' | string;
  error_message?: string | null;
  created_at: string;
}

interface SearchPassage {
  chunk_id: string;
  document_id: string;
  document_title: string;
  page_number: number;
  chunk_index: number;
  text: string;
  score: number;
  metadata: {
    embedding_model?: string;
    embedding_dimension?: number;
    character_count?: number;
  };
}

export default function App() {
  // System Health state
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loadingHealth, setLoadingHealth] = useState<boolean>(true);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);

  // Authentication state
  const [token, setToken] = useState<string | null>(() => localStorage.getItem('documind_token'));
  const [currentUser, setCurrentUser] = useState<UserProfile | null>(() => {
    const saved = localStorage.getItem('documind_user');
    return saved ? JSON.parse(saved) : null;
  });
  const [showAuthModal, setShowAuthModal] = useState<boolean>(false);
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [authEmail, setAuthEmail] = useState('');
  const [authUsername, setAuthUsername] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authError, setAuthError] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(false);

  // Documents state
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [loadingDocs, setLoadingDocs] = useState<boolean>(false);
  const [selectedDoc, setSelectedDoc] = useState<DocumentItem | null>(null);
  const [searchFilter, setSearchFilter] = useState('');

  // Upload state
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadTitle, setUploadTitle] = useState('');
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Semantic Search state
  const [activeTab, setActiveTab] = useState<'search' | 'upload'>('search');
  const [searchQuery, setSearchQuery] = useState('');
  const [filterDocId, setFilterDocId] = useState<string>('all');
  const [searching, setSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchPassage[] | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Health fetcher
  const fetchHealth = async () => {
    setLoadingHealth(true);
    const start = performance.now();
    try {
      const response = await fetch('/api/health');
      setLatencyMs(Math.round(performance.now() - start));
      if (response.ok) {
        const data: HealthResponse = await response.json();
        setHealth(data);
      } else {
        setHealth(null);
      }
    } catch {
      setLatencyMs(Math.round(performance.now() - start));
      setHealth(null);
    } finally {
      setLoadingHealth(false);
    }
  };

  // Documents fetcher
  const fetchDocuments = async (authToken?: string) => {
    const activeToken = authToken || token;
    if (!activeToken) {
      setDocuments([]);
      return;
    }

    setLoadingDocs(true);
    try {
      const response = await fetch('/api/documents', {
        headers: {
          Authorization: `Bearer ${activeToken}`,
        },
      });

      if (response.status === 401) {
        handleLogout();
        return;
      }

      if (response.ok) {
        const data = await response.json();
        setDocuments(data.documents || []);
      }
    } catch (err) {
      console.error('Failed to load documents:', err);
    } finally {
      setLoadingDocs(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 20000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (token) {
      fetchDocuments();
    } else {
      setDocuments([]);
      setSelectedDoc(null);
      setSearchResults(null);
    }
  }, [token]);

  // Auth Handlers
  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError(null);
    setAuthLoading(true);

    try {
      const endpoint = authMode === 'register' ? '/api/auth/register' : '/api/auth/login';
      const payload = authMode === 'register'
        ? { email: authEmail, username: authUsername, password: authPassword }
        : { username_or_email: authUsername || authEmail, password: authPassword };

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Authentication failed');
      }

      localStorage.setItem('documind_token', data.access_token);
      localStorage.setItem('documind_user', JSON.stringify(data.user));
      setToken(data.access_token);
      setCurrentUser(data.user);
      setShowAuthModal(false);
      setAuthPassword('');
      setAuthError(null);

      fetchDocuments(data.access_token);
    } catch (err: unknown) {
      setAuthError(err instanceof Error ? err.message : 'An error occurred during authentication.');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('documind_token');
    localStorage.removeItem('documind_user');
    setToken(null);
    setCurrentUser(null);
    setDocuments([]);
    setSelectedDoc(null);
    setSearchResults(null);
  };

  // Upload Handler
  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token) {
      setShowAuthModal(true);
      return;
    }

    if (!uploadFile) {
      setUploadError('Please select a PDF document to upload.');
      return;
    }

    if (!uploadFile.name.toLowerCase().endsWith('.pdf')) {
      setUploadError('Only PDF documents are supported (.pdf).');
      return;
    }

    if (uploadFile.size > 15 * 1024 * 1024) {
      setUploadError(`File exceeds maximum size of 15MB (${(uploadFile.size / (1024 * 1024)).toFixed(1)}MB).`);
      return;
    }

    setUploading(true);
    setUploadProgress(20);
    setUploadError(null);
    setUploadSuccess(null);

    const formData = new FormData();
    formData.append('file', uploadFile);
    if (uploadTitle.trim()) {
      formData.append('title', uploadTitle.trim());
    }

    try {
      setUploadProgress(40);
      const response = await fetch('/api/documents/upload', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      setUploadProgress(80);
      const data: DocumentItem = await response.json();

      if (!response.ok) {
        throw new Error((data as any).detail || 'Upload failed');
      }

      setUploadProgress(100);
      if (data.status === 'completed') {
        setUploadSuccess(`"${data.title}" (${data.page_count} pages) indexed into pgvector!`);
      } else if (data.status === 'failed') {
        setUploadError(`Ingestion notice: ${data.error_message || 'Text extraction failed'}`);
      } else {
        setUploadSuccess(`Document uploaded successfully. Status: ${data.status}`);
      }

      setUploadFile(null);
      setUploadTitle('');
      if (fileInputRef.current) fileInputRef.current.value = '';

      await fetchDocuments();
      setSelectedDoc(data);
    } catch (err: unknown) {
      setUploadError(err instanceof Error ? err.message : 'Failed to upload document.');
    } finally {
      setUploading(false);
      setTimeout(() => setUploadProgress(0), 1000);
    }
  };

  // Semantic Search Handler
  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!token) {
      setShowAuthModal(true);
      return;
    }

    if (!searchQuery.trim()) return;

    setSearching(true);
    setSearchError(null);

    try {
      const payload: { query: string; top_k: number; document_id?: string } = {
        query: searchQuery.trim(),
        top_k: 5,
      };

      if (filterDocId !== 'all') {
        payload.document_id = filterDocId;
      }

      const response = await fetch('/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Search failed');
      }

      const data = await response.json();
      setSearchResults(data.passages || []);
    } catch (err: unknown) {
      setSearchError(err instanceof Error ? err.message : 'Semantic search failed.');
    } finally {
      setSearching(false);
    }
  };

  // Delete Handler
  const handleDeleteDocument = async (docId: string, title: string) => {
    if (!token) return;
    if (!window.confirm(`Are you sure you want to delete "${title}"?`)) return;

    try {
      const response = await fetch(`/api/documents/${docId}`, {
        method: 'DELETE',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to delete document');
      }

      if (selectedDoc?.id === docId) {
        setSelectedDoc(null);
      }
      fetchDocuments();
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Deletion failed');
    }
  };

  // Retry Process Handler
  const handleRetryProcess = async (docId: string) => {
    if (!token) return;
    try {
      const res = await fetch(`/api/documents/${docId}/process`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        fetchDocuments();
      }
    } catch (err) {
      console.error('Failed to retry processing:', err);
    }
  };

  // Filtered document list for sidebar
  const filteredDocs = documents.filter(
    (d) =>
      d.title.toLowerCase().includes(searchFilter.toLowerCase()) ||
      d.filename.toLowerCase().includes(searchFilter.toLowerCase())
  );

  return (
    <div className="min-h-screen flex flex-col bg-[#080c14] text-slate-200">
      {/* Top Header */}
      <header className="h-16 border-b border-slate-800/80 bg-[#0c1220]/80 backdrop-blur-md sticky top-0 z-30 px-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Layers className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg text-white tracking-tight">DocuMind AI</span>
              <span className="text-[11px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                pgvector RAG
              </span>
            </div>
            <p className="text-xs text-slate-400">Document Intelligence & Semantic Retrieval</p>
          </div>
        </div>

        {/* Indicators & User Bar */}
        <div className="flex items-center gap-4">
          {/* Health Pill */}
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-xs">
            <Server className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-400">API:</span>
            {health ? (
              <span className="text-emerald-400 font-medium flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                Active {latencyMs !== null && `(${latencyMs}ms)`}
              </span>
            ) : (
              <span className="text-rose-400 font-medium flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-rose-400"></span>
                Degraded
              </span>
            )}
          </div>

          {/* Database Pill */}
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-xs">
            <Database className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-400">Embeddings:</span>
            <span className="text-cyan-400 font-mono">1536 dim</span>
          </div>

          <button
            onClick={fetchHealth}
            disabled={loadingHealth}
            className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white transition border border-slate-700/60 disabled:opacity-50 cursor-pointer"
            title="Refresh system health"
          >
            <RefreshCw className={`w-4 h-4 ${loadingHealth ? 'animate-spin text-indigo-400' : ''}`} />
          </button>

          {/* User Auth Section */}
          {currentUser ? (
            <div className="flex items-center gap-3 pl-2 border-l border-slate-800">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-indigo-600 to-cyan-600 flex items-center justify-center text-xs font-bold text-white shadow">
                  {currentUser.username.substring(0, 2).toUpperCase()}
                </div>
                <div className="hidden lg:block text-left text-xs">
                  <div className="font-semibold text-slate-200">{currentUser.username}</div>
                  <div className="text-[10px] text-slate-400 truncate max-w-[120px]">{currentUser.email}</div>
                </div>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/90 hover:bg-rose-950/40 hover:text-rose-400 text-slate-300 transition text-xs border border-slate-700/70 cursor-pointer"
              >
                <LogOut className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Sign Out</span>
              </button>
            </div>
          ) : (
            <button
              onClick={() => {
                setAuthMode('login');
                setShowAuthModal(true);
              }}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition shadow-lg shadow-indigo-600/20 cursor-pointer"
            >
              <LogIn className="w-3.5 h-3.5" />
              Sign In / Register
            </button>
          )}
        </div>
      </header>

      {/* Main Workspace */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar: Documents Knowledge Base */}
        <aside className="w-80 border-r border-slate-800/80 bg-[#0a0f1c] flex flex-col justify-between shrink-0">
          <div className="p-4 space-y-4 flex-1 flex flex-col overflow-hidden">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-indigo-400" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">My Knowledge Base</h3>
              </div>
              <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-slate-800 text-slate-400">
                {documents.length}
              </span>
            </div>

            {/* Document Filter Input */}
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-slate-500" />
              <input
                type="text"
                value={searchFilter}
                onChange={(e) => setSearchFilter(e.target.value)}
                placeholder="Filter documents..."
                className="w-full bg-slate-900/60 border border-slate-800/90 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-indigo-500 transition"
              />
            </div>

            {/* Document List */}
            <div className="flex-1 overflow-y-auto space-y-2 pr-1">
              {!currentUser ? (
                <div className="p-6 text-center border border-dashed border-slate-800/80 rounded-xl my-4">
                  <Lock className="w-8 h-8 text-slate-600 mx-auto mb-2" />
                  <p className="text-xs text-slate-400 font-medium mb-1">Authentication Required</p>
                  <p className="text-[11px] text-slate-500 mb-3">
                    Sign in to upload and search your private documents.
                  </p>
                  <button
                    onClick={() => {
                      setAuthMode('login');
                      setShowAuthModal(true);
                    }}
                    className="text-xs text-indigo-400 hover:text-indigo-300 font-semibold underline cursor-pointer"
                  >
                    Sign in now
                  </button>
                </div>
              ) : loadingDocs ? (
                <div className="py-8 text-center text-xs text-slate-500 flex items-center justify-center gap-2">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  Loading documents...
                </div>
              ) : filteredDocs.length === 0 ? (
                <div className="py-8 text-center border border-dashed border-slate-800/80 rounded-xl px-4">
                  <FileText className="w-8 h-8 text-slate-600 mx-auto mb-2" />
                  <p className="text-xs text-slate-400 font-medium">No documents found</p>
                  <p className="text-[11px] text-slate-500 mt-1">
                    Upload an enterprise PDF to begin vector extraction.
                  </p>
                </div>
              ) : (
                filteredDocs.map((doc) => {
                  const isSelected = selectedDoc?.id === doc.id;
                  return (
                    <div
                      key={doc.id}
                      onClick={() => setSelectedDoc(doc)}
                      className={`p-3 rounded-xl border transition cursor-pointer flex flex-col gap-1.5 group ${
                        isSelected
                          ? 'bg-indigo-950/30 border-indigo-500/50 shadow-sm'
                          : 'bg-slate-900/40 border-slate-800/60 hover:bg-slate-900/80 hover:border-slate-700/80'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-start gap-2.5 min-w-0 flex-1">
                          <div className="w-7 h-7 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400 flex items-center justify-center shrink-0 mt-0.5">
                            <FileText className="w-3.5 h-3.5" />
                          </div>
                          <div className="min-w-0 flex-1">
                            <h4 className="text-xs font-semibold text-slate-200 truncate group-hover:text-white">
                              {doc.title}
                            </h4>
                            <div className="flex items-center gap-1.5 text-[10px] text-slate-400 mt-0.5">
                              <span>{doc.page_count}p</span>
                              <span>&bull;</span>
                              <span>{(doc.file_size_bytes / 1024).toFixed(0)} KB</span>
                            </div>
                          </div>
                        </div>

                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteDocument(doc.id, doc.title);
                          }}
                          className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-rose-950/50 text-slate-500 hover:text-rose-400 transition"
                          title="Delete document"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>

                      {/* Status indicator */}
                      <div className="flex items-center justify-between text-[10px]">
                        <span
                          className={`px-1.5 py-0.5 rounded font-mono ${
                            doc.status === 'completed'
                              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                              : doc.status === 'processing'
                              ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 animate-pulse'
                              : doc.status === 'failed'
                              ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                              : 'bg-slate-800 text-slate-400'
                          }`}
                        >
                          {doc.status}
                        </span>

                        {doc.status === 'failed' && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleRetryProcess(doc.id);
                            }}
                            className="text-indigo-400 hover:underline flex items-center gap-1"
                          >
                            <RefreshCw className="w-2.5 h-2.5" /> Retry
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* User Status Bar */}
          <div className="p-4 border-t border-slate-800/80 bg-[#080d17]/80 text-xs text-slate-400 flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              Tenant Isolation
            </span>
            <span className="font-mono text-[11px] text-slate-300">
              {currentUser ? `User #${currentUser.id}` : 'Guest'}
            </span>
          </div>
        </aside>

        {/* Central Workspace */}
        <main className="flex-1 overflow-y-auto p-8 space-y-8 bg-gradient-to-b from-[#090e1a] to-[#080c14]">
          {/* Header Banner */}
          <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-r from-slate-900 via-indigo-950/20 to-slate-900 p-8 shadow-sm">
            <div className="max-w-2xl">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold mb-4">
                <CheckCircle2 className="w-3.5 h-3.5" />
                Stage 3: PDF Ingestion & Semantic Vector Search Active
              </div>
              <h1 className="text-3xl font-bold text-white tracking-tight mb-2">
                Semantic Vector Retrieval Engine
              </h1>
              <p className="text-slate-400 text-sm leading-relaxed mb-6">
                PDFs are parsed per-page, split into overlapping passages, converted into 1536-dimensional embeddings
                (text-embedding-3-small), and retrieved using PostgreSQL pgvector cosine similarity.
              </p>

              <div className="flex items-center gap-3">
                <button
                  onClick={() => setActiveTab('search')}
                  className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition cursor-pointer ${
                    activeTab === 'search'
                      ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                      : 'bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700'
                  }`}
                >
                  <Sparkles className="w-4 h-4 text-cyan-400" />
                  Semantic Search
                </button>
                <button
                  onClick={() => setActiveTab('upload')}
                  className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition cursor-pointer ${
                    activeTab === 'upload'
                      ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                      : 'bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700'
                  }`}
                >
                  <UploadCloud className="w-4 h-4" />
                  Upload Document
                </button>
                <a
                  href="http://localhost:8000/docs"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 text-xs font-semibold border border-slate-700 transition"
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                  Swagger
                </a>
              </div>
            </div>
          </div>

          {/* Document Inspector Card (If a document is selected) */}
          {selectedDoc && (
            <div className="rounded-xl border border-indigo-500/40 bg-gradient-to-br from-indigo-950/20 via-[#0e1424] to-[#0e1424] p-6 shadow-md space-y-4">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/30 text-indigo-400 flex items-center justify-center">
                    <FileCheck2 className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h2 className="text-lg font-bold text-white">{selectedDoc.title}</h2>
                      <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                        {selectedDoc.status}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400">{selectedDoc.filename}</p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => {
                      setFilterDocId(selectedDoc.id);
                      setActiveTab('search');
                    }}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition cursor-pointer"
                  >
                    <Search className="w-3.5 h-3.5" />
                    Search Document
                  </button>
                  <a
                    href={`/api/documents/${selectedDoc.id}/download`}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition"
                  >
                    <Download className="w-3.5 h-3.5 text-indigo-400" />
                    Download PDF
                  </a>
                  <button
                    onClick={() => handleDeleteDocument(selectedDoc.id, selectedDoc.title)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-950/30 hover:bg-rose-900/50 text-rose-300 text-xs font-medium border border-rose-800/40 transition cursor-pointer"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    Delete
                  </button>
                  <button
                    onClick={() => setSelectedDoc(null)}
                    className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {selectedDoc.error_message && (
                <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold">Ingestion notice: </span>
                    {selectedDoc.error_message}
                  </div>
                </div>
              )}

              {/* Metadata Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 rounded-lg bg-[#070b14] border border-slate-800/80 text-xs">
                <div>
                  <span className="text-slate-500 block text-[11px]">Document UUID</span>
                  <span className="font-mono text-slate-300 truncate block mt-0.5">{selectedDoc.id}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[11px]">Page Count</span>
                  <span className="font-semibold text-slate-200 block mt-0.5">{selectedDoc.page_count} Pages</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[11px]">File Size</span>
                  <span className="font-semibold text-slate-200 block mt-0.5">
                    {(selectedDoc.file_size_bytes / 1024).toFixed(1)} KB
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[11px]">Vector Ingestion</span>
                  <span className="text-emerald-400 font-semibold block mt-0.5">
                    {selectedDoc.status === 'completed' ? 'pgvector Indexed' : selectedDoc.status}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* TAB 1: Semantic Search View */}
          {activeTab === 'search' && (
            <div className="rounded-xl border border-slate-800 bg-[#0e1424] p-6 space-y-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-sm font-semibold text-white">Semantic Passage Retrieval (pgvector)</h3>
                </div>
                <span className="text-xs text-slate-400 font-mono">Model: text-embedding-3-small (1536d)</span>
              </div>

              {/* Search Form */}
              <form onSubmit={handleSearch} className="space-y-4">
                <div className="flex flex-col sm:flex-row gap-3">
                  <div className="relative flex-1">
                    <Search className="w-4 h-4 absolute left-3 top-3 text-slate-500" />
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Ask a question or enter conceptual query across your documents..."
                      className="w-full bg-slate-900 border border-slate-800 rounded-xl pl-9 pr-4 py-2.5 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-indigo-500 transition shadow-inner"
                    />
                  </div>

                  {/* Scope filter */}
                  <select
                    value={filterDocId}
                    onChange={(e) => setFilterDocId(e.target.value)}
                    className="bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-300 focus:outline-none focus:border-indigo-500 cursor-pointer"
                  >
                    <option value="all">All Documents ({documents.length})</option>
                    {documents.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.title}
                      </option>
                    ))}
                  </select>

                  <button
                    type="submit"
                    disabled={searching || !searchQuery.trim()}
                    className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-semibold text-xs transition flex items-center justify-center gap-2 cursor-pointer shadow-lg shadow-indigo-600/30"
                  >
                    {searching ? (
                      <>
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                        Retrieving...
                      </>
                    ) : (
                      <>
                        <Search className="w-3.5 h-3.5" />
                        Retrieve Passages
                      </>
                    )}
                  </button>
                </div>
              </form>

              {searchError && (
                <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-400 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>{searchError}</span>
                </div>
              )}

              {/* Search Results Display */}
              {searchResults !== null && (
                <div className="space-y-4 pt-2">
                  <div className="flex items-center justify-between text-xs text-slate-400 border-b border-slate-800 pb-2">
                    <span>
                      Retrieved <span className="text-white font-semibold">{searchResults.length}</span> passages using
                      cosine similarity
                    </span>
                    <span className="font-mono text-[11px] text-slate-500">Query: "{searchQuery}"</span>
                  </div>

                  {searchResults.length === 0 ? (
                    <div className="p-8 text-center border border-dashed border-slate-800 rounded-xl">
                      <p className="text-xs text-slate-400">No matching passages found for this query.</p>
                      <p className="text-[11px] text-slate-500 mt-1">
                        Try broader search terms or upload additional documents.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {searchResults.map((p, idx) => (
                        <div
                          key={p.chunk_id}
                          className="p-4 rounded-xl border border-slate-800/80 bg-[#090e1a] hover:border-indigo-500/40 transition space-y-2.5"
                        >
                          <div className="flex items-center justify-between text-xs">
                            <div className="flex items-center gap-2">
                              <span className="w-5 h-5 rounded-full bg-indigo-600/20 text-indigo-400 font-mono text-[11px] flex items-center justify-center font-bold">
                                {idx + 1}
                              </span>
                              <span className="font-semibold text-slate-200">{p.document_title}</span>
                              <span className="px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-semibold text-[10px]">
                                Page {p.page_number}
                              </span>
                            </div>
                            <span className="font-mono text-[11px] text-emerald-400 font-semibold">
                              {(p.score * 100).toFixed(1)}% match
                            </span>
                          </div>

                          <div className="flex items-start gap-2.5 text-xs text-slate-300 leading-relaxed bg-[#060a12] p-3 rounded-lg border border-slate-800/60 font-sans">
                            <Quote className="w-3.5 h-3.5 text-indigo-400 shrink-0 mt-0.5 opacity-60" />
                            <p>{p.text}</p>
                          </div>

                          <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono">
                            <span>Chunk ID: {p.chunk_id.substring(0, 8)}...</span>
                            <span>{p.text.length} characters &bull; 1536-dim vector</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* TAB 2: Upload Drop Zone & Form */}
          {activeTab === 'upload' && (
            <div className="rounded-xl border border-slate-800 bg-[#0e1424] p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <UploadCloud className="w-4 h-4 text-indigo-400" />
                  <h3 className="text-sm font-semibold text-white">Upload & Ingest Enterprise PDF</h3>
                </div>
                <span className="text-xs text-slate-500">Max limit: 15MB &bull; Automatic chunking & pgvector index</span>
              </div>

              {uploadError && (
                <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-400 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>{uploadError}</span>
                </div>
              )}

              {uploadSuccess && (
                <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-xs text-emerald-400 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 shrink-0" />
                  <span>{uploadSuccess}</span>
                </div>
              )}

              <form onSubmit={handleFileUpload} className="space-y-4">
                <input
                  type="file"
                  ref={fileInputRef}
                  accept=".pdf,application/pdf"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      setUploadFile(e.target.files[0]);
                      if (!uploadTitle) {
                        setUploadTitle(e.target.files[0].name.replace(/\.[^/.]+$/, ''));
                      }
                    }
                  }}
                  className="hidden"
                />

                <div
                  onClick={() => {
                    if (fileInputRef.current) fileInputRef.current.click();
                  }}
                  className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition ${
                    uploadFile
                      ? 'border-indigo-500/60 bg-indigo-950/10'
                      : 'border-slate-800 hover:border-indigo-500/40 hover:bg-slate-900/40'
                  }`}
                >
                  <UploadCloud className="w-10 h-10 text-indigo-400/80 mx-auto mb-2" />
                  {uploadFile ? (
                    <div className="space-y-1">
                      <p className="text-sm font-semibold text-white">{uploadFile.name}</p>
                      <p className="text-xs text-indigo-400">
                        {(uploadFile.size / (1024 * 1024)).toFixed(2)} MB &bull; Click to change file
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-1">
                      <p className="text-sm font-medium text-slate-300">
                        Drag and drop your PDF here, or <span className="text-indigo-400 underline">browse files</span>
                      </p>
                      <p className="text-xs text-slate-500">
                        Files are cryptographically verified for PDF magic headers (%PDF-).
                      </p>
                    </div>
                  )}
                </div>

                {uploadFile && (
                  <div className="flex flex-col sm:flex-row gap-3">
                    <input
                      type="text"
                      value={uploadTitle}
                      onChange={(e) => setUploadTitle(e.target.value)}
                      placeholder="Document Title (e.g. Corporate Security Guidelines)"
                      className="flex-1 bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-indigo-500"
                    />
                    <button
                      type="submit"
                      disabled={uploading}
                      className="px-5 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-semibold text-xs transition flex items-center justify-center gap-2 cursor-pointer shadow-md shadow-indigo-600/20"
                    >
                      {uploading ? (
                        <>
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          Indexing ({uploadProgress}%)...
                        </>
                      ) : (
                        <>
                          <UploadCloud className="w-3.5 h-3.5" />
                          Upload & Index
                        </>
                      )}
                    </button>
                  </div>
                )}

                {uploading && (
                  <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                    <div
                      className="bg-indigo-500 h-1.5 rounded-full transition-all duration-300"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                )}
              </form>
            </div>
          )}

          {/* Implementation Roadmap */}
          <div className="rounded-xl border border-slate-800 bg-[#0e1424] p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-semibold text-slate-200">DocuMind AI Architecture Roadmap</h3>
              </div>
              <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                Portfolio Lifecycle
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
              <div className="p-3.5 rounded-lg bg-emerald-500/5 border border-emerald-500/20 flex items-start gap-3">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-emerald-300 mb-0.5">Stage 1: Foundation (Completed)</h4>
                  <p className="text-slate-400 text-[11px] leading-relaxed">
                    React 19 + TypeScript, FastAPI backend, SQLAlchemy pool, Docker pgvector, and /health telemetry.
                  </p>
                </div>
              </div>

              <div className="p-3.5 rounded-lg bg-emerald-500/5 border border-emerald-500/20 flex items-start gap-3">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-emerald-300 mb-0.5">Stage 2: Auth & Documents (Completed)</h4>
                  <p className="text-slate-400 text-[11px] leading-relaxed">
                    Bcrypt hashing, JWT tokens, deep PDF validation (%PDF-), 15MB upload limits, and owner-authorized CRUD.
                  </p>
                </div>
              </div>

              <div className="p-3.5 rounded-lg bg-emerald-500/5 border border-emerald-500/20 flex items-start gap-3">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-emerald-300 mb-0.5">Stage 3: PDF Ingestion & Semantic Retrieval (Active)</h4>
                  <p className="text-slate-400 text-[11px] leading-relaxed">
                    Page-aware chunking, text-embedding-3-small (1536 dim), pgvector cosine search, and cross-user isolation.
                  </p>
                </div>
              </div>

              <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800/80 flex items-start gap-3 opacity-80">
                <CheckCircle2 className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-slate-300 mb-0.5">Stage 4: Grounded RAG with Citations</h4>
                  <p className="text-slate-400 text-[11px] leading-relaxed">
                    Evidence-grounded synthesis, page-level citation extraction, and non-hallucination fallbacks.
                  </p>
                </div>
              </div>

              <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800/80 flex items-start gap-3 opacity-80">
                <CheckCircle2 className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-slate-300 mb-0.5">Stage 5: Access Control & Governance</h4>
                  <p className="text-slate-400 text-[11px] leading-relaxed">
                    Granular document-level query filters and multi-user tenant boundaries.
                  </p>
                </div>
              </div>

              <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800/80 flex items-start gap-3 opacity-80">
                <CheckCircle2 className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-slate-300 mb-0.5">Stage 6: Evaluation & Observability</h4>
                  <p className="text-slate-400 text-[11px] leading-relaxed">
                    RAGAS metrics, token consumption, cost estimation, and latency telemetry.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* Authentication Modal */}
      {showAuthModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="w-full max-w-md rounded-2xl border border-slate-800 bg-[#0d1322] p-6 shadow-2xl relative space-y-5">
            <button
              onClick={() => setShowAuthModal(false)}
              className="absolute top-4 right-4 text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="text-center space-y-1">
              <div className="w-10 h-10 rounded-xl bg-indigo-600/20 border border-indigo-500/30 text-indigo-400 flex items-center justify-center mx-auto mb-2">
                {authMode === 'login' ? <LogIn className="w-5 h-5" /> : <UserPlus className="w-5 h-5" />}
              </div>
              <h3 className="text-lg font-bold text-white">
                {authMode === 'login' ? 'Welcome Back' : 'Create an Account'}
              </h3>
              <p className="text-xs text-slate-400">
                {authMode === 'login'
                  ? 'Sign in to access your private document intelligence vault.'
                  : 'Register to start indexing and querying enterprise documents.'}
              </p>
            </div>

            {/* Auth Mode Tabs */}
            <div className="flex rounded-lg bg-slate-900 p-1 border border-slate-800">
              <button
                type="button"
                onClick={() => {
                  setAuthMode('login');
                  setAuthError(null);
                }}
                className={`flex-1 py-1.5 text-xs font-semibold rounded-md transition ${
                  authMode === 'login' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Sign In
              </button>
              <button
                type="button"
                onClick={() => {
                  setAuthMode('register');
                  setAuthError(null);
                }}
                className={`flex-1 py-1.5 text-xs font-semibold rounded-md transition ${
                  authMode === 'register' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Register
              </button>
            </div>

            {authError && (
              <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-400 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>{authError}</span>
              </div>
            )}

            <form onSubmit={handleAuthSubmit} className="space-y-3.5">
              {authMode === 'register' && (
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">Email Address</label>
                  <div className="relative">
                    <Mail className="w-4 h-4 absolute left-3 top-2.5 text-slate-500" />
                    <input
                      type="email"
                      required
                      value={authEmail}
                      onChange={(e) => setAuthEmail(e.target.value)}
                      placeholder="analyst@enterprise.com"
                      className="w-full bg-slate-900/80 border border-slate-800 rounded-lg pl-9 pr-3 py-2 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-indigo-500"
                    />
                  </div>
                </div>
              )}

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  {authMode === 'login' ? 'Username or Email' : 'Username'}
                </label>
                <div className="relative">
                  <UserIcon className="w-4 h-4 absolute left-3 top-2.5 text-slate-500" />
                  <input
                    type="text"
                    required
                    value={authUsername}
                    onChange={(e) => setAuthUsername(e.target.value)}
                    placeholder={authMode === 'login' ? 'analyst or analyst@enterprise.com' : 'doc_analyst'}
                    className="w-full bg-slate-900/80 border border-slate-800 rounded-lg pl-9 pr-3 py-2 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-indigo-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Password</label>
                <div className="relative">
                  <Lock className="w-4 h-4 absolute left-3 top-2.5 text-slate-500" />
                  <input
                    type="password"
                    required
                    minLength={8}
                    value={authPassword}
                    onChange={(e) => setAuthPassword(e.target.value)}
                    placeholder="Min 8 characters"
                    className="w-full bg-slate-900/80 border border-slate-800 rounded-lg pl-9 pr-3 py-2 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-indigo-500"
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={authLoading}
                className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-semibold text-xs transition flex items-center justify-center gap-2 cursor-pointer shadow-lg shadow-indigo-600/30"
              >
                {authLoading ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    Authenticating...
                  </>
                ) : authMode === 'login' ? (
                  'Sign In to DocuMind'
                ) : (
                  'Create Secure Account'
                )}
              </button>
            </form>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="h-10 border-t border-slate-800/80 bg-[#0a0f1c] px-6 flex items-center justify-between text-[11px] text-slate-500">
        <div>DocuMind AI &bull; Forward Deployed AI Engineer Portfolio Project</div>
        <div className="flex items-center gap-4">
          <span>React 19 &bull; Vite &bull; FastAPI &bull; PostgreSQL (pgvector 1536d)</span>
        </div>
      </footer>
    </div>
  );
}
