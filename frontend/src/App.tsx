import React, { useState, useEffect, useRef } from 'react';
import {
  Database,
  Server,
  ShieldCheck,
  FileText,
  Layers,
  RefreshCw,
  CheckCircle2,
  ExternalLink,
  Search,
  UploadCloud,
  Trash2,
  Download,
  LogOut,
  LogIn,
  Lock,
  Mail,
  User as UserIcon,
  X,
  AlertTriangle,
  Sparkles,
  Quote,
  Activity,
  Send,
  Copy,
  Check,
  BookOpen,
  RotateCcw,
  Menu,
  MessageSquare,
  Plus,
  PanelLeft,
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
  is_admin?: boolean;
}

interface RequestMetricItem {
  id: number;
  request_id: string;
  timestamp: string;
  user_id: number;
  document_count: number;
  retrieval_latency_ms: number;
  generation_latency_ms: number;
  total_latency_ms: number;
  status: string;
  error_category: string | null;
  safe_message: string | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  estimated_cost_usd: number | null;
  pricing_version: string | null;
}

interface MonitoringDashboardData {
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  error_rate: number;
  avg_total_latency_ms: number;
  avg_retrieval_latency_ms: number;
  avg_generation_latency_ms: number;
  p95_total_latency_ms: number;
  token_usage: {
    total_prompt_tokens: number | null;
    total_completion_tokens: number | null;
    total_tokens: number | null;
    requests_with_unknown_tokens: number;
    known_token_requests: number;
    unknown_token_requests: number;
    is_partial: boolean;
  };
  total_estimated_cost_usd: number | null;
  cost_is_partial: boolean;
  known_cost_requests: number;
  unknown_cost_requests: number;
  pricing_version: string;
  recent_requests: RequestMetricItem[];
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

export interface CitationItem {
  chunk_id: string;
  document_id: string;
  document_title: string;
  page_number: number;
  excerpt: string;
  score: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  insufficient_evidence?: boolean;
  citations?: CitationItem[];
  model_used?: string;
  retrieval_count?: number;
  retrieval_latency_ms?: number;
  generation_latency_ms?: number;
  total_latency_ms?: number;
  token_usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  estimated_cost_usd?: number | null;
  error?: string;
}

export interface ConversationItem {
  id: string;
  user_id: number;
  document_id: string | null;
  title: string;
  document_scope_invalidated: boolean;
  created_at: string;
  updated_at: string;
  messages?: ChatMessage[];
  latest_message?: ChatMessage | null;
}

const SUGGESTED_QUESTIONS = [
  'What are the core security policies and compliance requirements?',
  'Summarize the financial performance and key metrics reported.',
  'What is the incident response protocol and reporting timeline?',
  'What encryption standards are enforced for data at rest and in transit?',
];

export default function App() {
  // Navigation: ask (Ask DocuMind), documents (Document KB), search (Semantic Search), upload, monitoring
  const [activeTab, setActiveTab] = useState<'ask' | 'documents' | 'search' | 'upload' | 'monitoring'>(() => {
    const saved = localStorage.getItem('documind_active_tab');
    if (saved && ['ask', 'documents', 'search', 'upload', 'monitoring'].includes(saved)) {
      return saved as 'ask' | 'documents' | 'search' | 'upload' | 'monitoring';
    }
    return 'search';
  });
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    localStorage.setItem('documind_active_tab', activeTab);
  }, [activeTab]);

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
  const [searchQuery, setSearchQuery] = useState('');
  const [filterDocId, setFilterDocId] = useState<string>('all');
  const [searching, setSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchPassage[] | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Monitoring state
  const [monitoringData, setMonitoringData] = useState<MonitoringDashboardData | null>(null);
  const [loadingMonitoring, setLoadingMonitoring] = useState<boolean>(false);
  const [monitoringError, setMonitoringError] = useState<string | null>(null);

  // Authenticated PDF Download state (SEC-02)
  const [downloadingDocId, setDownloadingDocId] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  // Ask DocuMind RAG Conversation state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [askInput, setAskInput] = useState<string>('');
  const [askingRAG, setAskingRAG] = useState<boolean>(false);
  const [ragError, setRagError] = useState<string | null>(null);
  const [activeCitation, setActiveCitation] = useState<CitationItem | null>(null);
  const [showSourcePanel, setShowSourcePanel] = useState<boolean>(true);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [copiedCitationsId, setCopiedCitationsId] = useState<string | null>(null);
  const [openingPdfPage, setOpeningPdfPage] = useState<boolean>(false);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const askInputRef = useRef<HTMLTextAreaElement>(null);

  // Persistent Conversation state (Checkpoint 13 / Checkpoint B)
  const [conversations, setConversations] = useState<ConversationItem[]>([]);
  const [loadingConversations, setLoadingConversations] = useState<boolean>(false);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(() => {
    return localStorage.getItem('documind_active_conversation_id');
  });
  const [loadingHistory, setLoadingHistory] = useState<boolean>(false);
  const [conversationError, setConversationError] = useState<string | null>(null);
  const [showConversationSidebar, setShowConversationSidebar] = useState<boolean>(true);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [isCreatingConv, setIsCreatingConv] = useState<boolean>(false);
  const activeConvRequestRef = useRef<string | null>(null);
  const historyRequestSeqRef = useRef<number>(0);

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

  // Monitoring fetcher
  const fetchMonitoring = async () => {
    if (!token) {
      setMonitoringError('Please sign in with an admin account to view monitoring telemetry.');
      return;
    }
    setLoadingMonitoring(true);
    setMonitoringError(null);
    try {
      const response = await fetch('/api/monitoring/dashboard', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.status === 403) {
        setMonitoringError('Access denied: Admin privileges required to view observability metrics.');
        setMonitoringData(null);
        return;
      }

      if (response.status === 401) {
        handleLogout();
        setShowAuthModal(true);
        return;
      }

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to fetch monitoring metrics');
      }

      const data: MonitoringDashboardData = await response.json();
      setMonitoringData(data);
    } catch (err: unknown) {
      setMonitoringError(err instanceof Error ? err.message : 'Failed to load monitoring dashboard');
    } finally {
      setLoadingMonitoring(false);
    }
  };

  // Load Conversation History
  const loadConversationHistory = async (convId: string, authToken?: string) => {
    const activeToken = authToken || token;
    if (!activeToken || !convId) return;

    activeConvRequestRef.current = convId;
    historyRequestSeqRef.current += 1;
    const reqSeq = historyRequestSeqRef.current;

    setLoadingHistory(true);
    setActiveConversationId(convId);
    localStorage.setItem('documind_active_conversation_id', convId);

    try {
      const response = await fetch(`/api/conversations/${convId}`, {
        headers: {
          Authorization: `Bearer ${activeToken}`,
        },
      });

      if (response.status === 401) {
        handleLogout();
        setShowAuthModal(true);
        return;
      }

      if (response.status === 404) {
        localStorage.removeItem('documind_active_conversation_id');
        setActiveConversationId(null);
        setChatMessages([]);
        fetchConversations(activeToken);
        return;
      }

      if (response.ok) {
        const data = await response.json();
        // Guard against stale response if user switched to another conversation
        if (activeConvRequestRef.current !== convId || historyRequestSeqRef.current !== reqSeq) {
          return;
        }

        const rawMessages = data.messages || [];
        const mappedMessages: ChatMessage[] = rawMessages.map((m: any) => ({
          id: m.id,
          role: m.role as 'user' | 'assistant',
          content: m.content,
          timestamp: m.created_at,
          insufficient_evidence: m.insufficient_evidence,
          citations: (m.citations as CitationItem[]) || [],
          model_used: m.model_used,
          retrieval_count: m.citations ? m.citations.length : 0,
          retrieval_latency_ms: m.retrieval_latency_ms,
          generation_latency_ms: m.generation_latency_ms,
          total_latency_ms: m.total_latency_ms,
          token_usage: m.token_usage,
          estimated_cost_usd: m.estimated_cost_usd,
          error: m.status === 'failed' ? (m.error_message || 'Generation failed') : undefined,
        }));

        setChatMessages(mappedMessages);

        if (data.document_id) {
          setFilterDocId(data.document_id);
        } else {
          setFilterDocId('all');
        }

        const lastAssistantMsg = [...mappedMessages]
          .reverse()
          .find((m) => m.role === 'assistant' && m.citations && m.citations.length > 0);
        if (lastAssistantMsg && lastAssistantMsg.citations && lastAssistantMsg.citations.length > 0) {
          setActiveCitation(lastAssistantMsg.citations[0]);
        } else {
          setActiveCitation(null);
        }
      }
    } catch (err) {
      console.error('Failed to load conversation history:', err);
    } finally {
      if (historyRequestSeqRef.current === reqSeq) {
        setLoadingHistory(false);
      }
    }
  };

  // Conversations fetcher
  const fetchConversations = async (authToken?: string) => {
    const activeToken = authToken || token;
    if (!activeToken) {
      setConversations([]);
      return;
    }

    setLoadingConversations(true);
    setConversationError(null);
    try {
      const response = await fetch('/api/conversations', {
        headers: {
          Authorization: `Bearer ${activeToken}`,
        },
      });

      if (response.status === 401) {
        handleLogout();
        setShowAuthModal(true);
        return;
      }

      if (response.ok) {
        const data: { total: number; conversations: ConversationItem[] } = await response.json();
        const convList = data.conversations || [];
        setConversations(convList);

        const savedId = localStorage.getItem('documind_active_conversation_id');
        if (savedId && convList.some((c) => c.id === savedId)) {
          setActiveConversationId(savedId);
          loadConversationHistory(savedId, activeToken);
        } else if (convList.length > 0) {
          const firstId = convList[0].id;
          setActiveConversationId(firstId);
          localStorage.setItem('documind_active_conversation_id', firstId);
          loadConversationHistory(firstId, activeToken);
        } else {
          setActiveConversationId(null);
          localStorage.removeItem('documind_active_conversation_id');
          setChatMessages([]);
        }
      }
    } catch (err) {
      console.error('Failed to load conversations:', err);
      setConversationError('Unable to load conversations.');
    } finally {
      setLoadingConversations(false);
    }
  };

  // Create Conversation Handler
  const handleCreateConversation = async (docId?: string, title?: string): Promise<string | null> => {
    if (!token) {
      setShowAuthModal(true);
      return null;
    }

    setIsCreatingConv(true);
    setRagError(null);
    try {
      const scopeDocId = docId !== undefined ? docId : filterDocId;
      const payload: { title: string; document_id?: string | null } = {
        title: (title || 'New Conversation').slice(0, 255),
        document_id: (scopeDocId && scopeDocId !== 'all') ? scopeDocId : null,
      };

      const response = await fetch('/api/conversations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });

      if (response.status === 401) {
        handleLogout();
        setShowAuthModal(true);
        return null;
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to create conversation');
      }

      const newConv: ConversationItem = await response.json();
      setConversations((prev) => [newConv, ...prev.filter((c) => c.id !== newConv.id)]);
      setActiveConversationId(newConv.id);
      activeConvRequestRef.current = newConv.id;
      localStorage.setItem('documind_active_conversation_id', newConv.id);
      setChatMessages([]);
      setActiveCitation(null);
      if (askInputRef.current) askInputRef.current.focus();
      return newConv.id;
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to create conversation';
      setRagError(msg);
      return null;
    } finally {
      setIsCreatingConv(false);
    }
  };

  // Delete Conversation Handler
  const handleDeleteConversation = async (convId: string) => {
    if (!token) return;

    try {
      const response = await fetch(`/api/conversations/${convId}`, {
        method: 'DELETE',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.status === 401) {
        handleLogout();
        setShowAuthModal(true);
        return;
      }

      if (response.ok) {
        const updatedList = conversations.filter((c) => c.id !== convId);
        setConversations(updatedList);
        setDeleteConfirmId(null);

        if (activeConversationId === convId) {
          if (updatedList.length > 0) {
            const nextConvId = updatedList[0].id;
            setActiveConversationId(nextConvId);
            localStorage.setItem('documind_active_conversation_id', nextConvId);
            loadConversationHistory(nextConvId);
          } else {
            setActiveConversationId(null);
            localStorage.removeItem('documind_active_conversation_id');
            setChatMessages([]);
            setActiveCitation(null);
          }
        }
      }
    } catch (err) {
      console.error('Failed to delete conversation:', err);
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
      fetchConversations();
    } else {
      setDocuments([]);
      setSelectedDoc(null);
      setSearchResults(null);
      setChatMessages([]);
      setConversations([]);
      setActiveConversationId(null);
      localStorage.removeItem('documind_active_conversation_id');
    }
  }, [token]);

  useEffect(() => {
    if (activeTab === 'ask' && token) {
      if (conversations.length === 0 && !loadingConversations) {
        fetchConversations();
      }
    }
  }, [activeTab, token]);

  // Scroll to bottom of chat when new messages arrive
  useEffect(() => {
    if (activeTab === 'ask' && chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [chatMessages, askingRAG, activeTab]);

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
      fetchConversations(data.access_token);
    } catch (err: unknown) {
      setAuthError(err instanceof Error ? err.message : 'An error occurred during authentication.');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('documind_token');
    localStorage.removeItem('documind_user');
    localStorage.removeItem('documind_active_conversation_id');
    setToken(null);
    setCurrentUser(null);
    setDocuments([]);
    setSelectedDoc(null);
    setSearchResults(null);
    setChatMessages([]);
    setActiveCitation(null);
    setConversations([]);
    setActiveConversationId(null);
    activeConvRequestRef.current = null;
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

  // Authenticated PDF Download Handler (SEC-02)
  const handleDownloadDocument = async (docId: string, filename: string) => {
    if (!token) {
      setDownloadError('Authentication required to download documents.');
      setShowAuthModal(true);
      return;
    }

    setDownloadingDocId(docId);
    setDownloadError(null);

    try {
      const response = await fetch(`/api/documents/${docId}/download`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.status === 401) {
        handleLogout();
        setShowAuthModal(true);
        throw new Error('Your session has expired. Please log in again.');
      }

      if (response.status === 403) {
        throw new Error('Access denied: You are not authorized to download this document.');
      }

      if (response.status === 404) {
        throw new Error('Document file not found on server.');
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Download failed with HTTP ${response.status}`);
      }

      const blob = await response.blob();
      const blobUrl = window.URL.createObjectURL(blob);
      const tempLink = document.createElement('a');
      tempLink.href = blobUrl;
      tempLink.download = filename || 'document.pdf';
      document.body.appendChild(tempLink);
      tempLink.click();
      document.body.removeChild(tempLink);
      window.URL.revokeObjectURL(blobUrl);
    } catch (err: unknown) {
      setDownloadError(err instanceof Error ? err.message : 'Failed to download document');
    } finally {
      setDownloadingDocId(null);
    }
  };

  // Authenticated PDF View at Cited Page Handler
  const handleOpenPdfAtPage = async (docId: string, pageNumber: number) => {
    if (!token) {
      setShowAuthModal(true);
      return;
    }

    setOpeningPdfPage(true);
    try {
      const response = await fetch(`/api/documents/${docId}/download`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.status === 401) {
        handleLogout();
        setShowAuthModal(true);
        return;
      }

      if (!response.ok) {
        throw new Error('Failed to load document for viewing.');
      }

      const blob = await response.blob();
      const blobUrl = window.URL.createObjectURL(blob);
      // Open in browser at the designated PDF fragment page
      window.open(`${blobUrl}#page=${pageNumber}`, '_blank');
      // Keep blob URL alive briefly for browser tab rendering
      setTimeout(() => window.URL.revokeObjectURL(blobUrl), 60000);
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Could not open cited PDF page');
    } finally {
      setOpeningPdfPage(false);
    }
  };

  // Ask DocuMind Persistent Conversation Submission Handler
  const handleAskQuestion = async (customQuestion?: string) => {
    const questionText = (customQuestion || askInput).trim();
    if (!questionText || askingRAG) return;

    if (!token) {
      setShowAuthModal(true);
      return;
    }

    if (questionText.length < 2) {
      setRagError('Question must contain at least 2 characters.');
      return;
    }

    // Ensure we have an active conversation ID, or create one immediately
    let targetConvId = activeConversationId;
    if (!targetConvId) {
      const createdId = await handleCreateConversation(filterDocId, questionText.slice(0, 30));
      if (!createdId) return;
      targetConvId = createdId;
    }

    // Verify active conversation document scope validity
    const currentConv = conversations.find((c) => c.id === targetConvId);
    if (currentConv && currentConv.document_scope_invalidated) {
      setRagError("This conversation's document scope was invalidated because the document was deleted. Please start a new conversation.");
      return;
    }

    // Append user question to conversation optimistically
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: questionText,
      timestamp: new Date().toISOString(),
    };

    setChatMessages((prev) => [...prev, userMsg]);
    setAskInput('');
    setAskingRAG(true);
    setRagError(null);

    const sendingConvId = targetConvId;

    try {
      const payload = {
        content: questionText,
        top_k: 4,
      };

      const response = await fetch(`/api/conversations/${targetConvId}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });

      if (response.status === 401) {
        handleLogout();
        setShowAuthModal(true);
        throw new Error('Your session has expired. Please log in again.');
      }

      if (response.status === 404) {
        throw new Error('Conversation not found or access denied.');
      }

      if (response.status === 409) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Conversation document scope has been invalidated or another generation is in-flight.');
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Query failed with HTTP ${response.status}`);
      }

      const data = await response.json();

      const assistantMsg: ChatMessage = {
        id: data.id || data.request_id || `asst-${Date.now()}`,
        role: 'assistant',
        content: data.content || data.answer || '',
        timestamp: data.created_at || new Date().toISOString(),
        insufficient_evidence: data.insufficient_evidence ?? false,
        citations: (data.citations as CitationItem[]) || [],
        model_used: data.model_used || 'gemini-2.5-flash',
        retrieval_count: data.citations ? data.citations.length : (data.retrieval_count || 0),
        retrieval_latency_ms: data.retrieval_latency_ms,
        generation_latency_ms: data.generation_latency_ms,
        total_latency_ms: data.total_latency_ms,
        token_usage: data.token_usage,
        estimated_cost_usd: data.estimated_cost_usd,
      };

      // Guard: only append to active message stream if user is still on this conversation
      if (activeConvRequestRef.current === sendingConvId) {
        setChatMessages((prev) => [...prev, assistantMsg]);
        if (assistantMsg.citations && assistantMsg.citations.length > 0) {
          setActiveCitation(assistantMsg.citations[0]);
          setShowSourcePanel(true);
        }
      }

      // Update sidebar conversation item's latest_message & updated_at
      setConversations((prev) =>
        prev.map((c) =>
          c.id === sendingConvId
            ? {
                ...c,
                updated_at: new Date().toISOString(),
                latest_message: assistantMsg,
              }
            : c
        )
      );
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to generate answer';
      setRagError(msg);
      // Guard: only add error message card if user is still on this conversation
      if (activeConvRequestRef.current === sendingConvId) {
        const errorMsg: ChatMessage = {
          id: `err-${Date.now()}`,
          role: 'assistant',
          content: 'I was unable to complete the request.',
          timestamp: new Date().toISOString(),
          error: msg,
        };
        setChatMessages((prev) => [...prev, errorMsg]);
      }
    } finally {
      setAskingRAG(false);
    }
  };

  const handleNewConversation = () => {
    handleCreateConversation();
  };

  const handleClearConversation = () => {
    setChatMessages([]);
    setActiveCitation(null);
    setRagError(null);
  };

  const handleCopyAnswer = (msg: ChatMessage) => {
    navigator.clipboard.writeText(msg.content);
    setCopiedMessageId(msg.id);
    setTimeout(() => setCopiedMessageId(null), 2000);
  };

  const handleCopyCitations = (msg: ChatMessage) => {
    if (!msg.citations || msg.citations.length === 0) return;
    const formatted = msg.citations
      .map(
        (c, idx) =>
          `[${idx + 1}] ${c.document_title} (Page ${c.page_number}):\n"${c.excerpt}"`
      )
      .join('\n\n');
    navigator.clipboard.writeText(formatted);
    setCopiedCitationsId(msg.id);
    setTimeout(() => setCopiedCitationsId(null), 2000);
  };

  // Filtered document list for sidebar & selector
  const filteredDocs = documents.filter(
    (d) =>
      d.title.toLowerCase().includes(searchFilter.toLowerCase()) ||
      d.filename.toLowerCase().includes(searchFilter.toLowerCase())
  );

  const completedDocs = documents.filter((d) => d.status === 'completed');

  return (
    <div className="min-h-screen flex flex-col bg-[#080c14] text-slate-200">
      {/* Top Header Bar */}
      <header className="h-16 border-b border-slate-800/80 bg-[#0c1220]/90 backdrop-blur-md sticky top-0 z-30 px-4 sm:px-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 rounded-lg bg-slate-800 text-slate-300 hover:text-white"
            title="Toggle Menu"
          >
            <Menu className="w-5 h-5" />
          </button>

          <div
            onClick={() => setActiveTab('ask')}
            className="flex items-center gap-3 cursor-pointer group"
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-indigo-500/20 group-hover:scale-105 transition">
              <Layers className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg text-white tracking-tight">DocuMind AI</span>
                <span className="text-[11px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  pgvector RAG
                </span>
              </div>
              <p className="text-xs text-slate-400">Enterprise Document Intelligence & Grounded Q&A</p>
            </div>
          </div>
        </div>

        {/* Indicators & User Bar */}
        <div className="flex items-center gap-3 sm:gap-4">
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
                  <div className="font-semibold text-slate-200 flex items-center gap-1.5">
                    {currentUser.username}
                    {currentUser.is_admin && (
                      <span className="text-[9px] uppercase px-1 py-0.2 bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 rounded font-semibold">
                        Admin
                      </span>
                    )}
                  </div>
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

      {/* Main Workspace with Three-Part Layout */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Left Sidebar Navigation */}
        <aside
          className={`w-64 border-r border-slate-800/80 bg-[#0a0f1c] flex flex-col justify-between shrink-0 z-20 transition-all duration-300 ${
            mobileMenuOpen ? 'absolute inset-y-0 left-0 shadow-2xl flex' : 'hidden md:flex'
          }`}
        >
          <div className="p-4 space-y-6 flex-1 flex flex-col overflow-y-auto">
            {/* Primary Navigation Menu */}
            <div className="space-y-1.5">
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 px-3">
                Workspace
              </span>

              <button
                onClick={() => {
                  setActiveTab('ask');
                  setMobileMenuOpen(false);
                }}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-semibold transition cursor-pointer ${
                  activeTab === 'ask'
                    ? 'bg-gradient-to-r from-indigo-600/20 to-cyan-600/20 border border-indigo-500/40 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <Sparkles className="w-4 h-4 text-cyan-400" />
                  <span>Ask DocuMind</span>
                </div>
                <span className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                  RAG
                </span>
              </button>

              <button
                onClick={() => {
                  setActiveTab('documents');
                  setMobileMenuOpen(false);
                }}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-semibold transition cursor-pointer ${
                  activeTab === 'documents' || activeTab === 'upload'
                    ? 'bg-slate-800 border border-slate-700 text-white'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <FileText className="w-4 h-4 text-indigo-400" />
                  <span>Documents</span>
                </div>
                <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-slate-800 text-slate-400">
                  {documents.length}
                </span>
              </button>

              <button
                onClick={() => {
                  setActiveTab('search');
                  setMobileMenuOpen(false);
                }}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-semibold transition cursor-pointer ${
                  activeTab === 'search'
                    ? 'bg-slate-800 border border-slate-700 text-white'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <Search className="w-4 h-4 text-slate-400" />
                  <span>Semantic Search</span>
                </div>
              </button>

              <button
                onClick={() => {
                  setActiveTab('monitoring');
                  fetchMonitoring();
                  setMobileMenuOpen(false);
                }}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-semibold transition cursor-pointer ${
                  activeTab === 'monitoring'
                    ? 'bg-slate-800 border border-slate-700 text-white'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <Activity className="w-4 h-4 text-emerald-400" />
                  <span>System Telemetry</span>
                </div>
                <span className="text-[10px] font-mono text-slate-500">Admin</span>
              </button>
            </div>

            {/* Document Knowledge Base Quick List */}
            <div className="pt-2 border-t border-slate-800/80 flex-1 flex flex-col space-y-2.5">
              <div className="flex items-center justify-between px-1">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                  Knowledge Base
                </span>
                <button
                  onClick={() => setActiveTab('upload')}
                  className="text-[11px] text-indigo-400 hover:text-indigo-300 font-medium flex items-center gap-1 cursor-pointer"
                  title="Upload new document"
                >
                  <UploadCloud className="w-3 h-3" />
                  <span>Add PDF</span>
                </button>
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

              {/* Sidebar Document List */}
              <div className="flex-1 overflow-y-auto space-y-1.5 pr-1 max-h-56">
                {!currentUser ? (
                  <div className="p-4 text-center border border-dashed border-slate-800/80 rounded-xl my-2">
                    <Lock className="w-6 h-6 text-slate-600 mx-auto mb-1.5" />
                    <p className="text-xs text-slate-400 font-medium">Auth Required</p>
                    <p className="text-[10px] text-slate-500 mt-0.5">
                      Sign in to index documents.
                    </p>
                  </div>
                ) : loadingDocs ? (
                  <div className="py-4 text-center text-xs text-slate-500 flex items-center justify-center gap-2">
                    <RefreshCw className="w-3 h-3 animate-spin" />
                    Loading...
                  </div>
                ) : filteredDocs.length === 0 ? (
                  <div className="py-4 text-center border border-dashed border-slate-800/80 rounded-xl px-2">
                    <p className="text-xs text-slate-500">No documents</p>
                  </div>
                ) : (
                  filteredDocs.map((doc) => {
                    const isSelected = filterDocId === doc.id;
                    return (
                      <div
                        key={doc.id}
                        onClick={() => {
                          setFilterDocId(isSelected ? 'all' : doc.id);
                          setSelectedDoc(doc);
                        }}
                        className={`p-2 rounded-lg border transition cursor-pointer flex items-center justify-between text-xs ${
                          isSelected
                            ? 'bg-indigo-950/40 border-indigo-500/50 text-white'
                            : 'bg-slate-900/30 border-slate-800/60 hover:bg-slate-900/70 text-slate-300'
                        }`}
                      >
                        <div className="flex items-center gap-2 min-w-0 flex-1">
                          <FileText className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
                          <span className="truncate text-xs">{doc.title}</span>
                        </div>
                        <span
                          className={`text-[9px] font-mono px-1 rounded ml-1 shrink-0 ${
                            doc.status === 'completed'
                              ? 'text-emerald-400'
                              : doc.status === 'processing'
                              ? 'text-indigo-400 animate-pulse'
                              : 'text-rose-400'
                          }`}
                        >
                          {doc.page_count}p
                        </span>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>

          {/* User Account / Tenant Status Footer */}
          <div className="p-4 border-t border-slate-800/80 bg-[#080d17]/90 text-xs text-slate-400 flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              Tenant Isolation
            </span>
            <span className="font-mono text-[11px] text-slate-300">
              {currentUser ? `User #${currentUser.id}` : 'Guest'}
            </span>
          </div>
        </aside>

        {/* Central Workspace Area */}
        <main className="flex-1 flex flex-col overflow-hidden bg-gradient-to-b from-[#090e1a] to-[#080c14]">
          {/* TAB 1: ASK DOCUMIND (Primary RAG Interface) */}
          {activeTab === 'ask' && (
            <div className="flex-1 flex overflow-hidden">
              {/* Collapsible Conversation History Sidebar */}
              {showConversationSidebar && (
                <div className="w-64 sm:w-72 border-r border-slate-800/80 bg-[#080d1a] flex flex-col shrink-0 overflow-hidden">
                  {/* Sidebar Header & New Chat button */}
                  <div className="p-3 border-b border-slate-800/80 flex items-center justify-between gap-2">
                    <button
                      onClick={handleNewConversation}
                      disabled={isCreatingConv}
                      className="flex-1 py-2 px-3 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-500 hover:to-indigo-600 text-white font-medium text-xs flex items-center justify-center gap-2 shadow-sm transition cursor-pointer disabled:opacity-50"
                      title="Start a new conversation"
                    >
                      <Plus className="w-3.5 h-3.5" />
                      <span>New Chat</span>
                    </button>
                  </div>

                  {/* Conversation Thread List */}
                  <div className="flex-1 overflow-y-auto p-2 space-y-1">
                    {loadingConversations ? (
                      <div className="p-4 text-center text-xs text-slate-500 flex items-center justify-center gap-2">
                        <RefreshCw className="w-3.5 h-3.5 animate-spin text-indigo-400" />
                        <span>Loading chats...</span>
                      </div>
                    ) : conversationError ? (
                      <div className="p-4 text-center text-xs text-rose-400">
                        {conversationError}
                      </div>
                    ) : conversations.length === 0 ? (
                      <div className="p-6 text-center text-xs text-slate-500 space-y-2">
                        <MessageSquare className="w-6 h-6 text-slate-600 mx-auto" />
                        <p>No conversations yet.</p>
                        <p className="text-[11px] text-slate-600">Start a new chat to begin.</p>
                      </div>
                    ) : (
                      conversations.map((conv) => {
                        const isActive = activeConversationId === conv.id;
                        const isConfirmingDelete = deleteConfirmId === conv.id;
                        const scopedDoc = conv.document_id ? documents.find((d) => d.id === conv.document_id) : null;

                        return (
                          <div
                            key={conv.id}
                            onClick={() => {
                              if (!isActive && !isConfirmingDelete) {
                                loadConversationHistory(conv.id);
                              }
                            }}
                            className={`group relative rounded-xl p-2.5 text-xs transition border cursor-pointer ${
                              isActive
                                ? 'bg-indigo-950/40 border-indigo-500/40 text-white'
                                : 'bg-slate-900/30 hover:bg-slate-800/50 border-slate-800/50 text-slate-300'
                            }`}
                          >
                            <div className="flex items-start justify-between gap-1.5">
                              <div className="flex-1 min-w-0 space-y-1">
                                <p className="font-medium truncate text-xs leading-snug">
                                  {conv.title || 'Untitled Chat'}
                                </p>
                                <div className="flex items-center gap-1.5 text-[10px] text-slate-500">
                                  {conv.document_scope_invalidated ? (
                                    <span className="text-amber-400 font-semibold flex items-center gap-1">
                                      <AlertTriangle className="w-2.5 h-2.5" /> Scope Invalidated
                                    </span>
                                  ) : scopedDoc ? (
                                    <span className="truncate max-w-[140px] text-indigo-300">
                                      {scopedDoc.title}
                                    </span>
                                  ) : (
                                    <span>Cross-Doc</span>
                                  )}
                                </div>
                              </div>

                              {/* Delete Action / Confirmation */}
                              <div className="shrink-0 flex items-center" onClick={(e) => e.stopPropagation()}>
                                {isConfirmingDelete ? (
                                  <div className="flex items-center gap-1 bg-red-950/80 border border-red-800/60 rounded-md p-0.5">
                                    <button
                                      onClick={() => handleDeleteConversation(conv.id)}
                                      className="px-1.5 py-0.5 rounded bg-red-600 text-white hover:bg-red-500 text-[10px] font-semibold"
                                      title="Confirm Delete"
                                    >
                                      Delete
                                    </button>
                                    <button
                                      onClick={() => setDeleteConfirmId(null)}
                                      className="p-0.5 text-slate-400 hover:text-white"
                                      title="Cancel"
                                    >
                                      <X className="w-3 h-3" />
                                    </button>
                                  </div>
                                ) : (
                                  <button
                                    onClick={() => setDeleteConfirmId(conv.id)}
                                    className={`p-1 rounded hover:bg-slate-700/80 text-slate-500 hover:text-red-400 transition ${
                                      isActive ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
                                    }`}
                                    title="Delete Conversation"
                                    aria-label="Delete Conversation"
                                  >
                                    <Trash2 className="w-3.5 h-3.5" />
                                  </button>
                                )}
                              </div>
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              )}

              {/* Main Conversation Stream */}
              <div className="flex-1 flex flex-col overflow-hidden border-r border-slate-800/60">
                {/* Conversation Header & Context Filter */}
                <div className="h-14 border-b border-slate-800/80 bg-[#0c1220]/60 px-6 flex items-center justify-between shrink-0">
                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => setShowConversationSidebar(!showConversationSidebar)}
                      className={`p-1.5 rounded-lg border transition cursor-pointer ${
                        showConversationSidebar
                          ? 'bg-indigo-950/40 border-indigo-500/40 text-indigo-300'
                          : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-white'
                      }`}
                      title="Toggle Chats Sidebar"
                      aria-label="Toggle Chats Sidebar"
                    >
                      <PanelLeft className="w-4 h-4" />
                    </button>
                    <div className="w-7 h-7 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center">
                      <Sparkles className="w-4 h-4 text-cyan-400" />
                    </div>
                    <div>
                      <h2 className="text-sm font-bold text-white flex items-center gap-2">
                        <span className="truncate max-w-[200px] sm:max-w-xs">
                          {conversations.find((c) => c.id === activeConversationId)?.title || 'Ask DocuMind'}
                        </span>
                        <span className="text-[10px] font-mono text-slate-400 font-normal">
                          {chatMessages.length > 0 ? `(${chatMessages.length} messages)` : ''}
                        </span>
                      </h2>
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    {/* Document Context Selector */}
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-slate-400 hidden sm:inline">Scope:</span>
                      <select
                        value={filterDocId}
                        onChange={(e) => setFilterDocId(e.target.value)}
                        className="bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 cursor-pointer max-w-[200px] truncate"
                        title="Filter question scope to a specific document"
                      >
                        <option value="all">All Documents ({documents.length})</option>
                        {documents.map((d) => (
                          <option key={d.id} value={d.id}>
                            {d.title}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Reset Chat View Button */}
                    <button
                      onClick={handleClearConversation}
                      className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-xs font-medium border border-slate-700 transition cursor-pointer"
                      title="Clear chat view"
                      aria-label="Clear chat"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                      <span className="hidden sm:inline">Clear Chat</span>
                    </button>

                    {/* Toggle Source Panel on Desktop */}
                    <button
                      onClick={() => setShowSourcePanel(!showSourcePanel)}
                      className={`hidden lg:flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium border transition cursor-pointer ${
                        showSourcePanel
                          ? 'bg-indigo-950/40 border-indigo-500/40 text-indigo-300'
                          : 'bg-slate-800 border-slate-700 text-slate-400'
                      }`}
                      title="Toggle Source Inspector"
                    >
                      <BookOpen className="w-3.5 h-3.5" />
                      <span>Sources</span>
                    </button>
                  </div>
                </div>

                {/* Scope Invalidation Alert Banner */}
                {conversations.find((c) => c.id === activeConversationId)?.document_scope_invalidated && (
                  <div className="mx-6 mt-4 p-3.5 rounded-xl bg-amber-950/40 border border-amber-500/40 text-amber-300 text-xs flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                      <span>This conversation's document scope was invalidated because the referenced document was deleted.</span>
                    </div>
                    <button
                      onClick={() => handleCreateConversation()}
                      className="px-2.5 py-1 rounded-lg bg-amber-600 hover:bg-amber-500 text-black font-semibold text-[11px] shrink-0"
                    >
                      Start New Chat
                    </button>
                  </div>
                )}

                {/* Loading History Indicator */}
                {loadingHistory && (
                  <div className="p-3 text-center text-xs text-indigo-300 bg-indigo-950/20 border-b border-indigo-900/30 flex items-center justify-center gap-2">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin text-indigo-400" />
                    <span>Loading conversation history...</span>
                  </div>
                )}

                {/* Scrollable Conversation Stream */}
                <div ref={chatScrollRef} className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
                  {/* Empty Welcome Screen with Suggested Questions */}
                  {chatMessages.length === 0 ? (
                    <div className="max-w-2xl mx-auto py-8 sm:py-12 space-y-8">
                      <div className="text-center space-y-3">
                        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-500/20 via-cyan-500/20 to-indigo-500/20 border border-indigo-500/30 flex items-center justify-center mx-auto shadow-lg shadow-indigo-500/10">
                          <Sparkles className="w-7 h-7 text-cyan-400" />
                        </div>
                        <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
                          Ask DocuMind Intelligence
                        </h1>
                        <p className="text-sm text-slate-400 leading-relaxed max-w-lg mx-auto">
                          Ask natural-language questions across your uploaded enterprise PDFs. Every answer is
                          synthesized with Gemini, backed by pgvector vector similarity, and attributed with
                          verifiable page citations.
                        </p>

                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs text-slate-400">
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                          <span>{completedDocs.length} of {documents.length} documents indexed and ready</span>
                        </div>
                      </div>

                      {/* Suggested Questions */}
                      <div className="space-y-3">
                        <span className="text-xs font-bold uppercase tracking-wider text-slate-500 block text-center">
                          Suggested Inquiries
                        </span>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                          {SUGGESTED_QUESTIONS.map((q, idx) => (
                            <button
                              key={idx}
                              onClick={() => handleAskQuestion(q)}
                              className="p-3.5 rounded-xl border border-slate-800/80 bg-slate-900/40 hover:bg-slate-900 hover:border-indigo-500/40 text-left text-xs text-slate-300 hover:text-white transition group space-y-1.5 cursor-pointer"
                            >
                              <div className="flex items-center gap-2 text-indigo-400 group-hover:text-cyan-400">
                                <Sparkles className="w-3.5 h-3.5" />
                                <span className="font-semibold text-[11px] uppercase tracking-wider">Example</span>
                              </div>
                              <p className="leading-snug">{q}</p>
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : (
                    // Conversation Message List
                    chatMessages.map((msg) => (
                      <div
                        key={msg.id}
                        className={`flex flex-col space-y-2 ${
                          msg.role === 'user' ? 'items-end' : 'items-start'
                        }`}
                      >
                        {/* User Message Bubble */}
                        {msg.role === 'user' ? (
                          <div className="max-w-xl bg-indigo-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 shadow-md text-xs sm:text-sm leading-relaxed">
                            {msg.content}
                          </div>
                        ) : (
                          // Assistant Message Card
                          <div className="w-full max-w-3xl rounded-2xl border border-slate-800 bg-[#0e1424] p-5 shadow-sm space-y-4">
                            {/* Assistant Card Header */}
                            <div className="flex items-center justify-between border-b border-slate-800/70 pb-3 text-xs">
                              <div className="flex items-center gap-2">
                                <div className="w-6 h-6 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-xs">
                                  <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                                </div>
                                <span className="font-semibold text-white">DocuMind AI</span>
                                {msg.model_used && (
                                  <span className="px-2 py-0.5 rounded-full bg-slate-800 text-[10px] text-slate-300 font-mono">
                                    {msg.model_used}
                                  </span>
                                )}
                              </div>

                              <div className="flex items-center gap-3 text-[11px] text-slate-400">
                                {msg.total_latency_ms && (
                                  <span className="font-mono text-emerald-400">
                                    {msg.total_latency_ms}ms
                                  </span>
                                )}
                                {msg.token_usage?.total_tokens && (
                                  <span className="font-mono text-amber-300 hidden sm:inline">
                                    {msg.token_usage.total_tokens} tokens
                                  </span>
                                )}
                              </div>
                            </div>

                            {/* Abstention / Insufficient Evidence Banner */}
                            {msg.insufficient_evidence && (
                              <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-xs text-amber-300 flex items-start gap-2.5">
                                <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                                <div>
                                  <span className="font-semibold">Abstention Alert: </span>
                                  The system abstained from answering because the retrieved document context does
                                  not contain sufficient verifiable evidence.
                                </div>
                              </div>
                            )}

                            {/* Error Message Notice */}
                            {msg.error && (
                              <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 flex items-start gap-2.5">
                                <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                                <div>
                                  <span className="font-semibold">Error: </span>
                                  {msg.error}
                                </div>
                              </div>
                            )}

                            {/* Answer Body */}
                            <div className="text-xs sm:text-sm text-slate-200 leading-relaxed font-sans whitespace-pre-wrap">
                              {msg.content}
                            </div>

                            {/* Citations Badges List */}
                            {msg.citations && msg.citations.length > 0 && (
                              <div className="pt-3 border-t border-slate-800/70 space-y-2">
                                <div className="flex items-center justify-between text-xs text-slate-400">
                                  <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                                    <BookOpen className="w-3.5 h-3.5 text-indigo-400" />
                                    Source Citations ({msg.citations.length})
                                  </span>
                                  <span className="text-[11px] text-slate-500">
                                    Click citation to inspect source page
                                  </span>
                                </div>

                                <div className="flex flex-wrap gap-2">
                                  {msg.citations.map((c, cIdx) => {
                                    const isSelected =
                                      activeCitation?.chunk_id === c.chunk_id;
                                    return (
                                      <button
                                        key={c.chunk_id}
                                        onClick={() => {
                                          setActiveCitation(c);
                                          setShowSourcePanel(true);
                                        }}
                                        className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition cursor-pointer ${
                                          isSelected
                                            ? 'bg-indigo-600/30 border-indigo-500 text-white shadow-sm'
                                            : 'bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700 hover:text-white'
                                        }`}
                                      >
                                        <span className="w-4 h-4 rounded-full bg-indigo-500/20 text-indigo-400 font-mono text-[10px] flex items-center justify-center font-bold">
                                          {cIdx + 1}
                                        </span>
                                        <span className="truncate max-w-[140px] sm:max-w-[200px]">
                                          {c.document_title}
                                        </span>
                                        <span className="px-1.5 py-0.2 rounded bg-cyan-500/10 text-cyan-400 text-[10px] font-mono">
                                          p.{c.page_number}
                                        </span>
                                      </button>
                                    );
                                  })}
                                </div>
                              </div>
                            )}

                            {/* Assistant Card Footer Actions */}
                            <div className="flex items-center justify-between pt-2 border-t border-slate-800/50 text-xs text-slate-500">
                              <div className="text-[11px] font-mono">
                                {new Date(msg.timestamp).toLocaleTimeString()}
                              </div>
                              <div className="flex items-center gap-2">
                                <button
                                  onClick={() => handleCopyAnswer(msg)}
                                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition cursor-pointer"
                                  title="Copy Answer text"
                                >
                                  {copiedMessageId === msg.id ? (
                                    <>
                                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                                      <span className="text-emerald-400 text-[11px]">Copied</span>
                                    </>
                                  ) : (
                                    <>
                                      <Copy className="w-3.5 h-3.5" />
                                      <span className="text-[11px]">Copy Answer</span>
                                    </>
                                  )}
                                </button>

                                {msg.citations && msg.citations.length > 0 && (
                                  <button
                                    onClick={() => handleCopyCitations(msg)}
                                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition cursor-pointer"
                                    title="Copy Citations references"
                                  >
                                    {copiedCitationsId === msg.id ? (
                                      <>
                                        <Check className="w-3.5 h-3.5 text-emerald-400" />
                                        <span className="text-emerald-400 text-[11px]">Copied</span>
                                      </>
                                    ) : (
                                      <>
                                        <Quote className="w-3.5 h-3.5" />
                                        <span className="text-[11px]">Copy Citations</span>
                                      </>
                                    )}
                                  </button>
                                )}
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    ))
                  )}

                  {/* Active In-Flight Generation Loader */}
                  {askingRAG && (
                    <div className="w-full max-w-3xl rounded-2xl border border-indigo-500/30 bg-[#0e1424] p-5 space-y-3 animate-pulse">
                      <div className="flex items-center gap-2 text-xs text-indigo-400">
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span className="font-semibold">Synthesizing Answer with Gemini...</span>
                      </div>
                      <p className="text-xs text-slate-400">
                        Retrieving passages from pgvector, validating citations, and checking evidence sufficiency.
                      </p>
                    </div>
                  )}

                  {/* RAG Error Display */}
                  {ragError && !askingRAG && (
                    <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
                        <span>{ragError}</span>
                      </div>
                      <button
                        onClick={() => setRagError(null)}
                        className="text-slate-400 hover:text-white"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  )}
                </div>

                {/* Bottom Input Prompt Bar */}
                <div className="p-4 bg-[#0a0f1c] border-t border-slate-800/80 shrink-0">
                  <div className="max-w-3xl mx-auto space-y-2">
                    <div className="relative rounded-2xl border border-slate-800 bg-[#0e1424] focus-within:border-indigo-500 transition shadow-inner">
                      <textarea
                        ref={askInputRef}
                        rows={2}
                        value={askInput}
                        onChange={(e) => setAskInput(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleAskQuestion();
                          }
                        }}
                        disabled={askingRAG}
                        placeholder="Ask a natural-language question across your documents... (Press Enter to submit, Shift+Enter for new line)"
                        className="w-full bg-transparent px-4 py-3 text-xs sm:text-sm text-slate-200 placeholder:text-slate-500 focus:outline-none resize-none"
                      />

                      <div className="flex items-center justify-between px-4 pb-2 text-[11px] text-slate-500">
                        <div className="flex items-center gap-2">
                          <span>Enter to submit</span>
                          <span>&bull;</span>
                          <span>Shift+Enter for new line</span>
                        </div>

                        <button
                          onClick={() => handleAskQuestion()}
                          disabled={askingRAG || !askInput.trim()}
                          aria-label="Submit question"
                          className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white font-semibold text-xs transition cursor-pointer shadow-md shadow-indigo-600/30"
                          title="Submit question"
                        >
                          {askingRAG ? (
                            <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          ) : (
                            <Send className="w-3.5 h-3.5" />
                          )}
                          <span>Ask</span>
                        </button>
                      </div>
                    </div>

                    <div className="flex items-center justify-between text-[10px] text-slate-500 px-2">
                      <span>DocuMind AI answers are strictly grounded in your authorized document context.</span>
                      {chatMessages.length > 0 && (
                        <button
                          onClick={handleClearConversation}
                          className="hover:text-slate-400 underline cursor-pointer"
                        >
                          Clear Chat
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Document Source Inspector Panel */}
              {showSourcePanel && (
                <aside className="w-80 sm:w-96 border-l border-slate-800/80 bg-[#0a0f1c] flex flex-col justify-between shrink-0">
                  <div className="p-5 space-y-5 flex-1 flex flex-col overflow-y-auto">
                    {/* Source Inspector Header */}
                    <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                      <div className="flex items-center gap-2">
                        <BookOpen className="w-4 h-4 text-indigo-400" />
                        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                          Source Inspector
                        </h3>
                      </div>
                      <button
                        onClick={() => setShowSourcePanel(false)}
                        className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-white"
                        title="Close Inspector"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>

                    {/* Source Detail Card */}
                    {activeCitation ? (
                      <div className="space-y-4">
                        <div className="p-4 rounded-xl border border-indigo-500/30 bg-slate-900/60 space-y-3">
                          <div className="flex items-start justify-between gap-2">
                            <div>
                              <span className="text-[10px] font-semibold uppercase tracking-wider text-indigo-400 block mb-0.5">
                                Verified Source
                              </span>
                              <h4 className="text-sm font-bold text-white leading-tight">
                                {activeCitation.document_title}
                              </h4>
                            </div>
                            <span className="px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 text-xs font-mono font-semibold">
                              Page {activeCitation.page_number}
                            </span>
                          </div>

                          <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1 border-t border-slate-800/60">
                            <span>Hybrid Similarity:</span>
                            <span className="font-mono text-emerald-400 font-semibold">
                              {(activeCitation.score * 100).toFixed(1)}% match
                            </span>
                          </div>
                        </div>

                        {/* Passage Excerpt Block */}
                        <div className="space-y-1.5">
                          <span className="text-[11px] font-semibold text-slate-400 block">
                            Retrieved Context Excerpt
                          </span>
                          <div className="p-3.5 rounded-xl bg-[#070b14] border border-slate-800 text-xs text-slate-300 leading-relaxed font-sans relative">
                            <Quote className="w-4 h-4 text-indigo-400/40 absolute top-2 right-2" />
                            <p className="whitespace-pre-wrap">{activeCitation.excerpt}</p>
                          </div>
                        </div>

                        {/* Action Buttons: Open PDF at Page & Download */}
                        <div className="space-y-2 pt-2">
                          <button
                            type="button"
                            onClick={() =>
                              handleOpenPdfAtPage(
                                activeCitation.document_id,
                                activeCitation.page_number
                              )
                            }
                            disabled={openingPdfPage}
                            className="w-full py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-cyan-600 hover:from-indigo-500 hover:to-cyan-500 text-white font-semibold text-xs transition flex items-center justify-center gap-2 cursor-pointer shadow-md shadow-indigo-600/20 disabled:opacity-50"
                          >
                            <ExternalLink className="w-3.5 h-3.5" />
                            {openingPdfPage
                              ? 'Loading PDF Page...'
                              : `Open PDF at Page ${activeCitation.page_number}`}
                          </button>

                          <button
                            type="button"
                            onClick={() =>
                              handleDownloadDocument(
                                activeCitation.document_id,
                                activeCitation.document_title
                              )
                            }
                            disabled={downloadingDocId === activeCitation.document_id}
                            className="w-full py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
                          >
                            <Download className="w-3.5 h-3.5 text-indigo-400" />
                            <span>Download Full Document</span>
                          </button>
                        </div>
                      </div>
                    ) : (
                      // Empty Inspector Placeholder
                      <div className="p-8 text-center border border-dashed border-slate-800 rounded-xl my-6 space-y-2">
                        <BookOpen className="w-8 h-8 text-slate-600 mx-auto" />
                        <h4 className="text-xs font-semibold text-slate-300">
                          No Citation Selected
                        </h4>
                        <p className="text-[11px] text-slate-500 leading-relaxed">
                          Click any numbered citation badge <span className="text-indigo-400 font-mono">[1]</span> in an AI
                          answer to inspect the exact underlying document page and excerpt passage.
                        </p>
                      </div>
                    )}
                  </div>

                  <div className="p-4 border-t border-slate-800/80 bg-[#080d17]/80 text-[11px] text-slate-500 text-center">
                    Citations provide verifiable evidence provenance
                  </div>
                </aside>
              )}
            </div>
          )}

          {/* TAB 2: DOCUMENTS & KNOWLEDGE BASE MANAGEMENT */}
          {(activeTab === 'documents' || activeTab === 'upload') && (
            <div className="flex-1 overflow-y-auto p-6 sm:p-8 space-y-8">
              {/* Header Banner */}
              <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-r from-slate-900 via-indigo-950/20 to-slate-900 p-6 sm:p-8 shadow-sm">
                <div className="max-w-2xl">
                  <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold mb-4">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    Stage 3: PDF Ingestion & Semantic Vector Search Active
                  </div>
                  <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight mb-2">
                    Document Knowledge Base
                  </h1>
                  <p className="text-slate-400 text-xs sm:text-sm leading-relaxed mb-6">
                    Upload technical PDFs, policy manuals, or financial filings. Text is extracted page-by-page,
                    chunked with punctuation snapping, and stored as 1536-dimensional embeddings in pgvector.
                  </p>

                  <div className="flex flex-wrap items-center gap-3">
                    <button
                      onClick={() => setActiveTab('ask')}
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition cursor-pointer shadow-md shadow-indigo-600/30"
                    >
                      <Sparkles className="w-4 h-4 text-cyan-400" />
                      Ask Questions
                    </button>
                    <button
                      onClick={() => {
                        if (fileInputRef.current) fileInputRef.current.click();
                      }}
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold transition cursor-pointer"
                    >
                      <UploadCloud className="w-4 h-4 text-indigo-400" />
                      Upload PDF
                    </button>
                    <a
                      href="http://localhost:8000/docs"
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 text-xs font-semibold border border-slate-700 transition"
                    >
                      <ExternalLink className="w-3.5 h-3.5" />
                      Swagger API
                    </a>
                  </div>
                </div>
              </div>

              {/* Upload Dropzone Form */}
              <div className="rounded-xl border border-slate-800 bg-[#0e1424] p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <UploadCloud className="w-4 h-4 text-indigo-400" />
                    <h3 className="text-sm font-semibold text-white">Upload & Ingest Enterprise PDF</h3>
                  </div>
                  <span className="text-xs text-slate-500">
                    Max 15MB &bull; Automatic chunking & pgvector indexing
                  </span>
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
                        placeholder="Document Title (e.g. Information Security Policy)"
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

              {/* Document List Table */}
              <div className="rounded-xl border border-slate-800 bg-[#0e1424] p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-white">
                    Authorized Documents ({documents.length})
                  </h3>
                  <button
                    onClick={() => fetchDocuments()}
                    disabled={loadingDocs}
                    className="p-1.5 rounded hover:bg-slate-800 text-slate-400 hover:text-white"
                    title="Refresh document list"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${loadingDocs ? 'animate-spin' : ''}`} />
                  </button>
                </div>

                {downloadError && (
                  <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-400 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 shrink-0" />
                      <span>{downloadError}</span>
                    </div>
                    <button
                      type="button"
                      onClick={() => setDownloadError(null)}
                      className="text-slate-400 hover:text-white"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                )}

                {documents.length === 0 ? (
                  <div className="p-8 text-center border border-dashed border-slate-800 rounded-xl text-xs text-slate-500">
                    No documents uploaded yet. Upload your first PDF to begin.
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {documents.map((doc) => (
                      <div
                        key={doc.id}
                        className="p-4 rounded-xl border border-slate-800/80 bg-slate-900/40 hover:border-indigo-500/40 transition space-y-3 flex flex-col justify-between"
                      >
                        <div className="space-y-2">
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex items-center gap-2 min-w-0">
                              <FileText className="w-4 h-4 text-rose-400 shrink-0" />
                              <h4 className="text-xs font-bold text-white truncate">{doc.title}</h4>
                            </div>
                            <span
                              className={`text-[9px] font-mono px-1.5 py-0.5 rounded font-semibold uppercase ${
                                doc.status === 'completed'
                                  ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                  : doc.status === 'processing'
                                  ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 animate-pulse'
                                  : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                              }`}
                            >
                              {doc.status}
                            </span>
                          </div>

                          <div className="grid grid-cols-2 gap-2 text-[10px] text-slate-400 font-mono">
                            <div>Pages: {doc.page_count}</div>
                            <div>Size: {(doc.file_size_bytes / 1024).toFixed(0)} KB</div>
                          </div>

                          {doc.status === 'failed' && (
                            <div className="flex items-center justify-between text-[11px] pt-1">
                              <span className="text-rose-400 truncate max-w-[140px]">
                                {doc.error_message || 'Processing failed'}
                              </span>
                              <button
                                type="button"
                                onClick={() => handleRetryProcess(doc.id)}
                                className="text-indigo-400 hover:text-indigo-300 font-semibold flex items-center gap-1 cursor-pointer"
                              >
                                <RefreshCw className="w-3 h-3" /> Retry
                              </button>
                            </div>
                          )}
                        </div>

                        <div className="flex items-center gap-2 pt-2 border-t border-slate-800/60">
                          <button
                            type="button"
                            onClick={() => {
                              setFilterDocId(doc.id);
                              setActiveTab('ask');
                            }}
                            className="flex-1 py-1.5 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/40 text-indigo-300 text-xs font-medium border border-indigo-500/30 transition text-center cursor-pointer"
                          >
                            Ask Questions
                          </button>
                          <button
                            type="button"
                            onClick={() => handleDownloadDocument(doc.id, doc.filename || doc.title)}
                            disabled={downloadingDocId === doc.id}
                            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition cursor-pointer"
                            title="Download PDF"
                          >
                            <Download className="w-3.5 h-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => handleDeleteDocument(doc.id, doc.title)}
                            className="p-1.5 rounded-lg hover:bg-rose-950/40 text-slate-500 hover:text-rose-400 transition cursor-pointer"
                            title="Delete"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 3: SEMANTIC VECTOR SEARCH INSPECTOR */}
          {activeTab === 'search' && (
            <div className="flex-1 overflow-y-auto p-6 sm:p-8 space-y-6">
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
                      onClick={() => setActiveTab('ask')}
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition cursor-pointer bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
                    >
                      <Sparkles className="w-4 h-4 text-cyan-400" />
                      Ask DocuMind
                    </button>
                    <button
                      onClick={() => setActiveTab('search')}
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition cursor-pointer bg-indigo-600 text-white shadow-lg shadow-indigo-600/30"
                    >
                      <Search className="w-4 h-4" />
                      Semantic Search
                    </button>
                    <button
                      onClick={() => setActiveTab('upload')}
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition cursor-pointer bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
                    >
                      <UploadCloud className="w-4 h-4" />
                      Upload Document
                    </button>
                    <button
                      onClick={() => {
                        setActiveTab('monitoring');
                        fetchMonitoring();
                      }}
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition cursor-pointer bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
                    >
                      <Activity className="w-4 h-4 text-emerald-400" />
                      Observability
                    </button>
                  </div>
                </div>
              </div>

              {/* Document Inspector Card (If a document is selected) */}
              {selectedDoc && (
                <div className="rounded-xl border border-indigo-500/40 bg-gradient-to-br from-indigo-950/20 via-[#0e1424] to-[#0e1424] p-6 shadow-md space-y-4">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/30 text-indigo-400 flex items-center justify-center">
                        <FileText className="w-5 h-5 text-indigo-400" />
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
                          setActiveTab('ask');
                        }}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition cursor-pointer"
                      >
                        <Sparkles className="w-3.5 h-3.5" />
                        Ask Questions
                      </button>
                      <button
                        type="button"
                        onClick={() => handleDownloadDocument(selectedDoc.id, selectedDoc.filename || selectedDoc.title)}
                        disabled={downloadingDocId === selectedDoc.id}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition cursor-pointer disabled:opacity-50"
                      >
                        <Download className={`w-3.5 h-3.5 text-indigo-400 ${downloadingDocId === selectedDoc.id ? 'animate-bounce' : ''}`} />
                        {downloadingDocId === selectedDoc.id ? 'Downloading...' : 'Download PDF'}
                      </button>
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
                </div>
              )}

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
                        Retrieved <span className="text-white font-semibold">{searchResults.length}</span> passages
                      </span>
                      <span className="font-mono text-[11px] text-slate-500">Query: "{searchQuery}"</span>
                    </div>

                    {searchResults.length === 0 ? (
                      <div className="p-8 text-center border border-dashed border-slate-800 rounded-xl">
                        <p className="text-xs text-slate-400">No matching passages found for this query.</p>
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
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 4: OBSERVABILITY & MONITORING TELEMETRY */}
          {activeTab === 'monitoring' && (
            <div className="flex-1 overflow-y-auto p-6 sm:p-8 space-y-6">
              <div className="rounded-xl border border-slate-800 bg-[#0e1424] p-6 space-y-4">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 flex items-center justify-center">
                      <Activity className="w-5 h-5" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="text-sm font-semibold text-white">Observability & Request Telemetry</h3>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                          In-App Monitoring
                        </span>
                      </div>
                      <p className="text-xs text-slate-400">
                        Tracks request performance, token usage, versioned cost estimation, and failure categories.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {monitoringData && (
                      <span className="text-[11px] font-mono text-slate-400 bg-slate-900 px-2.5 py-1 rounded-lg border border-slate-800">
                        Pricing: {monitoringData.pricing_version}
                      </span>
                    )}
                    <button
                      onClick={fetchMonitoring}
                      disabled={loadingMonitoring}
                      className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                    >
                      <RefreshCw className={`w-3.5 h-3.5 ${loadingMonitoring ? 'animate-spin text-emerald-400' : ''}`} />
                      Refresh
                    </button>
                  </div>
                </div>

                {monitoringError && (
                  <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-400 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 shrink-0" />
                    <span>{monitoringError}</span>
                  </div>
                )}
              </div>

              {/* Stat Cards Grid */}
              {monitoringData && (
                <>
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                    <div className="p-4 rounded-xl border border-slate-800 bg-[#0e1424] space-y-1">
                      <span className="text-[11px] text-slate-400 font-medium block">Total Requests</span>
                      <div className="text-2xl font-bold text-white font-mono">{monitoringData.total_requests}</div>
                      <div className="text-[10px] text-slate-500 flex items-center gap-1.5">
                        <span className="text-emerald-400 font-semibold">{monitoringData.successful_requests} ok</span>
                        <span>&bull;</span>
                        <span className="text-rose-400 font-semibold">{monitoringData.failed_requests} err</span>
                      </div>
                    </div>

                    <div className="p-4 rounded-xl border border-slate-800 bg-[#0e1424] space-y-1">
                      <span className="text-[11px] text-slate-400 font-medium block">Avg Latency</span>
                      <div className="text-2xl font-bold text-indigo-400 font-mono">
                        {monitoringData.avg_total_latency_ms}
                        <span className="text-xs text-slate-400 font-normal ml-0.5">ms</span>
                      </div>
                      <div className="text-[10px] text-slate-500">End-to-end average</div>
                    </div>

                    <div className="p-4 rounded-xl border border-slate-800 bg-[#0e1424] space-y-1">
                      <span className="text-[11px] text-slate-400 font-medium block">P95 Latency</span>
                      <div className="text-2xl font-bold text-cyan-400 font-mono">
                        {monitoringData.p95_total_latency_ms}
                        <span className="text-xs text-slate-400 font-normal ml-0.5">ms</span>
                      </div>
                      <div className="text-[10px] text-slate-500">95th percentile SLA</div>
                    </div>

                    <div className="p-4 rounded-xl border border-slate-800 bg-[#0e1424] space-y-1">
                      <span className="text-[11px] text-slate-400 font-medium block">Error Rate</span>
                      <div
                        className={`text-2xl font-bold font-mono ${
                          monitoringData.error_rate > 0 ? 'text-rose-400' : 'text-emerald-400'
                        }`}
                      >
                        {(monitoringData.error_rate * 100).toFixed(1)}%
                      </div>
                      <div className="text-[10px] text-slate-500">Failed / Total</div>
                    </div>

                    <div className="p-4 rounded-xl border border-slate-800 bg-[#0e1424] space-y-1">
                      <span className="text-[11px] text-slate-400 font-medium block">Token Usage</span>
                      <div className="text-2xl font-bold text-amber-300 font-mono">
                        {monitoringData.token_usage?.total_tokens !== null &&
                        monitoringData.token_usage?.total_tokens !== undefined
                          ? monitoringData.token_usage.total_tokens.toLocaleString()
                          : <span className="text-slate-500 text-lg italic">Unknown</span>}
                      </div>
                      <div className="text-[10px] text-slate-500">Prompt & Completion</div>
                    </div>

                    <div className="p-4 rounded-xl border border-slate-800 bg-[#0e1424] space-y-1">
                      <span className="text-[11px] text-slate-400 font-medium block">Estimated Cost</span>
                      <div className="text-2xl font-bold text-emerald-400 font-mono">
                        {monitoringData.total_estimated_cost_usd !== null &&
                        monitoringData.total_estimated_cost_usd !== undefined
                          ? `$${monitoringData.total_estimated_cost_usd.toFixed(4)}`
                          : <span className="text-slate-500 text-lg italic">Unknown</span>}
                      </div>
                      <div className="text-[10px] text-slate-500">Catalog-based</div>
                    </div>
                  </div>

                  {/* Recent Requests Table */}
                  <div className="rounded-xl border border-slate-800 bg-[#0e1424] p-6 space-y-4">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                        Recent Request Lifecycle Log ({monitoringData.recent_requests.length})
                      </h4>
                      <span className="text-[11px] text-slate-500 font-mono">
                        Zero sensitive document text stored
                      </span>
                    </div>

                    {monitoringData.recent_requests.length === 0 ? (
                      <div className="p-8 text-center border border-dashed border-slate-800 rounded-xl text-xs text-slate-500">
                        No requests recorded yet. Run a RAG query to see live telemetry.
                      </div>
                    ) : (
                      <div className="overflow-x-auto">
                        <table className="w-full text-left text-xs">
                          <thead>
                            <tr className="border-b border-slate-800 text-slate-400 font-medium">
                              <th className="pb-2.5 font-semibold">Request ID</th>
                              <th className="pb-2.5 font-semibold">Status</th>
                              <th className="pb-2.5 font-semibold">Retrieval</th>
                              <th className="pb-2.5 font-semibold">Generation</th>
                              <th className="pb-2.5 font-semibold">Total Latency</th>
                              <th className="pb-2.5 font-semibold">Tokens</th>
                              <th className="pb-2.5 font-semibold">Estimated Cost</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                            {monitoringData.recent_requests.map((req) => (
                              <tr key={req.id} className="hover:bg-slate-900/40 transition">
                                <td className="py-2.5 font-semibold text-slate-200">{req.request_id}</td>
                                <td className="py-2.5">
                                  <span
                                    className={`px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase ${
                                      req.status === 'success'
                                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                        : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                                    }`}
                                  >
                                    {req.status}
                                  </span>
                                </td>
                                <td className="py-2.5 text-slate-400">{req.retrieval_latency_ms}ms</td>
                                <td className="py-2.5 text-slate-400">{req.generation_latency_ms}ms</td>
                                <td className="py-2.5 text-slate-200 font-semibold">{req.total_latency_ms}ms</td>
                                <td className="py-2.5">
                                  {req.total_tokens !== null && req.total_tokens !== undefined ? (
                                    <span className="text-amber-300">{req.total_tokens}</span>
                                  ) : (
                                    <span className="text-slate-500 italic">Unknown</span>
                                  )}
                                </td>
                                <td className="py-2.5">
                                  {req.estimated_cost_usd !== null && req.estimated_cost_usd !== undefined ? (
                                    <span className="text-emerald-400">${req.estimated_cost_usd.toFixed(4)}</span>
                                  ) : (
                                    <span className="text-slate-500 italic">Unknown</span>
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          )}
        </main>
      </div>

      {/* Authentication Modal */}
      {showAuthModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="bg-[#0e1424] border border-slate-800 rounded-2xl max-w-sm w-full p-6 shadow-2xl relative space-y-4">
            <button
              onClick={() => {
                setShowAuthModal(false);
                setAuthError(null);
              }}
              className="absolute top-4 right-4 text-slate-400 hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="text-center space-y-1">
              <div className="w-10 h-10 rounded-xl bg-indigo-500/20 text-indigo-400 flex items-center justify-center mx-auto mb-2">
                <Lock className="w-5 h-5" />
              </div>
              <h3 className="text-lg font-bold text-white">
                {authMode === 'login' ? 'Welcome Back' : 'Create Account'}
              </h3>
              <p className="text-xs text-slate-400">
                {authMode === 'login'
                  ? 'Sign in to access your authorized enterprise documents'
                  : 'Register a new tenant user profile'}
              </p>
            </div>

            {/* Mode Switcher */}
            <div className="flex rounded-lg bg-slate-900 p-1 border border-slate-800 text-xs">
              <button
                type="button"
                onClick={() => {
                  setAuthMode('login');
                  setAuthError(null);
                }}
                className={`flex-1 py-1.5 rounded-md font-semibold transition cursor-pointer ${
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
                className={`flex-1 py-1.5 rounded-md font-semibold transition cursor-pointer ${
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

      {/* Global Footer */}
      <footer className="h-10 border-t border-slate-800/80 bg-[#0a0f1c] px-6 flex items-center justify-between text-[11px] text-slate-500 shrink-0">
        <div>DocuMind AI &bull; Enterprise Document Intelligence & RAG Platform</div>
        <div className="flex items-center gap-4">
          <span>React 19 &bull; Vite &bull; FastAPI &bull; PostgreSQL (pgvector 1536d)</span>
        </div>
      </footer>
    </div>
  );
}
