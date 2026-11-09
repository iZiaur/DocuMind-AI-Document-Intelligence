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
  LayoutDashboard,
  FileEdit,
  Settings,
  HelpCircle,
  Users,
  ChevronDown,
  ChevronRight,
  ChevronLeft,
  MoreVertical,
  Edit3,
  GitFork,
  GitBranch,
  Globe,
  ArrowRight,
  Sun,
  Bell,
  Terminal,
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
    <div className="min-h-screen flex flex-col bg-[#F9F9FB] text-slate-800 font-sans">
      {/* Top Header Bar */}
      <header className="h-16 border-b border-slate-200/80 bg-white sticky top-0 z-30 px-4 sm:px-6 flex items-center justify-between shadow-[0_1px_2px_rgba(0,0,0,0.02)]">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 rounded-xl bg-slate-100 text-slate-600 hover:text-slate-900"
            title="Toggle Menu"
          >
            <Menu className="w-5 h-5" />
          </button>

          <div
            onClick={() => setActiveTab('search')}
            className="flex items-center gap-2.5 cursor-pointer group"
          >
            <div className="w-9 h-9 rounded-xl bg-[#3B4BF7] flex items-center justify-center shadow-sm shadow-[#3B4BF7]/20 group-hover:scale-105 transition">
              <Layers className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-base text-slate-900 tracking-tight">DocuMind AI</span>
                <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-indigo-50 text-[#3B4BF7] border border-indigo-200/60">
                  pgvector RAG
                </span>
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block">Enterprise Document Intelligence & Grounded Q&A</p>
            </div>
          </div>
        </div>

        {/* Global Search Bar & Indicators & User Bar */}
        <div className="flex items-center gap-2.5 sm:gap-3">
          {/* Header Search Input */}
          <div className="relative hidden md:flex items-center">
            <Search className="w-3.5 h-3.5 absolute left-3 text-slate-400" />
            <input
              type="text"
              placeholder="Search..."
              className="bg-slate-50 border border-slate-200/80 rounded-xl pl-8 pr-12 py-1.5 text-xs text-slate-700 placeholder:text-slate-400 focus:outline-none focus:border-[#3B4BF7] w-40 lg:w-56 transition"
            />
            <span className="absolute right-2.5 text-[10px] font-mono text-slate-400 bg-white border border-slate-200 px-1 rounded shadow-2xs">
              ⌘F
            </span>
          </div>

          {/* Quick Controls: Terminal, Theme Toggle, Notifications */}
          <button
            className="p-2 rounded-xl text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition hidden sm:inline-flex"
            title="Terminal / Commands"
          >
            <Terminal className="w-4 h-4" />
          </button>

          <button
            className="p-2 rounded-xl text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition hidden sm:inline-flex"
            title="Toggle Theme"
          >
            <Sun className="w-4 h-4" />
          </button>

          <button
            className="p-2 rounded-xl text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition relative"
            title="Notifications"
          >
            <Bell className="w-4 h-4" />
            <span className="w-1.5 h-1.5 rounded-full bg-[#3B4BF7] absolute top-2 right-2"></span>
          </button>

          {/* Health Pill */}
          <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-50 border border-slate-200/80 text-xs text-slate-600">
            <Server className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-400 hidden lg:inline">API:</span>
            {health ? (
              <span className="text-emerald-600 font-medium flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                Active {latencyMs !== null && `(${latencyMs}ms)`}
              </span>
            ) : (
              <span className="text-rose-500 font-medium flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-rose-500"></span>
                Degraded
              </span>
            )}
          </div>

          {/* Database Pill */}
          <div className="hidden xl:flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-50 border border-slate-200/80 text-xs text-slate-600">
            <Database className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-400">Embeddings:</span>
            <span className="text-indigo-600 font-mono font-medium">1536 dim</span>
          </div>

          <button
            onClick={fetchHealth}
            disabled={loadingHealth}
            className="p-2 rounded-xl bg-slate-50 hover:bg-slate-100 text-slate-500 hover:text-slate-800 transition border border-slate-200/80 disabled:opacity-50 cursor-pointer"
            title="Refresh system health"
          >
            <RefreshCw className={`w-4 h-4 ${loadingHealth ? 'animate-spin text-[#3B4BF7]' : ''}`} />
          </button>

          {/* User Auth Section */}
          {currentUser ? (
            <div className="flex items-center gap-2.5 pl-2 border-l border-slate-200">
              <div className="flex items-center gap-2 cursor-pointer">
                <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-[#3B4BF7] to-indigo-400 flex items-center justify-center text-xs font-bold text-white shadow-sm ring-2 ring-slate-100">
                  {currentUser.username.substring(0, 2).toUpperCase()}
                </div>
                <div className="hidden lg:block text-left text-xs">
                  <div className="font-semibold text-slate-800 flex items-center gap-1">
                    {currentUser.username}
                    <ChevronDown className="w-3 h-3 text-slate-400" />
                  </div>
                  <div className="text-[10px] text-slate-400 truncate max-w-[110px]">{currentUser.email}</div>
                </div>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-50 hover:bg-rose-50 hover:text-rose-600 text-slate-600 transition text-xs font-medium border border-slate-200/80 cursor-pointer"
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
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-[#3B4BF7] hover:bg-[#2F3EE0] text-white font-medium text-xs transition shadow-sm shadow-[#3B4BF7]/20 cursor-pointer"
            >
              <LogIn className="w-3.5 h-3.5" />
              Sign In / Register
            </button>
          )}
        </div>
      </header>

      {/* Main Workspace with Structured Dokumin Layout */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Left Sidebar Navigation */}
        <aside
          className={`w-64 border-r border-slate-200/80 bg-white flex flex-col justify-between shrink-0 z-20 transition-all duration-300 ${
            mobileMenuOpen ? 'absolute inset-y-0 left-0 shadow-2xl flex' : 'hidden md:flex'
          }`}
        >
          <div className="p-4 space-y-5 flex-1 flex flex-col overflow-y-auto">
            {/* Sidebar Brand & Collapse */}
            <div className="flex items-center justify-between px-1">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 rounded-lg bg-[#3B4BF7] flex items-center justify-center text-white text-xs font-bold">
                  D
                </div>
                <span className="font-bold text-sm text-slate-900 tracking-tight">Dokumin</span>
              </div>
              <button
                onClick={() => setMobileMenuOpen(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100"
                title="Collapse Sidebar"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
            </div>

            {/* Sidebar Search Bar */}
            <div className="relative">
              <div className="flex items-center justify-between px-3 py-2 bg-slate-50 border border-slate-200/80 rounded-xl text-xs text-slate-400">
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <Search className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                  <input
                    type="text"
                    value={searchFilter}
                    onChange={(e) => setSearchFilter(e.target.value)}
                    placeholder="Search docs..."
                    className="bg-transparent text-xs text-slate-800 placeholder:text-slate-400 focus:outline-none w-full"
                  />
                </div>
                <span className="text-[10px] bg-white border border-slate-200 text-slate-400 px-1 rounded font-mono shadow-2xs shrink-0">
                  ⌘ F
                </span>
              </div>
            </div>

            {/* CATEGORY: MAIN */}
            <div className="space-y-1">
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 px-3">
                MAIN
              </span>

              <button
                onClick={() => {
                  setActiveTab('search');
                  setMobileMenuOpen(false);
                }}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-medium transition cursor-pointer ${
                  activeTab === 'search'
                    ? 'bg-[#3B4BF7] text-white shadow-sm shadow-[#3B4BF7]/20 font-semibold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <LayoutDashboard className={`w-4 h-4 ${activeTab === 'search' ? 'text-white' : 'text-slate-500'}`} />
                  <span>Overview</span>
                </div>
                {activeTab === 'search' && <ChevronRight className="w-3.5 h-3.5 text-white/80" />}
              </button>

              <button
                onClick={() => {
                  setActiveTab('upload');
                  setMobileMenuOpen(false);
                }}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-medium transition cursor-pointer ${
                  activeTab === 'upload'
                    ? 'bg-[#3B4BF7] text-white shadow-sm shadow-[#3B4BF7]/20 font-semibold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <FileEdit className={`w-4 h-4 ${activeTab === 'upload' ? 'text-white' : 'text-slate-500'}`} />
                  <span>Editor</span>
                </div>
              </button>
            </div>

            {/* CATEGORY: PRODUCTS */}
            <div className="space-y-1">
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 px-3">
                PRODUCTS
              </span>

              <button
                onClick={() => {
                  setActiveTab('ask');
                  setMobileMenuOpen(false);
                }}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-medium transition cursor-pointer ${
                  activeTab === 'ask'
                    ? 'bg-[#3B4BF7] text-white shadow-sm shadow-[#3B4BF7]/20 font-semibold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <Sparkles className={`w-4 h-4 ${activeTab === 'ask' ? 'text-white' : 'text-indigo-500'}`} />
                  <span>Ask DocuMind</span>
                </div>
                <span className={`text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded ${
                  activeTab === 'ask' ? 'bg-white/20 text-white' : 'bg-indigo-50 text-indigo-600 border border-indigo-200/60'
                }`}>
                  RAG
                </span>
              </button>

              <button
                onClick={() => {
                  setActiveTab('monitoring');
                  fetchMonitoring();
                  setMobileMenuOpen(false);
                }}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-medium transition cursor-pointer ${
                  activeTab === 'monitoring'
                    ? 'bg-[#3B4BF7] text-white shadow-sm shadow-[#3B4BF7]/20 font-semibold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <Activity className={`w-4 h-4 ${activeTab === 'monitoring' ? 'text-white' : 'text-emerald-500'}`} />
                  <span>Analytics</span>
                </div>
                <span className={`text-[10px] font-mono ${activeTab === 'monitoring' ? 'text-white/80' : 'text-slate-400'}`}>
                  Admin
                </span>
              </button>

              <button
                onClick={() => {
                  setActiveTab('documents');
                  setMobileMenuOpen(false);
                }}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-medium transition cursor-pointer ${
                  activeTab === 'documents'
                    ? 'bg-[#3B4BF7] text-white shadow-sm shadow-[#3B4BF7]/20 font-semibold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <FileText className={`w-4 h-4 ${activeTab === 'documents' ? 'text-white' : 'text-slate-500'}`} />
                  <span>Documents</span>
                </div>
                <span className={`text-xs font-mono px-2 py-0.5 rounded-full ${
                  activeTab === 'documents' ? 'bg-white/20 text-white' : 'bg-slate-100 text-slate-500'
                }`}>
                  {documents.length}
                </span>
              </button>

              <div className="w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-medium text-slate-500 hover:bg-slate-50 cursor-pointer">
                <div className="flex items-center gap-2.5">
                  <Layers className="w-4 h-4 text-slate-400" />
                  <span>Add-ons</span>
                </div>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-400 font-mono">v0.5.0</span>
              </div>
            </div>

            {/* CATEGORY: TOOLS */}
            <div className="space-y-1">
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 px-3">
                TOOLS
              </span>

              <div className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-50 cursor-pointer">
                <Settings className="w-4 h-4 text-slate-400" />
                <span>Settings</span>
              </div>

              <a
                href="https://documind-ai-backend-jhug.onrender.com/docs"
                target="_blank"
                rel="noreferrer"
                className="w-full flex items-center justify-between px-3 py-2 rounded-xl text-xs font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-50 cursor-pointer"
              >
                <div className="flex items-center gap-2.5">
                  <BookOpen className="w-4 h-4 text-slate-400" />
                  <span>Documentation</span>
                </div>
                <ExternalLink className="w-3 h-3 text-slate-400" />
              </a>

              <div className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-50 cursor-pointer">
                <Users className="w-4 h-4 text-slate-400" />
                <span>Invite Members</span>
              </div>

              <div className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-50 cursor-pointer">
                <HelpCircle className="w-4 h-4 text-slate-400" />
                <span>Support</span>
              </div>
            </div>

            {/* Document Knowledge Base Filter Dropdown */}
            <div className="pt-2 border-t border-slate-100 space-y-2">
              <div className="flex items-center justify-between px-1">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                  Knowledge Base
                </span>
                <button
                  onClick={() => setActiveTab('upload')}
                  className="text-[11px] text-[#3B4BF7] hover:underline font-medium flex items-center gap-1 cursor-pointer"
                  title="Upload new document"
                >
                  <Plus className="w-3 h-3" />
                  <span>Add PDF</span>
                </button>
              </div>

              {filteredDocs.length > 0 && (
                <div className="space-y-1 max-h-36 overflow-y-auto pr-1">
                  {filteredDocs.slice(0, 5).map((doc) => {
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
                            ? 'bg-indigo-50 border-indigo-200 text-[#3B4BF7] font-medium'
                            : 'bg-white border-slate-100 hover:bg-slate-50 text-slate-700'
                        }`}
                      >
                        <div className="flex items-center gap-2 min-w-0 flex-1">
                          <FileText className="w-3.5 h-3.5 text-indigo-500 shrink-0" />
                          <span className="truncate text-xs">{doc.title}</span>
                        </div>
                        <span className="text-[9px] font-mono px-1 rounded bg-slate-100 text-slate-500 ml-1 shrink-0">
                          {doc.page_count}p
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Bottom Onboarding Readiness Card (Dokumin reference) */}
            <div className="p-3.5 rounded-2xl border border-slate-200/80 bg-slate-50/70 space-y-2.5">
              <div className="flex items-center justify-between text-xs font-semibold text-slate-800">
                <div className="flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-[#3B4BF7]" />
                  <span>2 steps left out of 5</span>
                </div>
                <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
              </div>
              <div className="w-full bg-slate-200/70 rounded-full h-1.5 overflow-hidden">
                <div className="bg-[#3B4BF7] h-1.5 rounded-full" style={{ width: '40%' }}></div>
              </div>
              <div className="text-[11px] font-medium text-slate-400">Get your docs ready</div>
              <div className="space-y-1.5 text-xs">
                <div className="flex items-center gap-2 text-slate-700 font-medium">
                  <span className="w-3.5 h-3.5 rounded-full bg-[#3B4BF7] text-white flex items-center justify-center text-[9px] font-bold">
                    ✓
                  </span>
                  <span>Add custom domain</span>
                </div>
                <div className="flex items-center gap-2 text-slate-400">
                  <span className="w-3.5 h-3.5 rounded-full border border-slate-300"></span>
                  <span>Connect GitHub Repo</span>
                </div>
              </div>
            </div>

            {/* Workspace Selector Pill (Dokumin reference) */}
            <div className="p-2.5 rounded-xl border border-slate-200/80 bg-white flex items-center justify-between text-xs font-semibold text-slate-800 hover:bg-slate-50 transition cursor-pointer shadow-2xs">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 rounded-full bg-slate-900 text-white flex items-center justify-center font-bold text-[10px]">
                  P
                </div>
                <span>Pickolab Studio</span>
              </div>
              <div className="flex flex-col text-slate-400 text-[8px] leading-tight">
                <span>▲</span>
                <span>▼</span>
              </div>
            </div>
          </div>

          {/* User Account / Tenant Status Footer */}
          <div className="p-3 border-t border-slate-200/80 bg-slate-50/60 text-xs text-slate-500 flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
              Tenant Isolation
            </span>
            <span className="font-mono text-[11px] text-slate-600">
              {currentUser ? `User #${currentUser.id}` : 'Guest'}
            </span>
          </div>
        </aside>

        {/* Central Workspace Area */}
        <main className="flex-1 flex flex-col overflow-hidden bg-[#F9F9FB]">
          {/* TAB 1: ASK DOCUMIND (Primary RAG Interface) */}
          {activeTab === 'ask' && (
            <div className="flex-1 flex overflow-hidden">
              {/* Collapsible Conversation History Sidebar */}
              {showConversationSidebar && (
                <div className="w-64 sm:w-72 border-r border-slate-200/80 bg-white flex flex-col shrink-0 overflow-hidden">
                  {/* Sidebar Header & New Chat button */}
                  <div className="p-3 border-b border-slate-200/80 flex items-center justify-between gap-2">
                    <button
                      onClick={handleNewConversation}
                      disabled={isCreatingConv}
                      className="flex-1 py-2 px-3 rounded-xl bg-[#3B4BF7] hover:bg-[#2F3EE0] text-white font-medium text-xs flex items-center justify-center gap-2 shadow-sm shadow-[#3B4BF7]/20 transition cursor-pointer disabled:opacity-50"
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
                        <RefreshCw className="w-3.5 h-3.5 animate-spin text-[#3B4BF7]" />
                        <span>Loading chats...</span>
                      </div>
                    ) : conversationError ? (
                      <div className="p-4 text-center text-xs text-rose-500">
                        {conversationError}
                      </div>
                    ) : conversations.length === 0 ? (
                      <div className="p-6 text-center text-xs text-slate-400 space-y-2">
                        <MessageSquare className="w-6 h-6 text-slate-300 mx-auto" />
                        <p>No conversations yet.</p>
                        <p className="text-[11px] text-slate-400">Start a new chat to begin.</p>
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
                                ? 'bg-indigo-50 border-indigo-200 text-[#3B4BF7] font-semibold'
                                : 'bg-white hover:bg-slate-50 border-slate-200/70 text-slate-700'
                            }`}
                          >
                            <div className="flex items-start justify-between gap-1.5">
                              <div className="flex-1 min-w-0 space-y-1">
                                <p className="font-medium truncate text-xs leading-snug">
                                  {conv.title || 'Untitled Chat'}
                                </p>
                                <div className="flex items-center gap-1.5 text-[10px] text-slate-400">
                                  {conv.document_scope_invalidated ? (
                                    <span className="text-amber-600 font-semibold flex items-center gap-1">
                                      <AlertTriangle className="w-2.5 h-2.5" /> Scope Invalidated
                                    </span>
                                  ) : scopedDoc ? (
                                    <span className="truncate max-w-[140px] text-indigo-600">
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
                                  <div className="flex items-center gap-1 bg-rose-50 border border-rose-200 rounded-md p-0.5">
                                    <button
                                      onClick={() => handleDeleteConversation(conv.id)}
                                      className="px-1.5 py-0.5 rounded bg-rose-600 text-white hover:bg-rose-500 text-[10px] font-semibold"
                                      title="Confirm Delete"
                                    >
                                      Delete
                                    </button>
                                    <button
                                      onClick={() => setDeleteConfirmId(null)}
                                      className="p-0.5 text-slate-400 hover:text-slate-600"
                                      title="Cancel"
                                    >
                                      <X className="w-3 h-3" />
                                    </button>
                                  </div>
                                ) : (
                                  <button
                                    onClick={() => setDeleteConfirmId(conv.id)}
                                    className={`p-1 rounded hover:bg-slate-100 text-slate-400 hover:text-rose-600 transition ${
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
              <div className="flex-1 flex flex-col overflow-hidden border-r border-slate-200/80 bg-[#F9F9FB]">
                {/* Conversation Header & Context Filter */}
                <div className="h-14 border-b border-slate-200/80 bg-white px-6 flex items-center justify-between shrink-0 shadow-2xs">
                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => setShowConversationSidebar(!showConversationSidebar)}
                      className={`p-1.5 rounded-xl border transition cursor-pointer ${
                        showConversationSidebar
                          ? 'bg-indigo-50 border-indigo-200 text-[#3B4BF7]'
                          : 'bg-slate-50 border-slate-200 text-slate-500 hover:bg-slate-100'
                      }`}
                      title="Toggle Chats Sidebar"
                      aria-label="Toggle Chats Sidebar"
                    >
                      <PanelLeft className="w-4 h-4" />
                    </button>
                    <div className="w-7 h-7 rounded-xl bg-indigo-50 border border-indigo-100 text-[#3B4BF7] flex items-center justify-center">
                      <Sparkles className="w-4 h-4 text-[#3B4BF7]" />
                    </div>
                    <div>
                      <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
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
                        className="bg-slate-50 border border-slate-200/80 rounded-xl px-2.5 py-1 text-xs text-slate-700 focus:outline-none focus:border-[#3B4BF7] cursor-pointer max-w-[200px] truncate"
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
                      className="inline-flex items-center gap-1.5 px-3 py-1 rounded-xl bg-slate-50 hover:bg-slate-100 text-slate-600 text-xs font-medium border border-slate-200/80 transition cursor-pointer"
                      title="Clear chat view"
                      aria-label="Clear chat"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                      <span className="hidden sm:inline">Clear Chat</span>
                    </button>

                    {/* Toggle Source Panel on Desktop */}
                    <button
                      onClick={() => setShowSourcePanel(!showSourcePanel)}
                      className={`hidden lg:flex items-center gap-1.5 px-3 py-1 rounded-xl text-xs font-medium border transition cursor-pointer ${
                        showSourcePanel
                          ? 'bg-indigo-50 border-indigo-200 text-[#3B4BF7]'
                          : 'bg-slate-50 border-slate-200 text-slate-500 hover:bg-slate-100'
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
                  <div className="mx-6 mt-4 p-3.5 rounded-xl bg-amber-50 border border-amber-200 text-amber-800 text-xs flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
                      <span>This conversation's document scope was invalidated because the referenced document was deleted.</span>
                    </div>
                    <button
                      onClick={() => handleCreateConversation()}
                      className="px-2.5 py-1 rounded-lg bg-amber-600 hover:bg-amber-500 text-white font-semibold text-[11px] shrink-0"
                    >
                      Start New Chat
                    </button>
                  </div>
                )}

                {/* Loading History Indicator */}
                {loadingHistory && (
                  <div className="p-3 text-center text-xs text-[#3B4BF7] bg-indigo-50 border-b border-indigo-100 flex items-center justify-center gap-2">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin text-[#3B4BF7]" />
                    <span>Loading conversation history...</span>
                  </div>
                )}

                {/* Scrollable Conversation Stream */}
                <div ref={chatScrollRef} className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
                  {/* Empty Welcome Screen with Suggested Questions */}
                  {chatMessages.length === 0 ? (
                    <div className="max-w-2xl mx-auto py-8 sm:py-12 space-y-8">
                      <div className="text-center space-y-3">
                        <div className="w-14 h-14 rounded-2xl bg-indigo-50 border border-indigo-100 flex items-center justify-center mx-auto shadow-sm">
                          <Sparkles className="w-7 h-7 text-[#3B4BF7]" />
                        </div>
                        <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
                          Ask DocuMind Intelligence
                        </h1>
                        <p className="text-sm text-slate-500 leading-relaxed max-w-lg mx-auto">
                          Ask natural-language questions across your uploaded enterprise PDFs. Every answer is
                          synthesized with Gemini, backed by pgvector vector similarity, and attributed with
                          verifiable page citations.
                        </p>

                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-slate-200 text-xs text-slate-600 shadow-2xs">
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                          <span>{completedDocs.length} of {documents.length} documents indexed and ready</span>
                        </div>
                      </div>

                      {/* Suggested Questions */}
                      <div className="space-y-3">
                        <span className="text-xs font-bold uppercase tracking-wider text-slate-400 block text-center">
                          Suggested Inquiries
                        </span>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                          {SUGGESTED_QUESTIONS.map((q, idx) => (
                            <button
                              key={idx}
                              onClick={() => handleAskQuestion(q)}
                              className="p-4 rounded-2xl border border-slate-200/80 bg-white hover:border-[#3B4BF7] hover:shadow-sm text-left text-xs text-slate-700 transition group space-y-1.5 cursor-pointer"
                            >
                              <div className="flex items-center gap-2 text-[#3B4BF7]">
                                <Sparkles className="w-3.5 h-3.5" />
                                <span className="font-semibold text-[11px] uppercase tracking-wider">Example</span>
                              </div>
                              <p className="leading-snug text-slate-800">{q}</p>
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
                          <div className="max-w-xl bg-[#3B4BF7] text-white rounded-2xl rounded-tr-sm px-4 py-3 shadow-sm text-xs sm:text-sm leading-relaxed">
                            {msg.content}
                          </div>
                        ) : (
                          // Assistant Message Card
                          <div className="w-full max-w-3xl rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm space-y-4">
                            {/* Assistant Card Header */}
                            <div className="flex items-center justify-between border-b border-slate-100 pb-3 text-xs">
                              <div className="flex items-center gap-2">
                                <div className="w-6 h-6 rounded-lg bg-indigo-50 text-[#3B4BF7] flex items-center justify-center font-bold text-xs">
                                  <Sparkles className="w-3.5 h-3.5 text-[#3B4BF7]" />
                                </div>
                                <span className="font-semibold text-slate-900">DocuMind AI</span>
                                {msg.model_used && (
                                  <span className="px-2 py-0.5 rounded-full bg-slate-100 text-[10px] text-slate-600 font-mono border border-slate-200/80">
                                    {msg.model_used}
                                  </span>
                                )}
                              </div>

                              <div className="flex items-center gap-3 text-[11px] text-slate-400">
                                {msg.total_latency_ms && (
                                  <span className="font-mono text-emerald-600 font-medium">
                                    {msg.total_latency_ms}ms
                                  </span>
                                )}
                                {msg.token_usage?.total_tokens && (
                                  <span className="font-mono text-amber-600 hidden sm:inline">
                                    {msg.token_usage.total_tokens} tokens
                                  </span>
                                )}
                              </div>
                            </div>

                            {/* Abstention / Insufficient Evidence Banner */}
                            {msg.insufficient_evidence && (
                              <div className="p-3 rounded-xl bg-amber-50 border border-amber-200 text-xs text-amber-800 flex items-start gap-2.5">
                                <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                                <div>
                                  <span className="font-semibold">Abstention Alert: </span>
                                  The system abstained from answering because the retrieved document context does
                                  not contain sufficient verifiable evidence.
                                </div>
                              </div>
                            )}

                            {/* Error Message Notice */}
                            {msg.error && (
                              <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-600 flex items-start gap-2.5">
                                <AlertTriangle className="w-4 h-4 text-rose-500 shrink-0 mt-0.5" />
                                <div>
                                  <span className="font-semibold">Error: </span>
                                  {msg.error}
                                </div>
                              </div>
                            )}

                            {/* Answer Body */}
                            <div className="text-xs sm:text-sm text-slate-800 leading-relaxed font-sans whitespace-pre-wrap">
                              {msg.content}
                            </div>

                            {/* Citations Badges List */}
                            {msg.citations && msg.citations.length > 0 && (
                              <div className="pt-3 border-t border-slate-100 space-y-2">
                                <div className="flex items-center justify-between text-xs text-slate-500">
                                  <span className="font-semibold text-slate-700 flex items-center gap-1.5">
                                    <BookOpen className="w-3.5 h-3.5 text-[#3B4BF7]" />
                                    Source Citations ({msg.citations.length})
                                  </span>
                                  <span className="text-[11px] text-slate-400">
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
                                        className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium border transition cursor-pointer ${
                                          isSelected
                                            ? 'bg-indigo-50 border-[#3B4BF7] text-[#3B4BF7] shadow-xs'
                                            : 'bg-slate-50 border-slate-200/80 text-slate-700 hover:border-slate-300 hover:bg-slate-100'
                                        }`}
                                      >
                                        <span className="w-4 h-4 rounded-full bg-indigo-100 text-[#3B4BF7] font-mono text-[10px] flex items-center justify-center font-bold">
                                          {cIdx + 1}
                                        </span>
                                        <span className="truncate max-w-[140px] sm:max-w-[200px]">
                                          {c.document_title}
                                        </span>
                                        <span className="px-1.5 py-0.2 rounded bg-indigo-50 text-[#3B4BF7] text-[10px] font-mono border border-indigo-200/60">
                                          p.{c.page_number}
                                        </span>
                                      </button>
                                    );
                                  })}
                                </div>
                              </div>
                            )}

                            {/* Assistant Card Footer Actions */}
                            <div className="flex items-center justify-between pt-2 border-t border-slate-100 text-xs text-slate-400">
                              <div className="text-[11px] font-mono">
                                {new Date(msg.timestamp).toLocaleTimeString()}
                              </div>
                              <div className="flex items-center gap-2">
                                <button
                                  onClick={() => handleCopyAnswer(msg)}
                                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg hover:bg-slate-100 text-slate-500 hover:text-slate-800 transition cursor-pointer"
                                  title="Copy Answer text"
                                >
                                  {copiedMessageId === msg.id ? (
                                    <>
                                      <Check className="w-3.5 h-3.5 text-emerald-600" />
                                      <span className="text-emerald-600 text-[11px]">Copied</span>
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
                                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg hover:bg-slate-100 text-slate-500 hover:text-slate-800 transition cursor-pointer"
                                    title="Copy Citations references"
                                  >
                                    {copiedCitationsId === msg.id ? (
                                      <>
                                        <Check className="w-3.5 h-3.5 text-emerald-600" />
                                        <span className="text-emerald-600 text-[11px]">Copied</span>
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
                    <div className="w-full max-w-3xl rounded-2xl border border-indigo-200 bg-white p-5 space-y-3 shadow-sm animate-pulse">
                      <div className="flex items-center gap-2 text-xs text-[#3B4BF7]">
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span className="font-semibold">Synthesizing Answer with Gemini...</span>
                      </div>
                      <p className="text-xs text-slate-500">
                        Retrieving passages from pgvector, validating citations, and checking evidence sufficiency.
                      </p>
                    </div>
                  )}

                  {/* RAG Error Display */}
                  {ragError && !askingRAG && (
                    <div className="p-4 rounded-2xl bg-rose-50 border border-rose-200 text-xs text-rose-600 flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4 text-rose-500 shrink-0" />
                        <span>{ragError}</span>
                      </div>
                      <button
                        onClick={() => setRagError(null)}
                        className="text-slate-400 hover:text-slate-600"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  )}
                </div>

                {/* Bottom Input Prompt Bar */}
                <div className="p-4 bg-white border-t border-slate-200/80 shrink-0">
                  <div className="max-w-3xl mx-auto space-y-2">
                    <div className="relative rounded-2xl border border-slate-200/90 bg-white focus-within:border-[#3B4BF7] focus-within:ring-1 focus-within:ring-[#3B4BF7]/20 transition shadow-xs">
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
                        className="w-full bg-transparent px-4 py-3 text-xs sm:text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none resize-none"
                      />

                      <div className="flex items-center justify-between px-4 pb-2 text-[11px] text-slate-400">
                        <div className="flex items-center gap-2">
                          <span>Enter to submit</span>
                          <span>&bull;</span>
                          <span>Shift+Enter for new line</span>
                        </div>

                        <button
                          onClick={() => handleAskQuestion()}
                          disabled={askingRAG || !askInput.trim()}
                          aria-label="Submit question"
                          className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-xl bg-[#3B4BF7] hover:bg-[#2F3EE0] disabled:opacity-40 text-white font-semibold text-xs transition cursor-pointer shadow-sm shadow-[#3B4BF7]/25"
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

                    <div className="flex items-center justify-between text-[10px] text-slate-400 px-2">
                      <span>DocuMind AI answers are strictly grounded in your authorized document context.</span>
                      {chatMessages.length > 0 && (
                        <button
                          onClick={handleClearConversation}
                          className="hover:text-slate-600 underline cursor-pointer"
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
                <aside className="w-80 sm:w-96 border-l border-slate-200/80 bg-white flex flex-col justify-between shrink-0">
                  <div className="p-5 space-y-5 flex-1 flex flex-col overflow-y-auto">
                    {/* Source Inspector Header */}
                    <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                      <div className="flex items-center gap-2">
                        <BookOpen className="w-4 h-4 text-[#3B4BF7]" />
                        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800">
                          Source Inspector
                        </h3>
                      </div>
                      <button
                        onClick={() => setShowSourcePanel(false)}
                        className="p-1 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-700"
                        title="Close Inspector"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>

                    {/* Source Detail Card */}
                    {activeCitation ? (
                      <div className="space-y-4">
                        <div className="p-4 rounded-xl border border-indigo-100 bg-indigo-50/40 space-y-3">
                          <div className="flex items-start justify-between gap-2">
                            <div>
                              <span className="text-[10px] font-semibold uppercase tracking-wider text-[#3B4BF7] block mb-0.5">
                                Verified Source
                              </span>
                              <h4 className="text-sm font-bold text-slate-900 leading-tight">
                                {activeCitation.document_title}
                              </h4>
                            </div>
                            <span className="px-2 py-0.5 rounded-full bg-white text-[#3B4BF7] border border-indigo-200 text-xs font-mono font-semibold">
                              Page {activeCitation.page_number}
                            </span>
                          </div>

                          <div className="flex items-center justify-between text-[11px] text-slate-500 pt-1 border-t border-indigo-100">
                            <span>Hybrid Similarity:</span>
                            <span className="font-mono text-emerald-600 font-semibold">
                              {(activeCitation.score * 100).toFixed(1)}% match
                            </span>
                          </div>
                        </div>

                        {/* Passage Excerpt Block */}
                        <div className="space-y-1.5">
                          <span className="text-[11px] font-semibold text-slate-500 block">
                            Retrieved Context Excerpt
                          </span>
                          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 text-xs text-slate-700 leading-relaxed font-sans relative">
                            <Quote className="w-4 h-4 text-[#3B4BF7]/40 absolute top-2 right-2" />
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
                            className="w-full py-2.5 rounded-xl bg-[#3B4BF7] hover:bg-[#2F3EE0] text-white font-semibold text-xs transition flex items-center justify-center gap-2 cursor-pointer shadow-sm shadow-[#3B4BF7]/20 disabled:opacity-50"
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
                            className="w-full py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-medium border border-slate-200 transition flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
                          >
                            <Download className="w-3.5 h-3.5 text-[#3B4BF7]" />
                            <span>Download Full Document</span>
                          </button>
                        </div>
                      </div>
                    ) : (
                      // Empty Inspector Placeholder
                      <div className="p-8 text-center border border-dashed border-slate-200 rounded-2xl my-6 space-y-2">
                        <BookOpen className="w-8 h-8 text-slate-300 mx-auto" />
                        <h4 className="text-xs font-semibold text-slate-700">
                          No Citation Selected
                        </h4>
                        <p className="text-[11px] text-slate-400 leading-relaxed">
                          Click any numbered citation badge <span className="text-[#3B4BF7] font-mono">[1]</span> in an AI
                          answer to inspect the exact underlying document page and excerpt passage.
                        </p>
                      </div>
                    )}
                  </div>

                  <div className="p-3 border-t border-slate-100 bg-slate-50/50 text-[11px] text-slate-400 text-center">
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
              <div className="relative overflow-hidden rounded-2xl border border-slate-200/80 bg-white p-6 sm:p-8 shadow-sm">
                <div className="max-w-2xl">
                  <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-semibold mb-4">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                    Stage 3: PDF Ingestion & Semantic Vector Search Active
                  </div>
                  <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight mb-2">
                    Document Knowledge Base
                  </h1>
                  <p className="text-slate-500 text-xs sm:text-sm leading-relaxed mb-6">
                    Upload technical PDFs, policy manuals, or financial filings. Text is extracted page-by-page,
                    chunked with punctuation snapping, and stored as 1536-dimensional embeddings in pgvector.
                  </p>

                  <div className="flex flex-wrap items-center gap-3">
                    <button
                      onClick={() => setActiveTab('ask')}
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-[#3B4BF7] hover:bg-[#2F3EE0] text-white text-xs font-semibold transition cursor-pointer shadow-sm shadow-[#3B4BF7]/20"
                    >
                      <Sparkles className="w-4 h-4 text-white" />
                      Ask Questions
                    </button>
                    <button
                      onClick={() => {
                        if (fileInputRef.current) fileInputRef.current.click();
                      }}
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200/80 text-xs font-semibold transition cursor-pointer"
                    >
                      <UploadCloud className="w-4 h-4 text-[#3B4BF7]" />
                      Upload PDF
                    </button>
                    <a
                      href="https://documind-ai-backend-jhug.onrender.com/docs"
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold border border-slate-200/80 transition"
                    >
                      <ExternalLink className="w-3.5 h-3.5 text-slate-500" />
                      Swagger API
                    </a>
                  </div>
                </div>
              </div>

              {/* Upload Dropzone Form */}
              <div className="rounded-2xl border border-slate-200/80 bg-white p-6 space-y-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <UploadCloud className="w-4 h-4 text-[#3B4BF7]" />
                    <h3 className="text-sm font-semibold text-slate-900">Upload & Ingest Enterprise PDF</h3>
                  </div>
                  <span className="text-xs text-slate-400">
                    Max 15MB &bull; Automatic chunking & pgvector indexing
                  </span>
                </div>

                {uploadError && (
                  <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-600 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 shrink-0" />
                    <span>{uploadError}</span>
                  </div>
                )}

                {uploadSuccess && (
                  <div className="p-3 rounded-xl bg-emerald-50 border border-emerald-200 text-xs text-emerald-700 flex items-center gap-2">
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
                    className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition ${
                      uploadFile
                        ? 'border-[#3B4BF7] bg-indigo-50/30'
                        : 'border-slate-200 hover:border-[#3B4BF7] hover:bg-slate-50'
                    }`}
                  >
                    <UploadCloud className="w-10 h-10 text-[#3B4BF7] mx-auto mb-2" />
                    {uploadFile ? (
                      <div className="space-y-1">
                        <p className="text-sm font-semibold text-slate-900">{uploadFile.name}</p>
                        <p className="text-xs text-[#3B4BF7]">
                          {(uploadFile.size / (1024 * 1024)).toFixed(2)} MB &bull; Click to change file
                        </p>
                      </div>
                    ) : (
                      <div className="space-y-1">
                        <p className="text-sm font-medium text-slate-700">
                          Drag and drop your PDF here, or <span className="text-[#3B4BF7] underline">browse files</span>
                        </p>
                        <p className="text-xs text-slate-400">
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
                        className="flex-1 bg-slate-50 border border-slate-200/80 rounded-xl px-3 py-2 text-xs text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-[#3B4BF7] focus:bg-white"
                      />
                      <button
                        type="submit"
                        disabled={uploading}
                        className="px-5 py-2.5 rounded-xl bg-[#3B4BF7] hover:bg-[#2F3EE0] disabled:opacity-50 text-white font-semibold text-xs transition flex items-center justify-center gap-2 cursor-pointer shadow-sm shadow-[#3B4BF7]/20"
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
                    <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                      <div
                        className="bg-[#3B4BF7] h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                  )}
                </form>
              </div>

              {/* Document List Table */}
              <div className="rounded-2xl border border-slate-200/80 bg-white p-6 space-y-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-900">
                    Authorized Documents ({documents.length})
                  </h3>
                  <button
                    onClick={() => fetchDocuments()}
                    disabled={loadingDocs}
                    className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-700 transition"
                    title="Refresh document list"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${loadingDocs ? 'animate-spin' : ''}`} />
                  </button>
                </div>

                {downloadError && (
                  <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-600 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 shrink-0" />
                      <span>{downloadError}</span>
                    </div>
                    <button
                      type="button"
                      onClick={() => setDownloadError(null)}
                      className="text-slate-400 hover:text-slate-600"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                )}

                {documents.length === 0 ? (
                  <div className="p-8 text-center border border-dashed border-slate-200 rounded-2xl text-xs text-slate-400">
                    No documents uploaded yet. Upload your first PDF to begin.
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {documents.map((doc) => (
                      <div
                        key={doc.id}
                        className="p-4 rounded-2xl border border-slate-200/80 bg-white hover:border-[#3B4BF7]/40 hover:shadow-xs transition space-y-3 flex flex-col justify-between"
                      >
                        <div className="space-y-2">
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex items-center gap-2 min-w-0">
                              <FileText className="w-4 h-4 text-rose-500 shrink-0" />
                              <h4 className="text-xs font-bold text-slate-800 truncate">{doc.title}</h4>
                            </div>
                            <span
                              className={`text-[9px] font-mono px-1.5 py-0.5 rounded-full font-semibold uppercase ${
                                doc.status === 'completed'
                                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                                  : doc.status === 'processing'
                                  ? 'bg-indigo-50 text-[#3B4BF7] border border-indigo-200 animate-pulse'
                                  : 'bg-rose-50 text-rose-600 border border-rose-200'
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
                              <span className="text-rose-500 truncate max-w-[140px]">
                                {doc.error_message || 'Processing failed'}
                              </span>
                              <button
                                type="button"
                                onClick={() => handleRetryProcess(doc.id)}
                                className="text-[#3B4BF7] hover:underline font-semibold flex items-center gap-1 cursor-pointer"
                              >
                                <RefreshCw className="w-3 h-3" /> Retry
                              </button>
                            </div>
                          )}
                        </div>

                        <div className="flex items-center gap-2 pt-2 border-t border-slate-100">
                          <button
                            type="button"
                            onClick={() => {
                              setFilterDocId(doc.id);
                              setActiveTab('ask');
                            }}
                            className="flex-1 py-1.5 rounded-xl bg-indigo-50 hover:bg-indigo-100 text-[#3B4BF7] text-xs font-medium border border-indigo-200/60 transition text-center cursor-pointer"
                          >
                            Ask Questions
                          </button>
                          <button
                            type="button"
                            onClick={() => handleDownloadDocument(doc.id, doc.filename || doc.title)}
                            disabled={downloadingDocId === doc.id}
                            className="p-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-600 transition cursor-pointer"
                            title="Download PDF"
                          >
                            <Download className="w-3.5 h-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => handleDeleteDocument(doc.id, doc.title)}
                            className="p-1.5 rounded-xl hover:bg-rose-50 text-slate-400 hover:text-rose-600 transition cursor-pointer"
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

          {/* TAB 3: SEMANTIC VECTOR SEARCH INSPECTOR / OVERVIEW DASHBOARD */}
          {activeTab === 'search' && (
            <div className="flex-1 overflow-y-auto p-6 sm:p-8 space-y-6">
              {/* Dokumin Dashboard Welcome Greeting Header */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
                    Welcome, {currentUser ? currentUser.username : 'Charlie'} !
                  </h1>
                  <p className="text-slate-500 text-xs sm:text-sm mt-1">
                    Let's take a look at your latest activity.
                  </p>
                </div>

                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setActiveTab('upload')}
                    className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-semibold transition cursor-pointer bg-[#3B4BF7] hover:bg-[#2F3EE0] text-white shadow-sm shadow-[#3B4BF7]/25"
                  >
                    <Plus className="w-4 h-4" />
                    <span>Create New Page +</span>
                  </button>
                </div>
              </div>

              {/* Status Banner & Navigation Action Bar */}
              <div className="flex flex-wrap items-center justify-between gap-3 p-4 rounded-2xl bg-white border border-slate-200/80 shadow-sm">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-semibold">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                  Stage 3: PDF Ingestion & Semantic Vector Search Active
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <button
                    onClick={() => setActiveTab('ask')}
                    className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-semibold transition cursor-pointer bg-slate-100 hover:bg-slate-200 text-slate-700"
                  >
                    <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
                    <span>Ask DocuMind</span>
                  </button>
                  <button
                    onClick={() => setActiveTab('search')}
                    className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-semibold transition cursor-pointer bg-[#3B4BF7] text-white shadow-sm shadow-[#3B4BF7]/20"
                  >
                    <Search className="w-3.5 h-3.5" />
                    <span>Semantic Search</span>
                  </button>
                  <button
                    onClick={() => setActiveTab('upload')}
                    className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-semibold transition cursor-pointer bg-slate-100 hover:bg-slate-200 text-slate-700"
                  >
                    <UploadCloud className="w-3.5 h-3.5 text-slate-500" />
                    <span>Upload Document</span>
                  </button>
                  <button
                    onClick={() => {
                      setActiveTab('monitoring');
                      fetchMonitoring();
                    }}
                    className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-semibold transition cursor-pointer bg-slate-100 hover:bg-slate-200 text-slate-700"
                  >
                    <Activity className="w-3.5 h-3.5 text-emerald-600" />
                    <span>Observability</span>
                  </button>
                </div>
              </div>

              {/* Selected Document Floating Card */}
              {selectedDoc && (
                <div className="rounded-2xl border border-indigo-200 bg-white p-5 shadow-sm space-y-4">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-indigo-50 border border-indigo-100 text-[#3B4BF7] flex items-center justify-center">
                        <FileText className="w-5 h-5 text-[#3B4BF7]" />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h2 className="text-base font-bold text-slate-900">{selectedDoc.title}</h2>
                          <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                            {selectedDoc.status}
                          </span>
                        </div>
                        <p className="text-xs text-slate-500">{selectedDoc.filename}</p>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => {
                          setFilterDocId(selectedDoc.id);
                          setActiveTab('ask');
                        }}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-[#3B4BF7] hover:bg-[#2F3EE0] text-white text-xs font-medium transition cursor-pointer shadow-sm shadow-[#3B4BF7]/20"
                      >
                        <Sparkles className="w-3.5 h-3.5" />
                        <span>Ask Questions</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => handleDownloadDocument(selectedDoc.id, selectedDoc.filename || selectedDoc.title)}
                        disabled={downloadingDocId === selectedDoc.id}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-medium border border-slate-200 transition cursor-pointer disabled:opacity-50"
                      >
                        <Download className={`w-3.5 h-3.5 text-[#3B4BF7] ${downloadingDocId === selectedDoc.id ? 'animate-bounce' : ''}`} />
                        <span>{downloadingDocId === selectedDoc.id ? 'Downloading...' : 'Download PDF'}</span>
                      </button>
                      <button
                        onClick={() => handleDeleteDocument(selectedDoc.id, selectedDoc.title)}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-rose-50 hover:bg-rose-100 text-rose-600 text-xs font-medium border border-rose-200 transition cursor-pointer"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                        <span>Delete</span>
                      </button>
                      <button
                        onClick={() => setSelectedDoc(null)}
                        className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Structured Two-Column Grid Layout (Dokumin Reference) */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* LEFT COLUMN: Docs Overview + Search */}
                <div className="lg:col-span-6 space-y-6">
                  {/* Dokumin "Docs Overview" Preview Card */}
                  <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-6 space-y-5">
                    {/* Header */}
                    <div className="flex items-center justify-between">
                      <div>
                        <h2 className="text-base font-bold text-slate-900">Docs Overview</h2>
                        <div className="flex items-center gap-1.5 text-xs text-slate-400 mt-0.5">
                          <span>Last updated: 10 minutes ago</span>
                          <button
                            onClick={() => fetchDocuments()}
                            className="hover:text-slate-600 transition"
                            title="Refresh document status"
                          >
                            <RefreshCw className="w-3 h-3 cursor-pointer" />
                          </button>
                        </div>
                      </div>
                      <button
                        onClick={() => setActiveTab('upload')}
                        className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition"
                        title="Edit Documentation"
                      >
                        <Edit3 className="w-4 h-4 cursor-pointer" />
                      </button>
                    </div>

                    {/* Framed Preview Box */}
                    <div className="bg-slate-50 border border-slate-200/80 rounded-xl p-4 relative overflow-hidden min-h-[220px]">
                      {/* Miniature Browser / Reader Header */}
                      <div className="flex items-center gap-2 pb-3 mb-3 border-b border-slate-200 text-[10px] text-slate-500 font-medium">
                        <div className="flex items-center gap-1">
                          <span className="w-2 h-2 rounded-full bg-rose-400"></span>
                          <span className="w-2 h-2 rounded-full bg-amber-400"></span>
                          <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                        </div>
                        <div className="flex items-center gap-3 ml-2">
                          <span className="font-semibold text-slate-800">Dokumin</span>
                          <span className="text-[#3B4BF7]">Getting Started</span>
                          <span className="hidden sm:inline">Architecture</span>
                          <span className="hidden sm:inline">API</span>
                        </div>
                      </div>

                      {/* Miniature Documentation Page Content */}
                      <div className="space-y-2.5">
                        <div className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider">
                          Setting Up Your Docs Site
                        </div>
                        <div className="text-xs font-bold text-slate-800">
                          Enterprise Document Intelligence Engine
                        </div>
                        <div className="space-y-1.5 text-[10px] text-slate-500">
                          <div className="h-2 bg-slate-200/80 rounded w-11/12"></div>
                          <div className="h-2 bg-slate-200/80 rounded w-4/5"></div>
                          <div className="h-2 bg-slate-200/80 rounded w-2/3"></div>
                        </div>

                        {/* Miniature Sections */}
                        <div className="grid grid-cols-2 gap-2 pt-2">
                          <div className="p-2 rounded-lg bg-white border border-slate-200/70 space-y-1">
                            <span className="text-[9px] font-bold text-slate-700 block">Semantic Vector Search</span>
                            <div className="h-1.5 bg-slate-200 rounded w-3/4"></div>
                          </div>
                          <div className="p-2 rounded-lg bg-white border border-slate-200/70 space-y-1">
                            <span className="text-[9px] font-bold text-slate-700 block">Grounded AI Synthesis</span>
                            <div className="h-1.5 bg-slate-200 rounded w-3/4"></div>
                          </div>
                        </div>
                      </div>

                      {/* Live Status Badge */}
                      <div className="absolute bottom-3 right-3 flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-500 text-white text-[10px] font-semibold shadow-sm">
                        <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse"></span>
                        Live
                      </div>
                    </div>

                    {/* Primary Action Button */}
                    <button
                      onClick={() => setActiveTab('documents')}
                      className="w-full py-3 px-4 rounded-xl bg-[#3B4BF7] hover:bg-[#2F3EE0] text-white font-medium text-xs sm:text-sm flex items-center justify-center gap-2 shadow-sm shadow-[#3B4BF7]/20 transition cursor-pointer"
                    >
                      <span>View Documentation</span>
                      <ArrowRight className="w-4 h-4" />
                    </button>

                    {/* Metadata Section */}
                    <div className="space-y-3 pt-3 text-xs border-t border-slate-100 text-slate-600">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 text-slate-500">
                          <Globe className="w-3.5 h-3.5" />
                          <span>Domain</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-slate-800 text-[11px]">pickolab.dokumin.app</span>
                          <button className="text-[10px] text-slate-600 bg-slate-100 hover:bg-slate-200 border border-slate-200/60 px-2 py-0.5 rounded-md font-medium transition cursor-pointer">
                            + Custom new domain
                          </button>
                        </div>
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 text-slate-500">
                          <GitFork className="w-3.5 h-3.5" />
                          <span>Github Repo</span>
                        </div>
                        <span className="font-mono text-slate-800 text-[11px]">charlie-chestnut / dokumin-docs</span>
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 text-slate-500">
                          <GitBranch className="w-3.5 h-3.5" />
                          <span>Branch</span>
                        </div>
                        <span className="font-mono text-slate-800 text-[11px]">main</span>
                      </div>
                    </div>
                  </div>

                  {/* Semantic Passage Retrieval (pgvector) Container Card */}
                  <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-6 space-y-5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-[#3B4BF7]" />
                        <h3 className="text-sm font-semibold text-slate-900">Semantic Passage Retrieval (pgvector)</h3>
                      </div>
                      <span className="text-xs text-slate-400 font-mono">Model: text-embedding-3-small (1536d)</span>
                    </div>

                    {/* Search Form */}
                    <form onSubmit={handleSearch} className="space-y-3.5">
                      <div className="flex flex-col sm:flex-row gap-2.5">
                        <div className="relative flex-1">
                          <Search className="w-4 h-4 absolute left-3 top-3 text-slate-400" />
                          <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder="Ask a question or enter conceptual query across your documents..."
                            className="w-full bg-slate-50 border border-slate-200/80 rounded-xl pl-9 pr-4 py-2.5 text-xs text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-[#3B4BF7] focus:bg-white transition"
                          />
                        </div>

                        <select
                          value={filterDocId}
                          onChange={(e) => setFilterDocId(e.target.value)}
                          className="bg-slate-50 border border-slate-200/80 rounded-xl px-3 py-2 text-xs text-slate-700 focus:outline-none focus:border-[#3B4BF7] cursor-pointer"
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
                          className="px-5 py-2.5 rounded-xl bg-[#3B4BF7] hover:bg-[#2F3EE0] disabled:opacity-50 text-white font-semibold text-xs transition flex items-center justify-center gap-2 cursor-pointer shadow-sm shadow-[#3B4BF7]/20"
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
                      <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-600 flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4 shrink-0" />
                        <span>{searchError}</span>
                      </div>
                    )}

                    {/* Search Results Display */}
                    {searchResults !== null && (
                      <div className="space-y-4 pt-2">
                        <div className="flex items-center justify-between text-xs text-slate-500 border-b border-slate-100 pb-2">
                          <span>
                            Retrieved <span className="text-slate-900 font-semibold">{searchResults.length}</span> passages
                          </span>
                          <span className="font-mono text-[11px] text-slate-400">Query: "{searchQuery}"</span>
                        </div>

                        {searchResults.length === 0 ? (
                          <div className="p-8 text-center border border-dashed border-slate-200 rounded-xl">
                            <p className="text-xs text-slate-400">No matching passages found for this query.</p>
                          </div>
                        ) : (
                          <div className="space-y-3">
                            {searchResults.map((p, idx) => (
                              <div
                                key={p.chunk_id}
                                className="p-4 rounded-xl border border-slate-200/80 bg-white hover:border-[#3B4BF7]/40 transition space-y-2.5 shadow-2xs"
                              >
                                <div className="flex items-center justify-between text-xs">
                                  <div className="flex items-center gap-2">
                                    <span className="w-5 h-5 rounded-full bg-indigo-50 text-[#3B4BF7] font-mono text-[11px] flex items-center justify-center font-bold">
                                      {idx + 1}
                                    </span>
                                    <span className="font-semibold text-slate-800">{p.document_title}</span>
                                    <span className="px-2 py-0.5 rounded-full bg-indigo-50 text-[#3B4BF7] border border-indigo-200/60 font-semibold text-[10px]">
                                      Page {p.page_number}
                                    </span>
                                  </div>
                                  <span className="font-mono text-[11px] text-emerald-600 font-semibold">
                                    {(p.score * 100).toFixed(1)}% match
                                  </span>
                                </div>

                                <div className="flex items-start gap-2.5 text-xs text-slate-600 leading-relaxed bg-slate-50 p-3 rounded-lg border border-slate-100 font-sans">
                                  <Quote className="w-3.5 h-3.5 text-[#3B4BF7] shrink-0 mt-0.5 opacity-60" />
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

                {/* RIGHT COLUMN: Recent Activity Card (Dokumin Reference) */}
                <div className="lg:col-span-6 space-y-6">
                  <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-6 space-y-5">
                    {/* Header */}
                    <div className="flex items-center justify-between">
                      <div>
                        <h2 className="text-base font-bold text-slate-900">Recent Activity</h2>
                        <p className="text-xs text-slate-400 mt-0.5">Recent documentation changes history</p>
                      </div>
                      <button
                        className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition"
                        title="More options"
                      >
                        <MoreVertical className="w-4 h-4 cursor-pointer" />
                      </button>
                    </div>

                    {/* Segmented Tab Controls */}
                    <div className="bg-slate-100/80 p-1 rounded-xl flex items-center gap-1">
                      <button
                        className="flex-1 py-1.5 px-3 rounded-lg bg-white text-slate-800 font-semibold text-xs shadow-xs flex items-center justify-center gap-1.5 transition"
                      >
                        <span>Published Docs</span>
                        <Check className="w-3 h-3 text-[#3B4BF7]" />
                      </button>
                      <button
                        className="flex-1 py-1.5 px-3 rounded-lg text-slate-500 hover:text-slate-700 font-medium text-xs flex items-center justify-center gap-1.5 transition"
                      >
                        <span>Unpublished Drafts</span>
                        <FileText className="w-3 h-3 text-slate-400" />
                      </button>
                    </div>

                    {/* Group Filter Header */}
                    <div className="flex items-center gap-1 text-xs font-bold text-slate-700 cursor-pointer pt-1">
                      <span>This Week</span>
                      <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
                    </div>

                    {/* Activity List Items */}
                    <div className="space-y-2.5">
                      {[
                        { time: 'Today, 11:42 AM', title: 'Fix Documentation Bug' },
                        { time: 'Today, 9:41 AM', title: 'Content Addition' },
                        { time: 'Today, 7:38 AM', title: 'Minor Fix' },
                        { time: 'Yesterday, 3:15 PM', title: 'API Reference Update' },
                        { time: 'Yesterday, 12:07 PM', title: 'Manual Update' },
                        { time: 'Yesterday, 9:32 AM', title: 'New Section Published' },
                      ].map((item, idx) => (
                        <div
                          key={idx}
                          className="p-3.5 rounded-xl border border-slate-200/80 bg-white hover:border-slate-300 transition flex items-center justify-between group shadow-2xs"
                        >
                          <div className="flex items-center gap-3">
                            <div className="w-7 h-7 rounded-full bg-emerald-50 text-emerald-600 border border-emerald-100 flex items-center justify-center shrink-0">
                              <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                            </div>
                            <div>
                              <div className="text-[10px] text-slate-400 font-medium">{item.time}</div>
                              <div className="text-xs font-semibold text-slate-800 group-hover:text-[#3B4BF7] transition">
                                {item.title}
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center gap-1 text-xs text-slate-400 group-hover:text-slate-600 cursor-pointer transition">
                            <span>See changes</span>
                            <ChevronDown className="w-3 h-3" />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: OBSERVABILITY & MONITORING TELEMETRY */}
          {activeTab === 'monitoring' && (
            <div className="flex-1 overflow-y-auto p-6 sm:p-8 space-y-6">
              <div className="rounded-2xl border border-slate-200/80 bg-white p-6 space-y-4 shadow-sm">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-emerald-50 border border-emerald-100 text-emerald-600 flex items-center justify-center">
                      <Activity className="w-5 h-5" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="text-sm font-semibold text-slate-900">Observability & Request Telemetry</h3>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
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
                      <span className="text-[11px] font-mono text-slate-500 bg-slate-50 px-2.5 py-1 rounded-xl border border-slate-200/80">
                        Pricing: {monitoringData.pricing_version}
                      </span>
                    )}
                    <button
                      onClick={fetchMonitoring}
                      disabled={loadingMonitoring}
                      className="px-3 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-medium border border-slate-200/80 transition flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                    >
                      <RefreshCw className={`w-3.5 h-3.5 ${loadingMonitoring ? 'animate-spin text-emerald-600' : ''}`} />
                      Refresh
                    </button>
                  </div>
                </div>

                {monitoringError && (
                  <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-600 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 shrink-0" />
                    <span>{monitoringError}</span>
                  </div>
                )}
              </div>

              {/* Stat Cards Grid */}
              {monitoringData && (
                <>
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                    <div className="p-4 rounded-2xl border border-slate-200/80 bg-white space-y-1 shadow-sm">
                      <span className="text-[11px] text-slate-400 font-medium block">Total Requests</span>
                      <div className="text-2xl font-bold text-slate-900 font-mono">{monitoringData.total_requests}</div>
                      <div className="text-[10px] text-slate-500 flex items-center gap-1.5">
                        <span className="text-emerald-600 font-semibold">{monitoringData.successful_requests} ok</span>
                        <span>&bull;</span>
                        <span className="text-rose-500 font-semibold">{monitoringData.failed_requests} err</span>
                      </div>
                    </div>

                    <div className="p-4 rounded-2xl border border-slate-200/80 bg-white space-y-1 shadow-sm">
                      <span className="text-[11px] text-slate-400 font-medium block">Avg Latency</span>
                      <div className="text-2xl font-bold text-[#3B4BF7] font-mono">
                        {monitoringData.avg_total_latency_ms}
                        <span className="text-xs text-slate-400 font-normal ml-0.5">ms</span>
                      </div>
                      <div className="text-[10px] text-slate-500">End-to-end average</div>
                    </div>

                    <div className="p-4 rounded-2xl border border-slate-200/80 bg-white space-y-1 shadow-sm">
                      <span className="text-[11px] text-slate-400 font-medium block">P95 Latency</span>
                      <div className="text-2xl font-bold text-indigo-600 font-mono">
                        {monitoringData.p95_total_latency_ms}
                        <span className="text-xs text-slate-400 font-normal ml-0.5">ms</span>
                      </div>
                      <div className="text-[10px] text-slate-500">95th percentile SLA</div>
                    </div>

                    <div className="p-4 rounded-2xl border border-slate-200/80 bg-white space-y-1 shadow-sm">
                      <span className="text-[11px] text-slate-400 font-medium block">Error Rate</span>
                      <div
                        className={`text-2xl font-bold font-mono ${
                          monitoringData.error_rate > 0 ? 'text-rose-500' : 'text-emerald-600'
                        }`}
                      >
                        {(monitoringData.error_rate * 100).toFixed(1)}%
                      </div>
                      <div className="text-[10px] text-slate-500">Failed / Total</div>
                    </div>

                    <div className="p-4 rounded-2xl border border-slate-200/80 bg-white space-y-1 shadow-sm">
                      <span className="text-[11px] text-slate-400 font-medium block">Token Usage</span>
                      <div className="text-2xl font-bold text-amber-600 font-mono">
                        {monitoringData.token_usage?.total_tokens !== null &&
                        monitoringData.token_usage?.total_tokens !== undefined
                          ? monitoringData.token_usage.total_tokens.toLocaleString()
                          : <span className="text-slate-400 text-lg italic">Unknown</span>}
                      </div>
                      <div className="text-[10px] text-slate-500">Prompt & Completion</div>
                    </div>

                    <div className="p-4 rounded-2xl border border-slate-200/80 bg-white space-y-1 shadow-sm">
                      <span className="text-[11px] text-slate-400 font-medium block">Estimated Cost</span>
                      <div className="text-2xl font-bold text-emerald-600 font-mono">
                        {monitoringData.total_estimated_cost_usd !== null &&
                        monitoringData.total_estimated_cost_usd !== undefined
                          ? `$${monitoringData.total_estimated_cost_usd.toFixed(4)}`
                          : <span className="text-slate-400 text-lg italic">Unknown</span>}
                      </div>
                      <div className="text-[10px] text-slate-500">Catalog-based</div>
                    </div>
                  </div>

                  {/* Recent Requests Table */}
                  <div className="rounded-2xl border border-slate-200/80 bg-white p-6 space-y-4 shadow-sm">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
                        Recent Request Lifecycle Log ({monitoringData.recent_requests.length})
                      </h4>
                      <span className="text-[11px] text-slate-400 font-mono">
                        Zero sensitive document text stored
                      </span>
                    </div>

                    {monitoringData.recent_requests.length === 0 ? (
                      <div className="p-8 text-center border border-dashed border-slate-200 rounded-2xl text-xs text-slate-400">
                        No requests recorded yet. Run a RAG query to see live telemetry.
                      </div>
                    ) : (
                      <div className="overflow-x-auto">
                        <table className="w-full text-left text-xs">
                          <thead>
                            <tr className="border-b border-slate-100 text-slate-500 font-medium">
                              <th className="pb-2.5 font-semibold">Request ID</th>
                              <th className="pb-2.5 font-semibold">Status</th>
                              <th className="pb-2.5 font-semibold">Retrieval</th>
                              <th className="pb-2.5 font-semibold">Generation</th>
                              <th className="pb-2.5 font-semibold">Total Latency</th>
                              <th className="pb-2.5 font-semibold">Tokens</th>
                              <th className="pb-2.5 font-semibold">Estimated Cost</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
                            {monitoringData.recent_requests.map((req) => (
                              <tr key={req.id} className="hover:bg-slate-50 transition">
                                <td className="py-2.5 font-semibold text-slate-800">{req.request_id}</td>
                                <td className="py-2.5">
                                  <span
                                    className={`px-1.5 py-0.5 rounded-full text-[10px] font-semibold uppercase ${
                                      req.status === 'success'
                                        ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                                        : 'bg-rose-50 text-rose-600 border border-rose-200'
                                    }`}
                                  >
                                    {req.status}
                                  </span>
                                </td>
                                <td className="py-2.5 text-slate-500">{req.retrieval_latency_ms}ms</td>
                                <td className="py-2.5 text-slate-500">{req.generation_latency_ms}ms</td>
                                <td className="py-2.5 text-slate-800 font-semibold">{req.total_latency_ms}ms</td>
                                <td className="py-2.5">
                                  {req.total_tokens !== null && req.total_tokens !== undefined ? (
                                    <span className="text-amber-600">{req.total_tokens}</span>
                                  ) : (
                                    <span className="text-slate-400 italic">Unknown</span>
                                  )}
                                </td>
                                <td className="py-2.5">
                                  {req.estimated_cost_usd !== null && req.estimated_cost_usd !== undefined ? (
                                    <span className="text-emerald-600">${req.estimated_cost_usd.toFixed(4)}</span>
                                  ) : (
                                    <span className="text-slate-400 italic">Unknown</span>
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
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-sm">
          <div className="bg-white border border-slate-200 rounded-2xl max-w-sm w-full p-6 shadow-2xl relative space-y-4">
            <button
              onClick={() => {
                setShowAuthModal(false);
                setAuthError(null);
              }}
              className="absolute top-4 right-4 text-slate-400 hover:text-slate-600"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="text-center space-y-1">
              <div className="w-10 h-10 rounded-xl bg-indigo-50 text-[#3B4BF7] flex items-center justify-center mx-auto mb-2">
                <Lock className="w-5 h-5" />
              </div>
              <h3 className="text-lg font-bold text-slate-900">
                {authMode === 'login' ? 'Welcome Back' : 'Create Account'}
              </h3>
              <p className="text-xs text-slate-500">
                {authMode === 'login'
                  ? 'Sign in to access your authorized enterprise documents'
                  : 'Register a new tenant user profile'}
              </p>
            </div>

            {/* Mode Switcher */}
            <div className="flex rounded-xl bg-slate-100 p-1 border border-slate-200/80 text-xs">
              <button
                type="button"
                onClick={() => {
                  setAuthMode('login');
                  setAuthError(null);
                }}
                className={`flex-1 py-1.5 rounded-lg font-semibold transition cursor-pointer ${
                  authMode === 'login' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-500 hover:text-slate-700'
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
                className={`flex-1 py-1.5 rounded-lg font-semibold transition cursor-pointer ${
                  authMode === 'register' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-500 hover:text-slate-700'
                }`}
              >
                Register
              </button>
            </div>

            {authError && (
              <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-600 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>{authError}</span>
              </div>
            )}

            <form onSubmit={handleAuthSubmit} className="space-y-3.5">
              {authMode === 'register' && (
                <div>
                  <label className="block text-xs font-medium text-slate-700 mb-1">Email Address</label>
                  <div className="relative">
                    <Mail className="w-4 h-4 absolute left-3 top-2.5 text-slate-400" />
                    <input
                      type="email"
                      required
                      value={authEmail}
                      onChange={(e) => setAuthEmail(e.target.value)}
                      placeholder="analyst@enterprise.com"
                      className="w-full bg-slate-50 border border-slate-200/80 rounded-xl pl-9 pr-3 py-2 text-xs text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-[#3B4BF7] focus:bg-white"
                    />
                  </div>
                </div>
              )}

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">
                  {authMode === 'login' ? 'Username or Email' : 'Username'}
                </label>
                <div className="relative">
                  <UserIcon className="w-4 h-4 absolute left-3 top-2.5 text-slate-400" />
                  <input
                    type="text"
                    required
                    value={authUsername}
                    onChange={(e) => setAuthUsername(e.target.value)}
                    placeholder={authMode === 'login' ? 'analyst or analyst@enterprise.com' : 'doc_analyst'}
                    className="w-full bg-slate-50 border border-slate-200/80 rounded-xl pl-9 pr-3 py-2 text-xs text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-[#3B4BF7] focus:bg-white"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">Password</label>
                <div className="relative">
                  <Lock className="w-4 h-4 absolute left-3 top-2.5 text-slate-400" />
                  <input
                    type="password"
                    required
                    minLength={8}
                    value={authPassword}
                    onChange={(e) => setAuthPassword(e.target.value)}
                    placeholder="Min 8 characters"
                    className="w-full bg-slate-50 border border-slate-200/80 rounded-xl pl-9 pr-3 py-2 text-xs text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-[#3B4BF7] focus:bg-white"
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={authLoading}
                className="w-full py-2.5 rounded-xl bg-[#3B4BF7] hover:bg-[#2F3EE0] disabled:opacity-50 text-white font-semibold text-xs transition flex items-center justify-center gap-2 cursor-pointer shadow-sm shadow-[#3B4BF7]/25"
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
      <footer className="h-10 border-t border-slate-200/80 bg-white px-6 flex items-center justify-between text-[11px] text-slate-400 shrink-0">
        <div>DocuMind AI &bull; Enterprise Document Intelligence & RAG Platform</div>
        <div className="flex items-center gap-4">
          <span>React 19 &bull; Vite &bull; FastAPI &bull; PostgreSQL (pgvector 1536d)</span>
        </div>
      </footer>
    </div>
  );
}
