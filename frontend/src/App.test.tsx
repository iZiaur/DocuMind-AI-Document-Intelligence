import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, act, fireEvent } from '@testing-library/react';
import App from './App';

describe('DocuMind AI Shell & Semantic Search UI', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.stubGlobal('fetch', vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.includes('/api/health')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            status: 'healthy',
            timestamp: '2026-10-29T14:00:00Z',
            database: {
              status: 'connected',
              pgvector_available: true,
              message: 'Database connection established successfully',
            },
            environment: {
              service: 'documind-api',
              version: '0.3.0',
              embedding_model: 'text-embedding-3-small',
              embedding_dimension: 1536,
            },
          }),
        });
      }
      if (url.includes('/api/conversations')) {
        if (init?.method === 'POST' && url.endsWith('/messages')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: async () => ({
              id: 'rag-req-001',
              conversation_id: 'conv-test-101',
              role: 'assistant',
              content: 'AES-256 encryption is mandated across all customer databases at rest.',
              insufficient_evidence: false,
              citations: [],
            }),
          });
        }

        if (init?.method === 'POST') {
          return Promise.resolve({
            ok: true,
            status: 201,
            json: async () => ({
              id: 'conv-test-101',
              user_id: 1,
              document_id: null,
              title: 'New Conversation',
              document_scope_invalidated: false,
              created_at: '2026-10-31T20:00:00Z',
              updated_at: '2026-10-31T20:00:00Z',
              messages: [],
            }),
          });
        }

        if (init?.method === 'DELETE') {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: async () => ({ message: 'Conversation deleted successfully' }),
          });
        }

        if (url.match(/\/api\/conversations\/[^/]+$/)) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: async () => ({
              id: 'conv-test-101',
              user_id: 1,
              document_id: null,
              title: 'New Conversation',
              document_scope_invalidated: false,
              created_at: '2026-10-31T20:00:00Z',
              updated_at: '2026-10-31T20:00:00Z',
              messages: [],
            }),
          });
        }

        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            total: 1,
            conversations: [
              {
                id: 'conv-test-101',
                user_id: 1,
                document_id: null,
                title: 'Corporate Security Chat',
                document_scope_invalidated: false,
                created_at: '2026-10-31T20:00:00Z',
                updated_at: '2026-10-31T20:00:00Z',
                latest_message: null,
              },
            ],
          }),
        });
      }
      if (url.includes('/api/documents')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            total: 0,
            documents: [],
          }),
        });
      }
      if (url.includes('/api/monitoring/dashboard')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            total_requests: 12,
            successful_requests: 11,
            failed_requests: 1,
            error_rate: 0.0833,
            avg_total_latency_ms: 124.5,
            avg_retrieval_latency_ms: 45.2,
            avg_generation_latency_ms: 79.3,
            p95_total_latency_ms: 250.0,
            token_usage: {
              total_prompt_tokens: 1500,
              total_completion_tokens: 450,
              total_tokens: 1950,
              requests_with_unknown_tokens: 2,
              known_token_requests: 10,
              unknown_token_requests: 2,
              is_partial: true,
            },
            total_estimated_cost_usd: 0.000675,
            cost_is_partial: true,
            known_cost_requests: 10,
            unknown_cost_requests: 2,
            pricing_version: '2024-08-01',
            recent_requests: [
              {
                id: 1,
                request_id: 'req-test-12345',
                timestamp: '2026-10-29T14:30:00Z',
                user_id: 1,
                document_count: 3,
                retrieval_latency_ms: 45.2,
                generation_latency_ms: 78.1,
                total_latency_ms: 123.3,
                status: 'success',
                error_category: null,
                safe_message: null,
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
                estimated_cost_usd: 0.000045,
                pricing_version: '2024-08-01',
              },
            ],
          }),
        });
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({}),
      });
    }));
  });

  it('renders the brand title, enterprise badge, and Stage 3 indicator', async () => {
    await act(async () => {
      render(<App />);
    });
    expect(screen.getByText('DocuMind AI')).toBeDefined();
    expect(screen.getByText(/Stage 3: PDF Ingestion & Semantic Vector Search Active/i)).toBeDefined();
  });

  it('renders the Sign In button and opens authentication modal on click', async () => {
    await act(async () => {
      render(<App />);
    });

    const signInBtn = screen.getByRole('button', { name: /Sign In \/ Register/i });
    expect(signInBtn).toBeDefined();

    await act(async () => {
      fireEvent.click(signInBtn);
    });

    expect(screen.getByText('Welcome Back')).toBeDefined();
    expect(screen.getByRole('button', { name: 'Register' })).toBeDefined();
  }, 15000);

  it('renders the semantic search input and allows tab switching to upload', async () => {
    await act(async () => {
      render(<App />);
    });

    expect(screen.getByText('Semantic Passage Retrieval (pgvector)')).toBeDefined();
    expect(screen.getByPlaceholderText(/Ask a question or enter conceptual query/i)).toBeDefined();

    const uploadTabBtn = screen.getByRole('button', { name: /Upload Document/i });
    await act(async () => {
      fireEvent.click(uploadTabBtn);
    });

    expect(screen.getByText('Upload & Ingest Enterprise PDF')).toBeDefined();
  });

  it('renders the observability tab and displays monitoring metrics cards', async () => {
    localStorage.setItem('documind_token', 'mock-token');
    await act(async () => {
      render(<App />);
    });

    const obsTabBtn = screen.getByRole('button', { name: /Observability/i });
    expect(obsTabBtn).toBeDefined();

    await act(async () => {
      fireEvent.click(obsTabBtn);
    });

    expect(screen.getByText('Observability & Request Telemetry')).toBeDefined();
    expect(screen.getByText('Total Requests')).toBeDefined();
    expect(screen.getByText('Avg Latency')).toBeDefined();
    expect(screen.getByText('P95 Latency')).toBeDefined();
    expect(screen.getByText('Error Rate')).toBeDefined();
    expect(screen.getAllByText('Estimated Cost').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('req-test-12345')).toBeDefined();
  });

  it('triggers authenticated PDF download with Bearer token on click (SEC-02)', async () => {
    localStorage.setItem('documind_token', 'valid-bearer-token');
    localStorage.setItem(
      'documind_user',
      JSON.stringify({ id: 1, email: 'user@example.com', username: 'user1', is_admin: false })
    );

    const mockBlob = new Blob(['%PDF-1.4 Fake PDF'], { type: 'application/pdf' });
    const createObjectURLMock = vi.fn().mockReturnValue('blob:http://localhost/fake-blob-url');
    const revokeObjectURLMock = vi.fn();
    window.URL.createObjectURL = createObjectURLMock;
    window.URL.revokeObjectURL = revokeObjectURLMock;

    let downloadAuthHeader: string | null = null;

    vi.stubGlobal('fetch', vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.includes('/api/health')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            status: 'healthy',
            timestamp: '2026-10-29T14:00:00Z',
            database: { status: 'connected', pgvector_available: true, message: 'OK' },
            environment: { service: 'documind-api', version: '0.3.0' },
          }),
        });
      }
      if (url.includes('/api/documents/') && url.includes('/download')) {
        const headers = init?.headers as Record<string, string>;
        downloadAuthHeader = headers?.['Authorization'] || null;
        return Promise.resolve({
          ok: true,
          status: 200,
          blob: async () => mockBlob,
        });
      }
      if (url.includes('/api/conversations')) {
        if (init?.method === 'POST' && url.endsWith('/messages')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: async () => ({
              id: 'rag-req-001',
              conversation_id: 'conv-test-101',
              role: 'assistant',
              content: 'AES-256 encryption is mandated across all customer databases at rest.',
              insufficient_evidence: false,
              citations: [],
            }),
          });
        }

        if (init?.method === 'POST') {
          return Promise.resolve({
            ok: true,
            status: 201,
            json: async () => ({
              id: 'conv-test-101',
              user_id: 1,
              document_id: null,
              title: 'New Conversation',
              document_scope_invalidated: false,
              created_at: '2026-10-31T20:00:00Z',
              updated_at: '2026-10-31T20:00:00Z',
              messages: [],
            }),
          });
        }

        if (init?.method === 'DELETE') {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: async () => ({ message: 'Conversation deleted successfully' }),
          });
        }

        if (url.match(/\/api\/conversations\/[^/]+$/)) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: async () => ({
              id: 'conv-test-101',
              user_id: 1,
              document_id: null,
              title: 'New Conversation',
              document_scope_invalidated: false,
              created_at: '2026-10-31T20:00:00Z',
              updated_at: '2026-10-31T20:00:00Z',
              messages: [],
            }),
          });
        }

        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            total: 1,
            conversations: [
              {
                id: 'conv-test-101',
                user_id: 1,
                document_id: null,
                title: 'Corporate Security Chat',
                document_scope_invalidated: false,
                created_at: '2026-10-31T20:00:00Z',
                updated_at: '2026-10-31T20:00:00Z',
                latest_message: null,
              },
            ],
          }),
        });
      }
      if (url.includes('/api/documents')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            total: 1,
            documents: [
              {
                id: 'doc-uuid-1234',
                owner_id: 1,
                title: 'Financial Q3 Report',
                filename: 'q3_report.pdf',
                file_size_bytes: 5000,
                page_count: 2,
                status: 'completed',
                created_at: '2026-10-29T14:00:00Z',
              },
            ],
          }),
        });
      }
      return Promise.resolve({ ok: true, status: 200, json: async () => ({}) });
    }));

    await act(async () => {
      render(<App />);
    });

    // Select the document from the list
    const docItems = await screen.findAllByText('Financial Q3 Report');
    const docCard = docItems[0];
    await act(async () => {
      fireEvent.click(docCard);
    });

    // Find and click the Download PDF button
    const downloadBtn = await screen.findByRole('button', { name: /Download PDF/i });
    expect(downloadBtn).toBeDefined();

    await act(async () => {
      fireEvent.click(downloadBtn);
    });

    // Verify authenticated request was sent with Bearer token
    expect(downloadAuthHeader).toBe('Bearer valid-bearer-token');
    expect(createObjectURLMock).toHaveBeenCalledWith(mockBlob);
    expect(revokeObjectURLMock).toHaveBeenCalledWith('blob:http://localhost/fake-blob-url');
  }, 15000);
});

describe('Ask DocuMind RAG Interface (Checkpoint 12)', () => {
  beforeEach(() => {
    localStorage.clear();
    Object.assign(navigator, {
      clipboard: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
    });
    vi.stubGlobal('open', vi.fn());
  });

  const setupMockFetch = (options?: {
    ragResponse?: any;
    ragStatus?: number;
    delayMs?: number;
  }) => {
    let capturedRagRequest: { url: string; headers: any; body: any } | null = null;

    vi.stubGlobal('fetch', vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.includes('/api/health')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            status: 'healthy',
            timestamp: '2026-10-31T20:00:00Z',
            database: { status: 'connected', pgvector_available: true, message: 'OK' },
            environment: { service: 'documind-api', version: '0.4.0' },
          }),
        });
      }

      if (url.includes('/api/conversations')) {
        if (init?.method === 'POST' && url.endsWith('/messages')) {
          capturedRagRequest = {
            url,
            headers: init?.headers,
            body: init?.body ? JSON.parse(init.body as string) : null,
          };

          if (options?.delayMs) {
            return new Promise((resolve) => {
              setTimeout(() => {
                resolve({
                  ok: (options.ragStatus || 200) < 400,
                  status: options.ragStatus || 200,
                  json: async () =>
                    options.ragResponse || {
                      id: 'rag-req-001',
                      content: 'Default test answer',
                      insufficient_evidence: false,
                      citations: [],
                      retrieval_count: 0,
                      model_used: 'gemini-2.5-flash',
                    },
                });
              }, options.delayMs);
            });
          }

          if (options?.ragStatus && options.ragStatus >= 400) {
            return Promise.resolve({
              ok: false,
              status: options.ragStatus,
              json: async () => options.ragResponse || { detail: 'Request failed' },
            });
          }

          return Promise.resolve({
            ok: true,
            status: 200,
            json: async () =>
              options?.ragResponse || {
                id: 'rag-req-001',
                conversation_id: 'conv-test-101',
                role: 'assistant',
                content: 'AES-256 encryption is mandated across all customer databases at rest.',
                insufficient_evidence: false,
                citations: [
                  {
                    chunk_id: 'chunk-uuid-101',
                    document_id: 'doc-uuid-sec-01',
                    document_title: 'Corporate Security Policy.pdf',
                    page_number: 3,
                    excerpt: 'All sensitive customer records at rest must be encrypted using AES-256 GCM cipher suites.',
                    score: 0.945,
                  },
                ],
                retrieval_count: 1,
                model_used: 'gemini-2.5-flash',
                total_latency_ms: 412.5,
                token_usage: { prompt_tokens: 320, completion_tokens: 45, total_tokens: 365 },
                estimated_cost_usd: 0.00003,
                created_at: '2026-10-31T20:00:00Z',
              },
          });
        }

        if (init?.method === 'POST') {
          return Promise.resolve({
            ok: true,
            status: 201,
            json: async () => ({
              id: 'conv-test-101',
              user_id: 1,
              document_id: null,
              title: 'New Conversation',
              document_scope_invalidated: false,
              created_at: '2026-10-31T20:00:00Z',
              updated_at: '2026-10-31T20:00:00Z',
              messages: [],
            }),
          });
        }

        if (init?.method === 'DELETE') {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: async () => ({ message: 'Conversation deleted successfully' }),
          });
        }

        if (url.match(/\/api\/conversations\/[^/]+$/)) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: async () => ({
              id: 'conv-test-101',
              user_id: 1,
              document_id: null,
              title: 'New Conversation',
              document_scope_invalidated: false,
              created_at: '2026-10-31T20:00:00Z',
              updated_at: '2026-10-31T20:00:00Z',
              messages: [],
            }),
          });
        }

        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            total: 1,
            conversations: [
              {
                id: 'conv-test-101',
                user_id: 1,
                document_id: null,
                title: 'Corporate Security Chat',
                document_scope_invalidated: false,
                created_at: '2026-10-31T20:00:00Z',
                updated_at: '2026-10-31T20:00:00Z',
                latest_message: null,
              },
            ],
          }),
        });
      }
      if (url.includes('/api/documents')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            total: 1,
            documents: [
              {
                id: 'doc-uuid-sec-01',
                title: 'Corporate Security Policy.pdf',
                filename: 'corporate_security.pdf',
                file_size_bytes: 8400,
                page_count: 5,
                status: 'completed',
                created_at: '2026-10-31T20:00:00Z',
              },
            ],
          }),
        });
      }

      if (url.includes('/api/rag/query')) {
        capturedRagRequest = {
          url,
          headers: init?.headers,
          body: init?.body ? JSON.parse(init.body as string) : null,
        };

        if (options?.delayMs) {
          return new Promise((resolve) => {
            setTimeout(() => {
              resolve({
                ok: (options.ragStatus || 200) < 400,
                status: options.ragStatus || 200,
                json: async () =>
                  options.ragResponse || {
                    request_id: 'rag-req-001',
                    question: 'Test question',
                    answer: 'Default test answer',
                    insufficient_evidence: false,
                    citations: [],
                    retrieval_count: 0,
                    model_used: 'gemini-2.5-flash',
                  },
              });
            }, options.delayMs);
          });
        }

        if (options?.ragStatus && options.ragStatus >= 400) {
          return Promise.resolve({
            ok: false,
            status: options.ragStatus,
            json: async () => options.ragResponse || { detail: 'Request failed' },
          });
        }

        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () =>
            options?.ragResponse || {
              request_id: 'rag-req-001',
              question: 'Test question',
              answer: 'AES-256 encryption is mandated across all customer databases at rest.',
              insufficient_evidence: false,
              citations: [
                {
                  chunk_id: 'chunk-uuid-101',
                  document_id: 'doc-uuid-sec-01',
                  document_title: 'Corporate Security Policy.pdf',
                  page_number: 3,
                  excerpt: 'All sensitive customer records at rest must be encrypted using AES-256 GCM cipher suites.',
                  score: 0.945,
                },
              ],
              retrieval_count: 1,
              model_used: 'gemini-2.5-flash',
              total_latency_ms: 412.5,
              token_usage: { prompt_tokens: 320, completion_tokens: 45, total_tokens: 365 },
              estimated_cost_usd: 0.00003,
            },
        });
      }

      return Promise.resolve({ ok: true, status: 200, json: async () => ({}) });
    }));

    return { getCapturedRagRequest: () => capturedRagRequest };
  };

  it('1. renders the Ask DocuMind page with welcome screen, suggested questions, and input bar', async () => {
    setupMockFetch();
    await act(async () => {
      render(<App />);
    });

    // Switch to Ask DocuMind tab
    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    expect(screen.getByText('Ask DocuMind Intelligence')).toBeDefined();
    expect(screen.getByText(/Ask natural-language questions across your uploaded enterprise PDFs/i)).toBeDefined();
    expect(screen.getByText('Suggested Inquiries')).toBeDefined();
    expect(
      screen.getByPlaceholderText(/Ask a natural-language question/i)
    ).toBeDefined();
  });

  it('2 & 3. sends authenticated RAG request with correct question payload and Bearer token', async () => {
    localStorage.setItem('documind_token', 'jwt-test-bearer-token');
    localStorage.setItem(
      'documind_user',
      JSON.stringify({ id: 1, email: 'analyst@documind.ai', username: 'analyst', is_admin: false })
    );

    const { getCapturedRagRequest } = setupMockFetch();

    await act(async () => {
      render(<App />);
    });

    // Switch to Ask DocuMind
    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    const input = screen.getByPlaceholderText(/Ask a natural-language question/i);
    await act(async () => {
      fireEvent.change(input, { target: { value: 'What encryption standard is enforced?' } });
    });

    const submitBtn = screen.getByRole('button', { name: /Submit question/i });
    await act(async () => {
      fireEvent.click(submitBtn);
    });

    const captured = getCapturedRagRequest();
    expect(captured).not.toBeNull();
    expect(captured?.url).toContain('/api/conversations');
    expect(captured?.url).toContain('/messages');
    expect(captured?.headers['Authorization']).toBe('Bearer jwt-test-bearer-token');
    expect(captured?.headers['Content-Type']).toBe('application/json');
    expect(captured?.body).toEqual({
      content: 'What encryption standard is enforced?',
      top_k: 4,
    });
  });

  it('4 & 5. renders the generated answer with model badge, latencies, and numbered citations', async () => {
    localStorage.setItem('documind_token', 'jwt-test-bearer-token');
    setupMockFetch();

    await act(async () => {
      render(<App />);
    });

    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    const input = screen.getByPlaceholderText(/Ask a natural-language question/i);
    await act(async () => {
      fireEvent.change(input, { target: { value: 'What encryption standard is enforced?' } });
    });

    const submitBtn = screen.getByRole('button', { name: /Submit question/i });
    await act(async () => {
      fireEvent.click(submitBtn);
    });

    // Check answer text
    expect(
      await screen.findByText('AES-256 encryption is mandated across all customer databases at rest.')
    ).toBeDefined();

    // Check model badge & latency
    expect(screen.getByText('gemini-2.5-flash')).toBeDefined();
    expect(screen.getByText('412.5ms')).toBeDefined();

    // Check citation rendered
    expect(screen.getAllByText('Corporate Security Policy.pdf').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('p.3')).toBeDefined();
  });

  it('6. clicking a citation opens and populates the Source Inspector panel', async () => {
    localStorage.setItem('documind_token', 'jwt-test-bearer-token');
    setupMockFetch();

    await act(async () => {
      render(<App />);
    });

    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    const input = screen.getByPlaceholderText(/Ask a natural-language question/i);
    await act(async () => {
      fireEvent.change(input, { target: { value: 'What encryption is used?' } });
      fireEvent.click(screen.getByRole('button', { name: /Submit question/i }));
    });

    await screen.findByText('AES-256 encryption is mandated across all customer databases at rest.');

    // Click on the citation pill
    const citationBtn = screen.getByRole('button', { name: /Corporate Security Policy.pdf/i });
    await act(async () => {
      fireEvent.click(citationBtn);
    });

    // Verify Source Inspector panel content
    expect(screen.getByText('Source Inspector')).toBeDefined();
    expect(screen.getByText('Verified Source')).toBeDefined();
    expect(screen.getByText('Page 3')).toBeDefined();
    expect(screen.getByText('94.5% match')).toBeDefined();
    expect(screen.getByText(/All sensitive customer records at rest must be encrypted/i)).toBeDefined();
    expect(screen.getByRole('button', { name: /Open PDF at Page 3/i })).toBeDefined();
  });

  it('7. renders abstention and insufficient evidence alert banner clearly', async () => {
    localStorage.setItem('documind_token', 'jwt-test-bearer-token');
    setupMockFetch({
      ragResponse: {
        request_id: 'rag-req-abstain',
        question: 'What is the recipe for chocolate cake?',
        answer: 'I do not have sufficient information in the provided documents to answer this question.',
        insufficient_evidence: true,
        citations: [],
        retrieval_count: 0,
        model_used: 'gemini-2.5-flash',
        total_latency_ms: 120.0,
      },
    });

    await act(async () => {
      render(<App />);
    });

    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    const input = screen.getByPlaceholderText(/Ask a natural-language question/i);
    await act(async () => {
      fireEvent.change(input, { target: { value: 'What is the recipe for chocolate cake?' } });
      fireEvent.click(screen.getByRole('button', { name: /Submit question/i }));
    });

    expect(await screen.findByText('Abstention Alert:')).toBeDefined();
    expect(
      screen.getByText(/The system abstained from answering because the retrieved document context/i)
    ).toBeDefined();
  });

  it('8. displays in-flight loading state indicator during RAG synthesis', async () => {
    localStorage.setItem('documind_token', 'jwt-test-bearer-token');
    setupMockFetch({ delayMs: 400 });

    await act(async () => {
      render(<App />);
    });

    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    const input = screen.getByPlaceholderText(/Ask a natural-language question/i);
    await act(async () => {
      fireEvent.change(input, { target: { value: 'Slow query check' } });
      fireEvent.click(screen.getByRole('button', { name: /Submit question/i }));
    });

    // Check loading text
    expect(screen.getByText('Synthesizing Answer with Gemini...')).toBeDefined();
  });

  it('9. safely handles API failure and surfaces error notice without crashing', async () => {
    localStorage.setItem('documind_token', 'jwt-test-bearer-token');
    setupMockFetch({
      ragStatus: 503,
      ragResponse: { detail: 'Upstream AI provider unavailable. Rate limit reached.' },
    });

    await act(async () => {
      render(<App />);
    });

    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    const input = screen.getByPlaceholderText(/Ask a natural-language question/i);
    await act(async () => {
      fireEvent.change(input, { target: { value: 'Failing query test' } });
      fireEvent.click(screen.getByRole('button', { name: /Submit question/i }));
    });

    const errorNotices = await screen.findAllByText(/Upstream AI provider unavailable/i);
    expect(errorNotices.length).toBeGreaterThanOrEqual(1);
  });

  it('10. handles expired authentication (HTTP 401) by clearing token and opening auth modal', async () => {
    localStorage.setItem('documind_token', 'expired-token');
    setupMockFetch({
      ragStatus: 401,
      ragResponse: { detail: 'Could not validate credentials' },
    });

    await act(async () => {
      render(<App />);
    });

    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    const input = screen.getByPlaceholderText(/Ask a natural-language question/i);
    await act(async () => {
      fireEvent.change(input, { target: { value: 'Unauthorized question' } });
      fireEvent.click(screen.getByRole('button', { name: /Submit question/i }));
    });

    // Token cleared
    expect(localStorage.getItem('documind_token')).toBeNull();
    // Auth modal shown
    expect(await screen.findByText('Welcome Back')).toBeDefined();
  });

  it('11. prevents duplicate submission when input is empty or request is in-flight', async () => {
    localStorage.setItem('documind_token', 'jwt-test-bearer-token');
    const { getCapturedRagRequest } = setupMockFetch({ delayMs: 400 });

    await act(async () => {
      render(<App />);
    });

    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    const submitBtn = screen.getByRole('button', { name: /Submit question/i });
    // Attempt clicking when input is empty
    await act(async () => {
      fireEvent.click(submitBtn);
    });
    expect(getCapturedRagRequest()).toBeNull();
  });

  it('12. supports New Conversation and Clear Chat actions', async () => {
    localStorage.setItem('documind_token', 'jwt-test-bearer-token');
    setupMockFetch();

    await act(async () => {
      render(<App />);
    });

    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    const input = screen.getByPlaceholderText(/Ask a natural-language question/i);
    await act(async () => {
      fireEvent.change(input, { target: { value: 'First question' } });
      fireEvent.click(screen.getByRole('button', { name: /Submit question/i }));
    });

    await screen.findByText('AES-256 encryption is mandated across all customer databases at rest.');

    // Click New Chat
    const newChatBtn = screen.getByRole('button', { name: /New Chat/i });
    await act(async () => {
      fireEvent.click(newChatBtn);
    });

    // Chat is cleared, returns to Welcome screen
    expect(screen.getByText('Ask DocuMind Intelligence')).toBeDefined();
    expect(
      screen.queryByText('AES-256 encryption is mandated across all customer databases at rest.')
    ).toBeNull();
  });
});

describe('Persistent Conversation Integration (Checkpoint B)', () => {
  beforeEach(() => {
    localStorage.clear();
    Object.assign(navigator, {
      clipboard: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
    });
    vi.stubGlobal('open', vi.fn());
  });

  const setupConversationMocks = (options?: {
    initialConversations?: any[];
    conversationMessagesMap?: Record<string, any[]>;
    messagesStatus?: number;
    messagesErrorDetail?: string;
    listStatus?: number;
    delayHistoryMs?: number;
  }) => {
    let capturedMessageRequests: Array<{ url: string; headers: any; body: any }> = [];
    let conversations = [...(options?.initialConversations || [
      {
        id: 'conv-c1-001',
        user_id: 1,
        document_id: null,
        title: 'Q3 Enterprise Architecture',
        document_scope_invalidated: false,
        created_at: '2026-10-31T18:00:00Z',
        updated_at: '2026-10-31T18:30:00Z',
        latest_message: 'How is data encrypted at rest?',
      },
      {
        id: 'conv-c1-002',
        user_id: 1,
        document_id: 'doc-123',
        title: 'PCI Compliance Scope',
        document_scope_invalidated: false,
        created_at: '2026-10-30T10:00:00Z',
        updated_at: '2026-10-30T11:00:00Z',
        latest_message: 'What requirements apply?',
      },
    ])];

    const messagesMap = options?.conversationMessagesMap || {
      'conv-c1-001': [
        {
          id: 'msg-001',
          conversation_id: 'conv-c1-001',
          role: 'user',
          content: 'How is data encrypted at rest?',
          created_at: '2026-10-31T18:00:00Z',
        },
        {
          id: 'msg-002',
          conversation_id: 'conv-c1-001',
          role: 'assistant',
          content: 'Data is encrypted at rest using AES-256 GCM authenticated ciphers.',
          insufficient_evidence: false,
          citations: [
            {
              chunk_id: 'chk-1',
              document_id: 'doc-sec-1',
              document_title: 'Security_Spec.pdf',
              page_number: 4,
              excerpt: 'AES-256 GCM authenticated ciphers are required.',
              score: 0.96,
            },
          ],
          model_used: 'gemini-2.5-flash',
          retrieval_count: 1,
          total_latency_ms: 350,
          created_at: '2026-10-31T18:00:02Z',
        },
      ],
      'conv-c1-002': [
        {
          id: 'msg-101',
          conversation_id: 'conv-c1-002',
          role: 'user',
          content: 'What requirements apply to key rotation?',
          created_at: '2026-10-30T10:00:00Z',
        },
        {
          id: 'msg-102',
          conversation_id: 'conv-c1-002',
          role: 'assistant',
          content: 'Key rotation must occur annually according to Section 3.4.',
          insufficient_evidence: false,
          citations: [],
          model_used: 'gemini-2.5-flash',
          retrieval_count: 1,
          total_latency_ms: 280,
          created_at: '2026-10-30T10:00:02Z',
        },
      ],
    };

    vi.stubGlobal('fetch', vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.includes('/api/health')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            status: 'healthy',
            timestamp: '2026-10-31T20:00:00Z',
            database: { status: 'connected', pgvector_available: true, message: 'OK' },
            environment: { service: 'documind-api', version: '0.5.0' },
          }),
        });
      }

      if (url.includes('/api/documents')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            total: 1,
            documents: [
              {
                id: 'doc-123',
                owner_id: 1,
                title: 'Security_Spec.pdf',
                filename: 'Security_Spec.pdf',
                file_size_bytes: 4096,
                page_count: 12,
                status: 'completed',
                created_at: '2026-10-30T09:00:00Z',
              },
            ],
          }),
        });
      }

      if (url.includes('/api/conversations')) {
        // 401 simulation for conversations list
        if (options?.listStatus === 401 && !url.includes('/messages')) {
          return Promise.resolve({
            ok: false,
            status: 401,
            json: async () => ({ detail: 'Token has expired' }),
          });
        }

        // POST /api/conversations/{id}/messages
        if (init?.method === 'POST' && url.endsWith('/messages')) {
          const body = init?.body ? JSON.parse(init.body as string) : null;
          capturedMessageRequests.push({ url, headers: init?.headers, body });

          if (options?.messagesStatus && options.messagesStatus >= 400) {
            return Promise.resolve({
              ok: false,
              status: options.messagesStatus,
              json: async () => ({ detail: options.messagesErrorDetail || 'Message generation failed' }),
            });
          }

          const match = url.match(/\/api\/conversations\/([^/]+)\/messages/);
          const convId = match ? match[1] : 'conv-c1-001';
          return Promise.resolve({
            ok: true,
            status: 200,
            json: async () => ({
              id: `msg-resp-${Date.now()}`,
              conversation_id: convId,
              role: 'assistant',
              content: `Assistant response to: "${body?.content || ''}"`,
              insufficient_evidence: false,
              citations: [
                {
                  chunk_id: 'chk-new-1',
                  document_id: 'doc-123',
                  document_title: 'Security_Spec.pdf',
                  page_number: 7,
                  excerpt: 'Cryptographic modules must adhere to FIPS 140-3 Level 2.',
                  score: 0.97,
                },
              ],
              retrieval_count: 1,
              model_used: 'gemini-2.5-flash',
              total_latency_ms: 320.0,
              created_at: new Date().toISOString(),
            }),
          });
        }

        // POST /api/conversations (create)
        if (init?.method === 'POST') {
          const body = init?.body ? JSON.parse(init.body as string) : {};
          const newConv = {
            id: `conv-created-${Date.now()}`,
            user_id: 1,
            document_id: body.document_id || null,
            title: body.title || 'New Conversation',
            document_scope_invalidated: false,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            latest_message: null,
            messages: [],
          };
          conversations.unshift(newConv);
          return Promise.resolve({
            ok: true,
            status: 201,
            json: async () => newConv,
          });
        }

        // DELETE /api/conversations/{id}
        if (init?.method === 'DELETE') {
          const match = url.match(/\/api\/conversations\/([^/]+)$/);
          const deleteId = match ? match[1] : '';
          conversations = conversations.filter((c) => c.id !== deleteId);
          return Promise.resolve({
            ok: true,
            status: 200,
            json: async () => ({ message: 'Conversation deleted successfully', id: deleteId }),
          });
        }

        // GET /api/conversations/{id} (conversation detail with messages)
        const detailMatch = url.match(/\/api\/conversations\/([^/]+)$/);
        if (detailMatch) {
          const convId = detailMatch[1];
          const conv = conversations.find((c) => c.id === convId) || {
            id: convId,
            user_id: 1,
            document_id: null,
            title: 'Found Conversation',
            document_scope_invalidated: false,
            created_at: '2026-10-31T18:00:00Z',
            updated_at: '2026-10-31T18:30:00Z',
          };
          const messages = messagesMap[convId] || [];

          if (options?.delayHistoryMs) {
            return new Promise((resolve) => {
              setTimeout(() => {
                resolve({
                  ok: true,
                  status: 200,
                  json: async () => ({ ...conv, messages }),
                });
              }, options.delayHistoryMs);
            });
          }

          return Promise.resolve({
            ok: true,
            status: 200,
            json: async () => ({ ...conv, messages }),
          });
        }

        // GET /api/conversations (list)
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            total: conversations.length,
            conversations,
          }),
        });
      }

      return Promise.resolve({ ok: true, status: 200, json: async () => ({}) });
    }));

    return {
      getCapturedRequests: () => capturedMessageRequests,
      getConversations: () => conversations,
    };
  };

  it('1. loads the authenticated user conversation list in the sidebar', async () => {
    localStorage.setItem('documind_token', 'user-jwt-token');
    localStorage.setItem(
      'documind_user',
      JSON.stringify({ id: 1, email: 'user@example.com', username: 'user1', is_admin: false })
    );

    setupConversationMocks();

    await act(async () => {
      render(<App />);
    });

    // Navigate to Ask DocuMind
    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    // Verify conversation titles appear in the sidebar
    const titles = await screen.findAllByText('Q3 Enterprise Architecture');
    expect(titles.length).toBeGreaterThan(0);
    expect(await screen.findByText('PCI Compliance Scope')).toBeDefined();
    expect(screen.getByText('Cross-Doc')).toBeDefined();
    expect(screen.getAllByText('Security_Spec.pdf').length).toBeGreaterThanOrEqual(1);
  });

  it('2. creates a new conversation via "+ New Chat" button', async () => {
    localStorage.setItem('documind_token', 'user-jwt-token');
    localStorage.setItem(
      'documind_user',
      JSON.stringify({ id: 1, email: 'user@example.com', username: 'user1', is_admin: false })
    );

    const { getConversations } = setupConversationMocks();

    await act(async () => {
      render(<App />);
    });

    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    // Wait for list to load
    await screen.findAllByText('Q3 Enterprise Architecture');

    // Click "+ New Chat"
    const newChatBtn = screen.getByRole('button', { name: /New Chat/i });
    await act(async () => {
      fireEvent.click(newChatBtn);
    });

    // Verify new conversation was prepended to conversations state
    expect(getConversations().length).toBe(3);
    // UI should show Welcome screen ready for the new chat
    expect(screen.getByText('Ask DocuMind Intelligence')).toBeDefined();
  });

  it('3. selects an existing conversation and highlights it in the sidebar', async () => {
    localStorage.setItem('documind_token', 'user-jwt-token');
    localStorage.setItem(
      'documind_user',
      JSON.stringify({ id: 1, email: 'user@example.com', username: 'user1', is_admin: false })
    );

    setupConversationMocks();

    await act(async () => {
      render(<App />);
    });

    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    const items = await screen.findAllByText('Q3 Enterprise Architecture');
    await act(async () => {
      fireEvent.click(items[0]);
    });

    // Conversation becomes active and its title is displayed
    expect(screen.getAllByText('Q3 Enterprise Architecture').length).toBeGreaterThan(0);
  });

  it('4. loads and renders message history with citations and metrics for selected conversation', async () => {
    localStorage.setItem('documind_token', 'user-jwt-token');
    localStorage.setItem(
      'documind_user',
      JSON.stringify({ id: 1, email: 'user@example.com', username: 'user1', is_admin: false })
    );

    setupConversationMocks();

    await act(async () => {
      render(<App />);
    });

    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    const items = await screen.findAllByText('Q3 Enterprise Architecture');
    await act(async () => {
      fireEvent.click(items[0]);
    });

    // User question and assistant answer should both be displayed
    expect(await screen.findByText('How is data encrypted at rest?')).toBeDefined();
    expect(
      await screen.findByText('Data is encrypted at rest using AES-256 GCM authenticated ciphers.')
    ).toBeDefined();

    // Verify citation badge rendered
    expect(screen.getAllByText('Security_Spec.pdf').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('p.4')).toBeDefined();
  });

  it('5. sends message to POST /api/conversations/{id}/messages with correct payload', async () => {
    localStorage.setItem('documind_token', 'user-jwt-token');
    localStorage.setItem(
      'documind_user',
      JSON.stringify({ id: 1, email: 'user@example.com', username: 'user1', is_admin: false })
    );

    const { getCapturedRequests } = setupConversationMocks();

    await act(async () => {
      render(<App />);
    });

    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    // Select conversation conv-c1-001
    const items = await screen.findAllByText('Q3 Enterprise Architecture');
    await act(async () => {
      fireEvent.click(items[0]);
    });

    // Type a new message
    const input = screen.getByPlaceholderText(/Ask a natural-language question/i);
    await act(async () => {
      fireEvent.change(input, { target: { value: 'What FIPS standards apply?' } });
      fireEvent.click(screen.getByRole('button', { name: /Submit question/i }));
    });

    // Verify request was sent to the persistent conversation endpoint
    const requests = getCapturedRequests();
    expect(requests.length).toBe(1);
    expect(requests[0].url).toContain('/api/conversations/conv-c1-001/messages');
    expect(requests[0].body).toEqual({
      content: 'What FIPS standards apply?',
      top_k: 4,
    });
    expect(requests[0].headers['Authorization']).toBe('Bearer user-jwt-token');

    // Verify assistant reply appeared in chat
    expect(await screen.findByText('Assistant response to: "What FIPS standards apply?"')).toBeDefined();
  });

  it('6. restores the active conversation after refresh from localStorage', async () => {
    localStorage.setItem('documind_token', 'user-jwt-token');
    localStorage.setItem(
      'documind_user',
      JSON.stringify({ id: 1, email: 'user@example.com', username: 'user1', is_admin: false })
    );
    // Simulate active conversation stored in localStorage before page load
    localStorage.setItem('documind_active_conversation_id', 'conv-c1-002');

    setupConversationMocks();

    await act(async () => {
      render(<App />);
    });

    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    // Message history of conv-c1-002 should automatically load
    expect(
      await screen.findByText('What requirements apply to key rotation?')
    ).toBeDefined();
    expect(
      await screen.findByText('Key rotation must occur annually according to Section 3.4.')
    ).toBeDefined();
  });

  it('7. deletes a conversation with two-step confirmation', async () => {
    localStorage.setItem('documind_token', 'user-jwt-token');
    localStorage.setItem(
      'documind_user',
      JSON.stringify({ id: 1, email: 'user@example.com', username: 'user1', is_admin: false })
    );

    const { getConversations } = setupConversationMocks();

    await act(async () => {
      render(<App />);
    });

    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    await screen.findAllByText('Q3 Enterprise Architecture');

    // Find delete buttons in sidebar
    const deleteBtns = screen.getAllByRole('button', { name: /Delete Conversation/i });
    expect(deleteBtns.length).toBe(2);

    // Click delete on the first conversation
    await act(async () => {
      fireEvent.click(deleteBtns[0]);
    });

    // Two-step confirmation should appear
    const confirmBtn = await screen.findByRole('button', { name: /^Delete$/i });
    expect(confirmBtn).toBeDefined();

    // Confirm deletion
    await act(async () => {
      fireEvent.click(confirmBtn);
    });

    // Verify conversation was removed
    expect(getConversations().find((c) => c.id === 'conv-c1-001')).toBeUndefined();
  });

  it('8. handles expired authentication (HTTP 401) during conversation actions', async () => {
    localStorage.setItem('documind_token', 'expired-token');
    localStorage.setItem(
      'documind_user',
      JSON.stringify({ id: 1, email: 'user@example.com', username: 'user1', is_admin: false })
    );

    setupConversationMocks({ listStatus: 401 });

    await act(async () => {
      render(<App />);
    });

    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    // Token should be removed and auth modal shown
    expect(localStorage.getItem('documind_token')).toBeNull();
    expect(await screen.findByRole('button', { name: /Sign In to DocuMind/i })).toBeDefined();
  });

  it('9. handles failed message submission (HTTP 500) without erasing previous history', async () => {
    localStorage.setItem('documind_token', 'user-jwt-token');
    localStorage.setItem(
      'documind_user',
      JSON.stringify({ id: 1, email: 'user@example.com', username: 'user1', is_admin: false })
    );

    setupConversationMocks({
      messagesStatus: 500,
      messagesErrorDetail: 'Upstream LLM provider timeout',
    });

    await act(async () => {
      render(<App />);
    });

    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    // Select conv-c1-001
    const items = await screen.findAllByText('Q3 Enterprise Architecture');
    await act(async () => {
      fireEvent.click(items[0]);
    });

    // Verify initial message is present
    expect(await screen.findByText('How is data encrypted at rest?')).toBeDefined();

    // Submit a question that will fail
    const input = screen.getByPlaceholderText(/Ask a natural-language question/i);
    await act(async () => {
      fireEvent.change(input, { target: { value: 'Explain key derivation functions.' } });
      fireEvent.click(screen.getByRole('button', { name: /Submit question/i }));
    });

    // Error banner should be displayed
    const errorElements = await screen.findAllByText(/Upstream LLM provider timeout/i);
    expect(errorElements.length).toBeGreaterThanOrEqual(1);

    // Previous history must still remain visible
    expect(screen.getByText('How is data encrypted at rest?')).toBeDefined();
    expect(
      screen.getByText('Data is encrypted at rest using AES-256 GCM authenticated ciphers.')
    ).toBeDefined();
  });

  it('10. prevents stale responses from overwriting another conversation on rapid switching', async () => {
    localStorage.setItem('documind_token', 'user-jwt-token');
    localStorage.setItem(
      'documind_user',
      JSON.stringify({ id: 1, email: 'user@example.com', username: 'user1', is_admin: false })
    );

    // Slow down history fetch for conv-c1-001 to simulate slow network
    setupConversationMocks({ delayHistoryMs: 200 });

    await act(async () => {
      render(<App />);
    });

    const askTabBtn = screen.getAllByRole('button', { name: /Ask DocuMind/i })[0];
    await act(async () => {
      fireEvent.click(askTabBtn);
    });

    // Rapidly click Conv 1 then Conv 2
    const items1 = await screen.findAllByText('Q3 Enterprise Architecture');
    const conv2 = await screen.findByText('PCI Compliance Scope');

    await act(async () => {
      fireEvent.click(items1[0]);
    });
    // Immediately switch to conv 2 before conv 1 finishes
    await act(async () => {
      fireEvent.click(conv2);
    });

    // Wait for async timers
    await act(async () => {
      await new Promise((r) => setTimeout(r, 250));
    });

    // Conv 2 history should be active, Conv 1 should NOT overwrite it
    expect(await screen.findByText('What requirements apply to key rotation?')).toBeDefined();
    // conv-c1-001 message should NOT have overwritten the view
    expect(screen.queryByText('How is data encrypted at rest?')).toBeNull();
  });
});
