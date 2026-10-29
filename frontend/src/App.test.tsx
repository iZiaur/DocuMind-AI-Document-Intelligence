import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, act, fireEvent } from '@testing-library/react';
import App from './App';

describe('DocuMind AI Shell & Semantic Search UI', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.stubGlobal('fetch', vi.fn().mockImplementation((url: string) => {
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
  });

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
});
