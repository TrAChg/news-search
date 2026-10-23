// Frontend client for GallicaLens Multimodal Search

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const searchForm = document.getElementById('search-form');
  const searchInput = document.getElementById('search-input');
  const clearBtn = document.getElementById('clear-btn');
  const searchBtn = document.getElementById('search-btn');
  const resultsBar = document.getElementById('results-bar');
  const resultsCount = document.getElementById('results-count');
  const resultsQuery = document.getElementById('results-query');
  const resultsLatency = document.getElementById('results-latency');
  const cardsGrid = document.getElementById('cards-grid');
  const loader = document.getElementById('loader');
  const emptyState = document.getElementById('empty-state');
  const systemStatusText = document.getElementById('system-status-text');

  // Modal Elements
  const imageModal = document.getElementById('image-modal');
  const modalClose = document.getElementById('modal-close');
  const modalImg = document.getElementById('modal-img');
  const modalTitle = document.getElementById('modal-title');
  const modalScore = document.getElementById('modal-score');
  const modalDistance = document.getElementById('modal-distance');
  const modalId = document.getElementById('modal-id');
  const modalIssue = document.getElementById('modal-issue');
  const modalPage = document.getElementById('modal-page');
  const modalFilename = document.getElementById('modal-filename');
  const modalDirectLink = document.getElementById('modal-direct-link');

  // 1. Initial Health Check
  fetchSystemHealth();

  async function fetchSystemHealth() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error('Health check failed');
      const data = await res.json();
      systemStatusText.textContent = `Online • ${data.total_indexed} Illustration${data.total_indexed !== 1 ? 's' : ''} Indexed`;
    } catch (err) {
      systemStatusText.textContent = 'Backend Offline';
      systemStatusText.parentElement.style.borderColor = 'rgba(239, 68, 68, 0.4)';
    }
  }

  // 2. Input Management
  searchInput.addEventListener('input', () => {
    clearBtn.style.display = searchInput.value ? 'block' : 'none';
  });

  clearBtn.addEventListener('click', () => {
    searchInput.value = '';
    clearBtn.style.display = 'none';
    searchInput.focus();
  });

  // 3. Sample Query Chips
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const query = chip.getAttribute('data-query');
      if (query) {
        searchInput.value = query;
        clearBtn.style.display = 'block';
        performSearch(query);
      }
    });
  });

  // 4. Form Submit
  searchForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = searchInput.value.trim();
    if (query) {
      performSearch(query);
    }
  });

  // 5. Search Execution
  async function performSearch(query) {
    // UI state: loading
    cardsGrid.innerHTML = '';
    emptyState.classList.add('hidden');
    resultsBar.classList.add('hidden');
    loader.classList.remove('hidden');
    searchBtn.disabled = true;

    try {
      const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&top_k=9`);
      if (!response.ok) {
        throw new Error(`Search error: ${response.statusText}`);
      }

      const data = await response.json();
      renderResults(data);
    } catch (err) {
      console.error(err);
      loader.classList.add('hidden');
      emptyState.classList.remove('hidden');
      emptyState.querySelector('h3').textContent = 'Search Error';
      emptyState.querySelector('p').textContent = err.message || 'Unable to connect to search API.';
    } finally {
      searchBtn.disabled = false;
    }
  }

  // 6. Render Search Results
  function renderResults(data) {
    loader.classList.add('hidden');

    // Update stats bar
    resultsCount.textContent = data.count;
    resultsQuery.textContent = `"${data.query}"`;
    resultsLatency.textContent = data.latency_ms;
    resultsBar.classList.remove('hidden');

    if (!data.results || data.results.length === 0) {
      emptyState.classList.remove('hidden');
      emptyState.querySelector('h3').textContent = 'No Confident Illustrations Found';
      emptyState.querySelector('p').textContent = 'All candidates fell below the relevance threshold (min similarity 20%). Try historical terms like "soldat", "guerre", "canon", "tranchée", or "cheval".';
      return;
    }

    data.results.forEach((item) => {
      const card = createCard(item);
      cardsGrid.appendChild(card);
    });
  }

  // 7. Card DOM Creation
  function createCard(item) {
    const card = document.createElement('div');
    card.className = 'result-card';

    const calibratedPct = Math.round((item.calibrated_score !== undefined ? item.calibrated_score : 0) * 100);
    const rawPct = (item.similarity * 100).toFixed(1);
    const isHighMatch = calibratedPct >= 50;

    const issueId = item.metadata?.issue_id || 'Gallica Archive';
    const pageNum = item.metadata?.page ? `Page ${item.metadata.page}` : 'Scan';

    card.innerHTML = `
      <div class="card-image-box">
        <img src="${item.image_url}" alt="${item.id}" loading="lazy" onerror="this.src='/static/ui/fallback.png';">
        <div class="card-badge ${isHighMatch ? 'high-match' : ''}">
          <span>${calibratedPct > 0 ? `${calibratedPct}% Relevance` : `${rawPct}% Raw`}</span>
        </div>
      </div>
      <div class="card-content">
        <div class="card-title">${item.id}</div>
        <div class="card-meta">
          <span class="font-mono">${issueId}</span>
          <span>${pageNum}</span>
        </div>
      </div>
    `;

    card.addEventListener('click', () => openModal(item));
    return card;
  }

  // 8. Lightbox Modal
  function openModal(item) {
    const calibratedPct = Math.round((item.calibrated_score !== undefined ? item.calibrated_score : 0) * 100);
    modalImg.src = item.image_url;
    modalTitle.textContent = item.id;
    modalScore.textContent = `${calibratedPct}% Relevance (Raw Cosine: ${(item.similarity * 100).toFixed(1)}%)`;
    modalDistance.textContent = `Distance: ${item.distance.toFixed(4)}`;

    modalId.textContent = item.id;
    modalIssue.textContent = item.metadata?.issue_id || 'bpt6k717093t';
    modalPage.textContent = item.metadata?.page || '1';
    modalFilename.textContent = item.metadata?.filename || 'crop.jpg';
    modalDirectLink.href = item.image_url;

    imageModal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
  }

  function closeModal() {
    imageModal.classList.add('hidden');
    document.body.style.overflow = 'auto';
  }

  modalClose.addEventListener('click', closeModal);

  imageModal.addEventListener('click', (e) => {
    if (e.target === imageModal) {
      closeModal();
    }
  });

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !imageModal.classList.contains('hidden')) {
      closeModal();
    }
  });
});
