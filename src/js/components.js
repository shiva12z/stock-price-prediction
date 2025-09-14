// Reusable UI components
export class LoadingSpinner {
  static create(text = 'Loading...') {
    const spinner = document.createElement('div');
    spinner.className = 'loading-spinner';
    spinner.innerHTML = `
      <div class="spinner-container">
        <div class="spinner"></div>
        <p class="spinner-text">${text}</p>
      </div>
    `;
    return spinner;
  }

  static show(container, text = 'Loading...') {
    const existing = container.querySelector('.loading-spinner');
    if (existing) existing.remove();
    
    const spinner = this.create(text);
    container.appendChild(spinner);
    return spinner;
  }

  static hide(container) {
    const spinner = container.querySelector('.loading-spinner');
    if (spinner) spinner.remove();
  }
}

export class Toast {
  static show(message, type = 'info', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
      <div class="toast-content">
        <span class="toast-message">${message}</span>
        <button class="toast-close" onclick="this.parentElement.parentElement.remove()">×</button>
      </div>
    `;

    // Add to page
    let container = document.querySelector('.toast-container');
    if (!container) {
      container = document.createElement('div');
      container.className = 'toast-container';
      document.body.appendChild(container);
    }
    
    container.appendChild(toast);

    // Auto remove
    setTimeout(() => {
      if (toast.parentElement) {
        toast.remove();
      }
    }, duration);

    return toast;
  }

  static success(message, duration) {
    return this.show(message, 'success', duration);
  }

  static error(message, duration) {
    return this.show(message, 'error', duration);
  }

  static warning(message, duration) {
    return this.show(message, 'warning', duration);
  }
}

export class Modal {
  constructor(id, title, content) {
    this.id = id;
    this.title = title;
    this.content = content;
    this.element = null;
  }

  create() {
    const modal = document.createElement('div');
    modal.id = this.id;
    modal.className = 'modal';
    modal.innerHTML = `
      <div class="modal-overlay" onclick="this.parentElement.remove()"></div>
      <div class="modal-content">
        <div class="modal-header">
          <h3 class="modal-title">${this.title}</h3>
          <button class="modal-close" onclick="this.closest('.modal').remove()">×</button>
        </div>
        <div class="modal-body">
          ${this.content}
        </div>
      </div>
    `;
    this.element = modal;
    return modal;
  }

  show() {
    if (!this.element) this.create();
    document.body.appendChild(this.element);
    
    // Prevent body scroll
    document.body.style.overflow = 'hidden';
    
    // Add event listener for escape key
    const handleEscape = (e) => {
      if (e.key === 'Escape') {
        this.hide();
        document.removeEventListener('keydown', handleEscape);
      }
    };
    document.addEventListener('keydown', handleEscape);
  }

  hide() {
    if (this.element && this.element.parentElement) {
      this.element.remove();
      document.body.style.overflow = '';
    }
  }
}

export class SearchDropdown {
  constructor(inputElement, onSelect) {
    this.input = inputElement;
    this.onSelect = onSelect;
    this.dropdown = null;
    this.results = [];
    this.selectedIndex = -1;
    
    this.init();
  }

  init() {
    // Create dropdown element
    this.dropdown = document.createElement('div');
    this.dropdown.className = 'search-dropdown';
    this.input.parentElement.appendChild(this.dropdown);

    // Add event listeners
    this.input.addEventListener('input', this.handleInput.bind(this));
    this.input.addEventListener('keydown', this.handleKeydown.bind(this));
    document.addEventListener('click', this.handleClickOutside.bind(this));
  }

  handleInput(e) {
    const query = e.target.value.trim();
    if (query.length < 2) {
      this.hide();
      return;
    }

    // Debounce search
    clearTimeout(this.searchTimeout);
    this.searchTimeout = setTimeout(() => {
      this.search(query);
    }, 300);
  }

  handleKeydown(e) {
    if (!this.dropdown.classList.contains('show')) return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        this.selectedIndex = Math.min(this.selectedIndex + 1, this.results.length - 1);
        this.updateSelection();
        break;
      case 'ArrowUp':
        e.preventDefault();
        this.selectedIndex = Math.max(this.selectedIndex - 1, -1);
        this.updateSelection();
        break;
      case 'Enter':
        e.preventDefault();
        if (this.selectedIndex >= 0) {
          this.selectResult(this.results[this.selectedIndex]);
        }
        break;
      case 'Escape':
        this.hide();
        break;
    }
  }

  handleClickOutside(e) {
    if (!this.input.contains(e.target) && !this.dropdown.contains(e.target)) {
      this.hide();
    }
  }

  async search(query) {
    try {
      const { apiClient } = await import('./api.js');
      const results = await apiClient.searchStocks(query);
      this.showResults(results);
    } catch (error) {
      console.error('Search failed:', error);
      this.showError('Search failed. Please try again.');
    }
  }

  showResults(results) {
    this.results = results;
    this.selectedIndex = -1;

    if (results.length === 0) {
      this.dropdown.innerHTML = '<div class="search-no-results">No results found</div>';
    } else {
      this.dropdown.innerHTML = results.map((result, index) => `
        <div class="search-result" data-index="${index}">
          <div class="search-result-symbol">${result.symbol}</div>
          <div class="search-result-name">${result.name}</div>
          <div class="search-result-exchange">${result.exchange}</div>
        </div>
      `).join('');

      // Add click listeners
      this.dropdown.querySelectorAll('.search-result').forEach((element, index) => {
        element.addEventListener('click', () => this.selectResult(results[index]));
      });
    }

    this.show();
  }

  showError(message) {
    this.dropdown.innerHTML = `<div class="search-error">${message}</div>`;
    this.show();
  }

  selectResult(result) {
    this.input.value = result.symbol;
    this.hide();
    if (this.onSelect) {
      this.onSelect(result);
    }
  }

  updateSelection() {
    this.dropdown.querySelectorAll('.search-result').forEach((element, index) => {
      element.classList.toggle('selected', index === this.selectedIndex);
    });
  }

  show() {
    this.dropdown.classList.add('show');
  }

  hide() {
    this.dropdown.classList.remove('show');
    this.selectedIndex = -1;
  }
}