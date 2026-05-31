/**
 * Settings modal for API key, model, and game configuration.
 */

export interface SettingsData {
  apiKey: string;
  model: string;
  baseUrl: string;
}

export class SettingsView {
  /** Initialize settings panel event handlers. */
  static init(onSave: (data: SettingsData) => void): void {
    const modal = document.getElementById('settings-modal')!;
    const openBtn = document.getElementById('btn-settings');
    const closeBtn = document.getElementById('btn-close-settings');
    const saveBtn = document.getElementById('btn-save-settings');

    // Load saved settings
    const saved = this.load();
    const apiKeyInput = document.getElementById('input-api-key') as HTMLInputElement;
    const modelInput = document.getElementById('input-model') as HTMLInputElement;
    const baseUrlInput = document.getElementById('input-base-url') as HTMLInputElement;

    if (saved.apiKey) apiKeyInput.value = saved.apiKey;
    if (saved.model) modelInput.value = saved.model;
    if (saved.baseUrl) baseUrlInput.value = saved.baseUrl;

    openBtn?.addEventListener('click', () => {
      modal.classList.remove('hidden');
    });

    closeBtn?.addEventListener('click', () => {
      modal.classList.add('hidden');
    });

    saveBtn?.addEventListener('click', () => {
      const data: SettingsData = {
        apiKey: apiKeyInput.value.trim(),
        model: modelInput.value.trim() || 'gpt-4o',
        baseUrl: baseUrlInput.value.trim() || 'https://api.openai.com/v1',
      };
      this.save(data);
      modal.classList.add('hidden');
      onSave(data);
    });

    // Close on backdrop click
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.classList.add('hidden');
      }
    });
  }

  /** Load settings from localStorage. */
  static load(): SettingsData {
    try {
      const raw = localStorage.getItem('tractor-game-settings');
      if (raw) {
        const parsed = JSON.parse(raw);
        return {
          apiKey: parsed.apiKey ?? '',
          model: parsed.model ?? 'gpt-4o',
          baseUrl: parsed.baseUrl ?? 'https://api.openai.com/v1',
        };
      }
    } catch {
      // Ignore parse errors
    }
    return { apiKey: '', model: 'gpt-4o', baseUrl: 'https://api.openai.com/v1' };
  }

  /** Save settings to localStorage. */
  static save(data: SettingsData): void {
    localStorage.setItem('tractor-game-settings', JSON.stringify(data));
  }
}
