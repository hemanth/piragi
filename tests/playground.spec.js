const { test, expect } = require('@playwright/test');

const BASE = 'http://localhost:8787';

test.describe('piragi playground', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(BASE, { waitUntil: 'networkidle' });
  });

  test('landing page visible with logo and features', async ({ page }) => {
    await expect(page.locator('#landing')).toBeVisible();
    await expect(page.locator('.landing-logo')).toHaveText('piragi');
    await expect(page.locator('.landing-card')).toHaveCount(12);
    await expect(page.locator('.landing-term-body')).toContainText('Ragi');
  });

  test('open playground hides landing', async ({ page }) => {
    await page.click('.landing-enter');
    await page.waitForTimeout(700);
    await expect(page.locator('#landing')).toHaveClass(/hidden/);
  });

  test('backend is live', async ({ page }) => {
    await page.click('.landing-enter');
    await expect(page.locator('#status-dot')).toHaveClass(/live/);
  });

  test('10 tabs in sidebar', async ({ page }) => {
    await page.click('.landing-enter');
    await expect(page.locator('.tab')).toHaveCount(10);
  });

  test('cell 1: Basic RAG runs', async ({ page }) => {
    test.setTimeout(120000);
    await page.click('.landing-enter');
    await page.click('#tab-0');
    await expect(page.locator('#cell-title')).toHaveText('Basic RAG');
    await page.click('#run-btn');
    await page.waitForFunction(() => {
      const btn = document.getElementById('run-btn');
      return btn && !btn.classList.contains('running');
    }, { timeout: 90000 });
    const output = await page.locator('#output').textContent();
    expect(output).toContain('indexed');
    expect(output).toContain('deploy.md');
    expect(output).not.toContain('Traceback');
  });

  test('cell 2: Embeddings runs', async ({ page }) => {
    test.setTimeout(60000);
    await page.click('.landing-enter');
    await page.click('#tab-1');
    await expect(page.locator('#cell-title')).toHaveText('Embeddings');
    await page.click('#run-btn');
    await page.waitForFunction(() => {
      const btn = document.getElementById('run-btn');
      return btn && !btn.classList.contains('running');
    }, { timeout: 45000 });
    const output = await page.locator('#output').textContent();
    expect(output).toContain('dim:');
    expect(output).toContain('384');
    expect(output).not.toContain('Traceback');
  });

  test('cell 3: Shared Model runs', async ({ page }) => {
    test.setTimeout(120000);
    await page.click('.landing-enter');
    await page.click('#tab-2');
    await expect(page.locator('#cell-title')).toHaveText('Shared Model');
    await page.click('#run-btn');
    await page.waitForFunction(() => {
      const btn = document.getElementById('run-btn');
      return btn && !btn.classList.contains('running');
    }, { timeout: 90000 });
    const output = await page.locator('#output').textContent();
    expect(output).toContain('kb1:');
    expect(output).toContain('kb2:');
    expect(output).toContain('shared embedder: True');
    expect(output).not.toContain('Traceback');
  });

  test('cell 4: Qdrant Store runs', async ({ page }) => {
    await page.click('.landing-enter');
    await page.click('#tab-3');
    await expect(page.locator('#cell-title')).toHaveText('Qdrant Store');
    await page.click('#run-btn');
    await page.waitForFunction(() => {
      const btn = document.getElementById('run-btn');
      return btn && !btn.classList.contains('running');
    }, { timeout: 15000 });
    const output = await page.locator('#output').textContent();
    expect(output).toContain('QdrantStore');
    expect(output).toContain('chunks:');
    expect(output).not.toContain('Traceback');
  });

  test('cell 5: Async runs', async ({ page }) => {
    test.setTimeout(120000);
    await page.click('.landing-enter');
    await page.click('#tab-7');
    await expect(page.locator('#cell-title')).toHaveText('Async');
    await page.click('#run-btn');
    await page.waitForFunction(() => {
      const btn = document.getElementById('run-btn');
      return btn && !btn.classList.contains('running');
    }, { timeout: 90000 });
    const output = await page.locator('#output').textContent();
    expect(output).toContain('AsyncRagi');
    expect(output).toContain('async methods:');
    expect(output).not.toContain('Traceback');
  });

  test('editor input and custom code execution', async ({ page }) => {
    test.setTimeout(30000);
    await page.click('.landing-enter');
    await page.click('#tab-0');
    await page.evaluate(() => {
      const ace = window.aceEditor || ace.edit('ace-editor');
      ace.setValue('print("hello from playwright")', -1);
    });
    await page.click('#run-btn');
    await page.waitForFunction(() => {
      const btn = document.getElementById('run-btn');
      return btn && !btn.classList.contains('running');
    }, { timeout: 15000 });
    await expect(page.locator('#output')).toContainText('hello from playwright');
  });

  test('clear button empties output', async ({ page }) => {
    await page.click('.landing-enter');
    await page.click('#tab-0');
    await page.evaluate(() => {
      const ace = window.aceEditor || ace.edit('ace-editor');
      ace.setValue('print("test clear")', -1);
    });
    await page.click('#run-btn');
    await page.waitForFunction(() => {
      const btn = document.getElementById('run-btn');
      return btn && !btn.classList.contains('running');
    }, { timeout: 15000 });
    await page.click('#output-bar .output-clear');
    await page.waitForTimeout(300);
    const text = await page.locator('#output').textContent();
    expect(text.length).toBe(0);
  });

  test('output pane is resizable', async ({ page }) => {
    await page.click('.landing-enter');
    const out = page.locator('#output');
    const before = await out.evaluate(el => el.offsetHeight);
    const bar = page.locator('#output-bar');
    const box = await bar.boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2, box.y - 100, { steps: 5 });
    await page.mouse.up();
    const after = await out.evaluate(el => el.offsetHeight);
    expect(after).toBeGreaterThan(before);
  });

  test('LLM config panel exists and has inputs', async ({ page }) => {
    await page.click('.landing-enter');
    await page.click('#llm-details summary');
    await page.waitForTimeout(300);
    await expect(page.locator('#llm-model')).toBeVisible();
    await expect(page.locator('#llm-url')).toBeVisible();
    await expect(page.locator('#llm-key')).toBeVisible();
    const model = await page.locator('#llm-model').inputValue();
    expect(model.length).toBeGreaterThan(0);
  });

  test('tab output persists across switches', async ({ page }) => {
    test.setTimeout(30000);
    await page.click('.landing-enter');
    await page.click('#tab-0');
    await page.evaluate(() => {
      window.aceEditor.setValue('print("persist test")', -1);
    });
    await page.click('#run-btn');
    await page.waitForFunction(() => {
      const btn = document.getElementById('run-btn');
      return btn && !btn.classList.contains('running');
    }, { timeout: 15000 });
    // Switch away
    await page.click('#tab-1');
    await page.waitForTimeout(300);
    // Switch back
    await page.click('#tab-0');
    await page.waitForTimeout(300);
    await expect(page.locator('#output')).toContainText('persist test');
  });

  test('data-prep panel shows for cells with setup', async ({ page }) => {
    await page.click('.landing-enter');
    // Cell 0 (Basic RAG) has setup
    await page.click('#tab-0');
    await expect(page.locator('#data-prep')).toBeVisible();
    await expect(page.locator('#data-prep-code')).toContainText('tempfile.mkdtemp');
    // Cell 1 (Embeddings) has no setup
    await page.click('#tab-1');
    const dp = page.locator('#data-prep');
    await expect(dp).toHaveCSS('display', 'none');
  });

  test('simulation mode toggle works', async ({ page }) => {
    await page.click('.landing-enter');
    // Enable simulation
    await page.check('#sim-check');
    await expect(page.locator('#status-text')).toHaveText('simulated');
    // Run cell in simulation mode -- should show simulated output
    await page.click('#tab-4'); // Chunking cell
    await page.click('#run-btn');
    await page.waitForFunction(() => {
      const out = document.getElementById('output');
      return out && out.textContent.length > 5 && !out.textContent.includes('running...');
    }, { timeout: 5000 });
    const output = await page.locator('#output').textContent();
    expect(output).toContain('fixed:');
    // Disable simulation
    await page.uncheck('#sim-check');
    await expect(page.locator('#status-text')).toHaveText('live');
  });

  test('landing terminal shows kb.retrieve not kb.search', async ({ page }) => {
    const body = await page.locator('.landing-term-body').textContent();
    expect(body).toContain('kb.retrieve');
    expect(body).not.toContain('kb.search');
  });
});
