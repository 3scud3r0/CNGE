/** Local browser smoke test. Start tools/live_demo.py before running. */
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium',
    headless: true,
    args: ['--no-sandbox'],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1200, height: 850 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.CNGE_TEST_URL || 'http://127.0.0.1:8081');
    await page.waitForFunction(() => document.querySelector('#status').textContent === 'Ao vivo');
    await page.waitForFunction(() => document.querySelector('#metrics').textContent.includes('dynamics_loss'));
    await page.click('#apply');
    await page.click('#phase', { position: { x: 280, y: 200 } });
    await page.waitForTimeout(1000);
    const interactionError = await page.locator('#error').textContent();
    if (interactionError || errors.length) throw new Error(interactionError || errors.join('; '));
    await page.setViewportSize({ width: 390, height: 844 });
    if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) {
      throw new Error('Mobile layout overflows');
    }
    console.log('PASS: live telemetry, interactions, JavaScript and mobile layout');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
