
const express = require('express');
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const OpenAI = require('openai');
const fs = require('fs');
const bodyParser = require('body-parser');
require('dotenv/config');

// Applying the stealth plugin to Puppeteer
puppeteer.use(StealthPlugin());

const app = express();
const port = process.env.PORT || 3000;

app.use(bodyParser.json());

async function imageToBase64(image_file) {
    return new Promise((resolve, reject) => {
        fs.readFile(image_file, (err, data) => {
            if (err) {
                reject(err);
                return;
            }
            const base64Data = data.toString('base64');
            resolve(`data:image/jpeg;base64,${base64Data}`);
        });
    });
}

async function startCrawler(url, prompt) {
    const browser = await puppeteer.launch({ headless: false });
    const page = await browser.newPage();

    await page.setViewport({ width: 1200, height: 1200 });

    const messages = [];
    messages.push({
        role: 'system',
        content: 'Welcome to GPT4V-Browsing. Feel free to ask questions or give instructions.',
    });

    if (url) {
        await page.goto(url, { waitUntil: 'domcontentloaded' });
    }

    const response = await new OpenAI().chat.completions.create({
        model: 'gpt-4-vision-preview',
        max_tokens: 1024,
        messages,
    });

    const message = response.choices[0].message;
    messages.push({ role: 'assistant', content: message.content });

    if (message.content.includes('{"click": "')) {
        // Extract link text and click on it
        const linkText = message.content.split('{"click": "')[1].split('"}')[0];
        const element = await page.$(`[gpt-link-text="${linkText}"]`);
        await element.click();
    } else if (message.content.includes('{"url": "')) {
        // Extract URL and navigate to it
        const newUrl = message.content.split('{"url": "')[1].split('"}')[0];
        await page.goto(newUrl, { waitUntil: 'domcontentloaded' });
    }

    await page.screenshot({ path: 'screenshot.jpg', fullPage: true });

    return await imageToBase64('screenshot.jpg');
}

app.post('/crawl', async (req, res) => {
    const { url, prompt } = req.body;
    try {
        const screenshotBase64 = await startCrawler(url, prompt);
        res.status(200).json({ success: true, screenshot: screenshotBase64 });
    } catch (error) {
        console.error('Error during crawling:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

app.listen(port, () => console.log(`Server listening on port ${port}`));
