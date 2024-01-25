
const express = require('express');
const bodyParser = require('body-parser');
require('dotenv/config');
const webCrawlerModule = require('./webCrawlerModuleRefactored'); // Importing the web crawler module

const app = express();
const port = process.env.PORT || 3000;

app.use(bodyParser.json());

app.post('/crawl', async (req, res) => {
    const { url, prompt } = req.body;
    try {
        const screenshotBase64 = await webCrawlerModule.startCrawler(url, prompt);
        res.status(200).json({ success: true, screenshot: screenshotBase64 });
    } catch (error) {
        console.error('Error during crawling:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

app.listen(port, () => console.log(`Server listening on port ${port}`));
