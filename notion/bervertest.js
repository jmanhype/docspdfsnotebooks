require('dotenv').config();

const express = require('express');
const https = require('https');
const fs = require('fs');
const { Client } = require('@notionhq/client');

const privateKey = fs.readFileSync("C:\\Users\\strau\\notion\\key.pem", 'utf8');
const certificate = fs.readFileSync("C:\\Users\\strau\\notion\\cert.pem", 'utf8');
const credentials = { key: privateKey, cert: certificate };
const app = express();

let notionToken = process.env.NOTION_TOKEN;
let notionDatabaseId = process.env.NOTION_DATABASE_ID;
let notion = new Client({ auth: notionToken });

app.use(express.json());

app.post('/config', (req, res) => {
    notionToken = req.body.notionToken;
    notionDatabaseId = req.body.notionDatabaseId;
    notion = new Client({ auth: notionToken });
    res.json({ message: 'Configuration updated successfully.' });
});

app.post('/send-data', async (req, res) => {
    const resourceData = req.body.resourceData;
    try {
        const { properties } = await notion.databases.retrieve({ database_id: notionDatabaseId });
        for (const entry of resourceData) {
            const notionProperties = mapPayloadToSchema(entry, properties);
            await notion.pages.create({
                parent: { database_id: notionDatabaseId },
                properties: notionProperties,
                children: []
            });
        }
        res.json({ message: 'Pages created successfully.' });
    } catch (error) {
        console.error('Error creating pages in Notion:', error);
        res.status(500).json({ message: 'Error creating pages.', error: error.message });
    }
});

app.get('/fetch-page/:pageId', async (req, res) => {
    try {
        const page = await notion.pages.retrieve({ page_id: req.params.pageId });
        res.json(page);
    } catch (error) {
        console.error('Error retrieving page:', error);
        res.status(500).json({ message: 'Error retrieving page.', error: error.message });
    }
});

app.post('/create-page', async (req, res) => {
    try {
        const newPageData = req.body;
        const newPage = await notion.pages.create(newPageData);
        res.json(newPage);
    } catch (error) {
        console.error('Error creating page:', error);
        res.status(500).json({ message: 'Error creating page.', error: error.message });
    }
});

app.get('/fetch-page-property/:pageId/:propertyId', async (req, res) => {
    try {
        const pageId = req.params.pageId;
        const propertyId = req.params.propertyId;
        const response = await notion.pages.properties.retrieve({ 
            page_id: pageId, 
            property_id: propertyId 
        });
        res.json(response);
    } catch (error) {
        console.error('Error retrieving page property item:', error);
        res.status(500).json({ message: 'Error retrieving page property item.', error: error.message });
    }
});

app.get('/fetch-data', async (req, res) => {
    try {
        const response = await notion.databases.query({ database_id: notionDatabaseId });
        res.json(response);
    } catch (error) {
        console.error('Error fetching data from Notion:', error);
        res.status(500).json({ message: 'Error fetching data.', error: error.message });
    }
});

app.put('/update-page', async (req, res) => {
    const { pageId, updates } = req.body;
    try {
        const updateResponse = await notion.pages.update({ page_id: pageId, properties: updates });
        res.json({ message: 'Page updated successfully.', updateResponse });
    } catch (error) {
        console.error('Error updating page:', error);
        res.status(500).json({ message: 'Error updating page.', error: error.message });
    }
});

app.delete('/delete-page', async (req, res) => {
    const { pageId } = req.body;
    try {
        await notion.pages.update({ page_id: pageId, archived: true });
        res.json({ message: 'Page archived successfully.' });
    } catch (error) {
        console.error('Error archiving page:', error);
        res.status(500).json({ message: 'Error archiving page.', error: error.message });
    }
});

app.get('/fetch-blocks/:pageId', async (req, res) => {
    try {
        const blocks = await notion.blocks.children.list({ block_id: req.params.pageId });
        res.json(blocks);
    } catch (error) {
        console.error('Error retrieving blocks:', error);
        res.status(500).json({ message: 'Error retrieving blocks.', error: error.message });
    }
});

app.get('/users', async (req, res) => {
    try {
        const users = await notion.users.list({});
        res.json(users);
    } catch (error) {
        console.error('Error retrieving users:', error);
        res.status(500).json({ message: 'Error retrieving users.', error: error.message });
    }
});

function mapPayloadToSchema(payload, schema) {
    const properties = {};
    for (const [key, config] of Object.entries(schema)) {
        const value = payload[key];
        if (value !== undefined) {
            properties[key] = buildNotionProperty(value, config.type);
        }
    }
    return properties;
}

function buildNotionProperty(value, type) {
    switch (type) {
        case 'title':
        case 'rich_text':
            return { [type]: [{ text: { content: value } }] };
        case 'number':
            return { number: value };
        case 'select':
            return { select: { name: value } };
        case 'multi_select':
            return { multi_select: value.split(',').map(name => ({ name: name.trim() })) };
        case 'date':
            return { date: { start: value } };
        case 'checkbox':
            return { checkbox: value };
        case 'people':
            return { people: value.map(userId => ({ object: 'user', id: userId })) };
        case 'files':
            return { files: value.map(fileUrl => ({ type: 'external', external: { url: fileUrl } })) };
        default:
            return { [type]: value };
    }
}

const httpsServer = https.createServer(credentials, app);
const PORT = process.env.PORT || 4000;
httpsServer.listen(PORT, () => {
    console.log(`Server is running on https://localhost:${PORT}`);
});
