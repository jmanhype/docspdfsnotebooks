require('dotenv').config();

const express = require('express');
const https = require('https');
const fs = require('fs');
const { Client } = require('@notionhq/client');

const privateKey = fs.readFileSync("C:\\Users\\strau\\notion\\key.pem", 'utf8');
const certificate = fs.readFileSync("C:\\Users\\strau\\notion\\cert.pem", 'utf8');

const credentials = { key: privateKey, cert: certificate };
const app = express();
const notion = new Client({ auth: process.env.NOTION_TOKEN });

app.use(express.json());

app.post('/send-data', async (req, res) => {
    const data = req.body.resourceData;

    try {
        for (const entry of data) {
            await notion.pages.create({
                parent: { database_id: process.env.NOTION_DATABASE_ID },
                properties: mapDataToNotionProperties(entry)
            });
        }
        res.json({ message: 'Notion table updated successfully.' });
    } catch (error) {
        console.error('Error updating Notion:', error);
        res.status(500).json({ message: 'Error updating Notion.', error: error.message });
    }
});

function mapDataToNotionProperties(entry) {
    const properties = {
        'Title': { title: [{ text: { content: entry.Title || 'Untitled' } }] },
        'Authors': { rich_text: [{ text: { content: entry.Authors || 'Unknown' } }] },
        'Abstract/Summary': { rich_text: [{ text: { content: entry.AbstractSummary || '' } }] },
        'Publication Date': entry.PublicationDate ? { rich_text: [{ text: { content: entry.PublicationDate } }] } : undefined,
        'Source/Link': entry.SourceLink ? { rich_text: [{ text: { content: entry.SourceLink } }] } : undefined,
        'Tags': typeof entry.TagsKeywords === 'string' ? { multi_select: entry.TagsKeywords.split(',').map(keyword => ({ name: keyword.trim() })) } : undefined,
        'Relevance Score': entry.RelevanceScore ? { rich_text: [{ text: { content: String(entry.RelevanceScore) } }] } : undefined,
        'Notes': entry.Notes ? { rich_text: [{ text: { content: entry.Notes } }] } : undefined,
        'Cited By': entry.CitedBy ? { rich_text: [{ text: { content: entry.CitedBy } }] } : undefined,
        'Impact Factor': entry.ImpactFactor ? { rich_text: [{ text: { content: String(entry.ImpactFactor) } }] } : undefined,
        'Technology Domain': entry.TechnologyDomain ? { rich_text: [{ text: { content: entry.TechnologyDomain } }] } : undefined,
        'Methodology': entry.Methodology ? { rich_text: [{ text: { content: entry.Methodology } }] } : undefined,
        'Dataset Used': entry.DatasetUsed ? { rich_text: [{ text: { content: entry.DatasetUsed } }] } : undefined,
        'Code Availability': entry.CodeAvailability ? { rich_text: [{ text: { content: entry.CodeAvailability } }] } : undefined,
        'Results/Findings': entry.ResultsFindings ? { rich_text: [{ text: { content: entry.ResultsFindings } }] } : undefined,
        'Application to Projects': entry.ApplicationToProjects ? { rich_text: [{ text: { content: entry.ApplicationToProjects } }] } : undefined,
        'Follow-Up Actions': entry.FollowUpActions ? { rich_text: [{ text: { content: entry.FollowUpActions } }] } : undefined,
        'Contributors': entry.Contributors ? { rich_text: [{ text: { content: entry.Contributors } }] } : undefined,
        'Attachments': entry.Attachments ? { rich_text: [{ text: { content: entry.Attachments.join(', ') } }] } : undefined,
    };

    return properties;
};

const httpsServer = https.createServer(credentials, app);

const PORT = process.env.PORT || 3000;

httpsServer.listen(PORT, () => {
    console.log(`HTTPS Server running on port ${PORT}`);
});
