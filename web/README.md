# RagWall React App

This is the React/Next.js version of the RagWall demo, featuring a modern, animated landing page with live API integration.

## Setup Instructions

### 1. Fix npm permissions (if needed)
```bash
sudo chown -R $(whoami) ~/.npm
```

### 2. Install dependencies
```bash
cd web
npm install --legacy-peer-deps
```

### 3. Start the RagWall API (in another terminal)
```bash
cd /Users/rjd/Documents/ragwall
RAGWALL_VECTORS=experiments/results/tiny_jb_vectors.pt python scripts/serve_api.py
```

The API should be running at http://127.0.0.1:8000

### 4. Start the React app
```bash
npm run dev
```

The React app will be available at http://localhost:3000

## Features

- **Modern UI**: Animated hero section with Framer Motion
- **Live Demo**: Interactive sanitization and reranking demos
- **Real-time Metrics**: Shows actual API performance
- **Responsive Design**: Works on all devices
- **TypeScript**: Full type safety
- **Tailwind CSS**: Utility-first styling

## Project Structure

```
web/
├── app/
│   ├── page.tsx        # Main landing page component
│   ├── layout.tsx      # Root layout
│   └── globals.css     # Global styles
├── package.json        # Dependencies
├── next.config.js      # Next.js configuration
├── tailwind.config.ts  # Tailwind configuration
└── .env.local         # Environment variables
```

## Environment Variables

The API endpoint is configured in `.env.local`:
```
NEXT_PUBLIC_RAGWALL_API=http://127.0.0.1:8000
```

## Differences from Static Version

| Feature | React Version | Static Version |
|---------|--------------|----------------|
| Framework | Next.js 14 | Vanilla HTML |
| Animations | Framer Motion | CSS only |
| State Management | React hooks | Vanilla JS |
| Build Size | ~200KB | ~50KB |
| Dev Experience | Hot reload | Manual refresh |
| Deployment | Vercel/Node.js | Static hosting |

## Deployment

### For Development
```bash
npm run dev
```

### For Production
```bash
npm run build
npm run start
```

### Deploy to Vercel
```bash
npx vercel
```

## Troubleshooting

### npm permission errors
```bash
sudo chown -R $(whoami) ~/.npm
sudo chown -R $(whoami) node_modules
```

### API connection issues
- Make sure the Python API is running on port 8000
- Check CORS is enabled in the API
- Verify `.env.local` has correct API URL

### Module not found errors
```bash
rm -rf node_modules package-lock.json
npm install --legacy-peer-deps
```

## Tech Stack

- **Next.js 14**: React framework with App Router
- **TypeScript**: Type safety
- **Tailwind CSS**: Utility-first CSS
- **Framer Motion**: Animation library
- **clsx**: Conditional className utility

## Why Two Versions?

1. **Static (`public/index.html`)**: 
   - Simple, fast, works anywhere
   - No build process needed
   - Good for demos and Docker

2. **React (`web/`)**: 
   - Rich interactions and animations
   - Better developer experience
   - Easier to extend with features
   - Better for SaaS product

Choose based on your needs!