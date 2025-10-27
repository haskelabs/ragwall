#!/bin/bash

echo "🚀 RagWall React App Setup"
echo "=========================="

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    echo ""
    echo "Please run manually:"
    echo "  cd web"
    echo "  sudo npm install --legacy-peer-deps"
    echo ""
    echo "Or if you have permission issues:"
    echo "  sudo chown -R $(whoami) ~/.npm"
    echo "  npm install --legacy-peer-deps"
    exit 1
fi

# Check if API is running
echo "🔍 Checking if RagWall API is running..."
if curl -s http://127.0.0.1:8000/health > /dev/null; then
    echo "✅ API is running on port 8000"
else
    echo "⚠️  API is not running!"
    echo "Please start it in another terminal:"
    echo "  cd /Users/rjd/Documents/ragwall"
    echo "  RAGWALL_VECTORS=experiments/results/tiny_jb_vectors.pt python scripts/serve_api.py"
    exit 1
fi

# Start the Next.js app
echo ""
echo "🎨 Starting React app..."
echo "========================"
npm run dev