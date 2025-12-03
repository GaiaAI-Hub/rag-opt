#!/usr/bin/env bash
set -euo pipefail

# Usage: ./publish.sh [description]
DESCRIPTION=${1:-""}  # Optional first argument as release description

# Extract version from pyproject.toml
VERSION=$(sed -n 's/^version *= *"\([^"]*\)"/\1/p' pyproject.toml)

if [[ -z "$VERSION" ]]; then
    echo "❌ Could not extract version from pyproject.toml"
    exit 1
fi

TAG="v$VERSION"
TAG_MSG="V$VERSION"

echo "📦 Publishing version: $TAG"

# Commit changes
git add .
git commit -m "Publish $TAG" || echo "✔ Nothing to commit"
git push

# Create tag if it doesn't exist
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "⚠️ Tag $TAG already exists! Skipping tag creation."
else
    git tag -a "$TAG" -m "$TAG_MSG"
    git push origin "$TAG"
fi

# Create GitHub release
if gh release view "$TAG" >/dev/null 2>&1; then
    echo "⚠️ Release $TAG already exists! Skipping release creation."
else
    echo "🚀 Creating GitHub release $TAG..."
    gh release create "$TAG" -t "$TAG_MSG" -n "$DESCRIPTION"
fi

echo "✅ Done! Published $TAG"
