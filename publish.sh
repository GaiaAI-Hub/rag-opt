#!/usr/bin/env bash
set -euo pipefail

# Extract version from pyproject.toml
VERSION=$(sed -n 's/^version *= *"\([^"]*\)"/\1/p' pyproject.toml)

if [[ -z "$VERSION" ]]; then
    echo "❌ Could not extract version from pyproject.toml"
    exit 1
fi

TAG="v$VERSION"
TAG_MSG="V$VERSION"

echo "📦 Publishing version: $TAG"

git add .
git commit -m "Publish $TAG" || echo "✔ Nothing to commit"
git push

if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "⚠️ Tag $TAG already exists! Skipping tag creation."
else
    git tag -a "$TAG" -m "$TAG_MSG"
    git push origin "$TAG"
fi

echo "✅ Done! Published $TAG"
