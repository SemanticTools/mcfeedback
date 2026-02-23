#!/usr/bin/env node
// Usage: node backup.mjs [comment words...]
// Creates: ./backup/YYYY-MM-DD_HH-MM-SS_word1_word2_word3/

import { cpSync, mkdirSync, readdirSync } from 'fs';
import { join } from 'path';

const EXCLUDED = new Set(['backup', 'attic', 'node_modules', '.git']);

const comment = process.argv.slice(2).join('_').replace(/\s+/g, '_') || 'manual_backup';
const now = new Date();
const timestamp = now.toISOString().replace(/T/, '_').replace(/:/g, '-').slice(0, 19);
const folderName = `${timestamp}_${comment}`;
const dest = join('./backup', folderName);

mkdirSync(dest, { recursive: true });

for (const entry of readdirSync('.', { withFileTypes: true })) {
  if (EXCLUDED.has(entry.name)) continue;
  const src = join('.', entry.name);
  cpSync(src, join(dest, entry.name), { recursive: true });
}

console.log(`Backup created: ${dest}`);
