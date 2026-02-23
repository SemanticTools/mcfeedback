#!/usr/bin/env node
// Usage: node restore.mjs <backup-folder-name>
// Lists available backups if no argument given.

import { cpSync, mkdirSync, readdirSync, rmSync, existsSync } from 'fs';
import { join } from 'path';
import { spawnSync } from 'child_process';

const EXCLUDED = new Set(['backup', 'attic', 'node_modules', '.git']);
const BACKUP_DIR = './backup';

if (!process.argv[2]) {
  if (!existsSync(BACKUP_DIR)) { console.log('No backups found.'); process.exit(0); }
  const list = readdirSync(BACKUP_DIR).sort();
  console.log('Available backups:');
  list.forEach(b => console.log(' ', b));
  process.exit(0);
}

const target = join(BACKUP_DIR, process.argv[2]);
if (!existsSync(target)) {
  console.error(`Backup not found: ${target}`);
  process.exit(1);
}

// Safety backup before restore
const result = spawnSync('node', ['backup.mjs', 'before_restore'], { stdio: 'inherit' });
if (result.status !== 0) { console.error('Pre-restore backup failed, aborting.'); process.exit(1); }

// Remove current files (excluding backup/attic/node_modules/.git)
for (const entry of readdirSync('.', { withFileTypes: true })) {
  if (EXCLUDED.has(entry.name)) continue;
  rmSync(join('.', entry.name), { recursive: true, force: true });
}

// Copy backup contents to project root
for (const entry of readdirSync(target, { withFileTypes: true })) {
  cpSync(join(target, entry.name), join('.', entry.name), { recursive: true });
}

console.log(`Restored from: ${target}`);
