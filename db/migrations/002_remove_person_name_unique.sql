-- Migration 002: Remove unique constraint from persons.name
-- Created: 2025-11-15

-- Drop the unique constraint on persons.name
ALTER TABLE persons DROP CONSTRAINT IF EXISTS persons_name_key;
