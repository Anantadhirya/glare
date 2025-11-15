-- Migration 001: Create initial tables for Person and Face
-- Created: 2025-11-15

-- Create persons table
CREATE TABLE IF NOT EXISTS persons (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create faces table with foreign key relationship
CREATE TABLE IF NOT EXISTS faces (
    id SERIAL PRIMARY KEY,
    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    image_url VARCHAR(512),
    embedding FLOAT8[512] NOT NULL,
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on person_id for faster queries
CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id);

-- Create index on verified for filtering
CREATE INDEX IF NOT EXISTS idx_faces_verified ON faces(verified);
