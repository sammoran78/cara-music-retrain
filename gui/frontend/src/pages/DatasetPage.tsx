import React from 'react';
import { DatasetDownloadConsole } from '../components/DatasetDownloadConsole';

export const DatasetPage: React.FC = () => {
  // The Dataset page intentionally reuses the existing
  // FREESOUND · MUSIC ATTRIBUTION POOL console without redesign.
  return <DatasetDownloadConsole />;
};
