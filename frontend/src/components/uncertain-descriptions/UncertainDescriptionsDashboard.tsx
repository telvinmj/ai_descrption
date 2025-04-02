import React, { useEffect, useState, useCallback } from 'react';
import { Box, Typography, CircularProgress, Alert } from '@mui/material';
import { UncertainModels } from './UncertainModels';

export const UncertainDescriptionsDashboard: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Uncertain Models Dashboard
      </Typography>
      
      <UncertainModels />
    </Box>
  );
}; 