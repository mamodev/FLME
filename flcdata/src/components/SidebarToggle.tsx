import React from 'react';
import { Box, IconButton } from '@mui/material';
import { ViewSidebarRounded } from '@mui/icons-material';
import { blue } from '@mui/material/colors';
import { useSidebarContext } from '../contexts/SidebarContext';

const SidebarToggle: React.FC = () => {
  const { hideSidebar, toggleSidebar } = useSidebarContext();
  
  if (!hideSidebar) {
    return null;
  }
  
  return (
    <Box
      sx={{
        position: "absolute",
        bottom: 8,
        left: 8,
        backgroundColor: blue[500],
        borderRadius: '50%',
        zIndex: 1000,
        boxShadow: 2,
      }}
    >
      <IconButton onClick={toggleSidebar}>
        <ViewSidebarRounded sx={{
          color: "white",
        }}/>
      </IconButton>
    </Box>
  );
};

export default SidebarToggle;
