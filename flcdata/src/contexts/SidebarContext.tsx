import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

type SidebarContextType = {
  hideSidebar: boolean;
  toggleSidebar: () => void;
};

const SidebarContext = createContext<SidebarContextType | null>(null);

export const useSidebarContext = (): SidebarContextType => {
  const context = useContext(SidebarContext);
  if (!context) {
    throw new Error('useSidebarContext must be used within a SidebarProvider');
  }
  return context;
};

interface SidebarProviderProps {
  children: ReactNode;
}

export const SidebarProvider: React.FC<SidebarProviderProps> = ({ children }) => {
  // Initialize with stored value or default to false (shown sidebar)
  const [hideSidebar, setHideSidebar] = useState<boolean>(() => {
    const stored = localStorage.getItem('app_sidebar_hidden');
    return stored ? JSON.parse(stored) : false;
  });

  // Save to localStorage when changed
  useEffect(() => {
    localStorage.setItem('app_sidebar_hidden', JSON.stringify(hideSidebar));
  }, [hideSidebar]);

  const toggleSidebar = () => {
    setHideSidebar(!hideSidebar);
  };

  const value: SidebarContextType = {
    hideSidebar,
    toggleSidebar,
  };

  return (
    <SidebarContext.Provider value={value}>
      {children}
    </SidebarContext.Provider>
  );
};
