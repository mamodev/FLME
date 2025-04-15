import React, {
  createContext,
  useContext,
  ReactNode,
  useState,
  useEffect,
} from 'react';
import { Config } from '../backend/interfaces';
import { useConfig } from '../api/useConfig';
import { isObject } from '../utils/deepEqual';

// Create a hash of the config for version checking
function hashConfig(config: any): string {
  const jsonString = JSON.stringify(config);
  let hash = 0;
  for (let i = 0; i < jsonString.length; i++) {
    const char = jsonString.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash).toString(36); // Convert to base36 for shorter string
}

type ConfigContextType = Config;

const ConfigContext = createContext<ConfigContextType | null>(null);

export const useConfigContext = (): ConfigContextType => {
  const context = useContext(ConfigContext);
  if (!context) {
    throw new Error('useConfigContext must be used within a ConfigProvider');
  }
  return context;
};

interface ConfigProviderProps {
  children: ReactNode;
}

const LOCAL_STORAGE_KEY = 'configCache';
const LOCAL_STORAGE_HASH_KEY = 'configHash';

const getCachedConfig = (): Config | null => {
  const cachedConfig = localStorage.getItem(LOCAL_STORAGE_KEY);
  const cachedHash = localStorage.getItem(LOCAL_STORAGE_HASH_KEY);

  if (cachedConfig && cachedHash) {
    try {
      const config = JSON.parse(cachedConfig) as Config;
      if (!isObject(config)) {
        console.warn('Cached config is not an object');
        return null;
      }

      const requiredKeys = [
        "data_generators",
        "distributions",
        "partitioners",
      ]

      for (const key of requiredKeys) {
        if (!(key in config)) {
          console.warn(`Cached config is missing required key: ${key}`);
          return null;
        }

        const el = config[key as keyof Config];

        if (!Array.isArray(el) || el.length === 0) {
          console.warn(`Cached config key "${key}" is not a non-empty array`);
          return null;
        }
      }

    } catch (error) {
      console.error('Error parsing cached config:', error);
      // Clear cache if parsing fails
      localStorage.removeItem(LOCAL_STORAGE_KEY);
      localStorage.removeItem(LOCAL_STORAGE_HASH_KEY);
      return null;
    }
  }

  return null;
};

export const ConfigProvider: React.FC<ConfigProviderProps> = ({
  children,
}) => {
  const [config, setConfig] = useState<Config | null>(getCachedConfig());
  const configQuery = useConfig();

  useEffect(() => {
    if (configQuery.data) {
      const newConfig = configQuery.data;
      const newHash = hashConfig(newConfig);
      const cachedHash = localStorage.getItem(LOCAL_STORAGE_HASH_KEY);

      if (newHash !== cachedHash) {
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(newConfig));
        localStorage.setItem(LOCAL_STORAGE_HASH_KEY, newHash);
        setConfig(newConfig);
      } else if (!config) {
        // If config is not yet set, use the configQuery data
        setConfig(newConfig);
      }
    }
  }, [configQuery.data, config]);

  return (
    <ConfigContext.Provider value={config as Config}>
      {!!config && children}
    </ConfigContext.Provider>
  );
};
