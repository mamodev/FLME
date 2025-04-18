import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  ReactNode,
  useCallback,
  useRef,
} from 'react';
import { useConfigContext } from './ConfigContext';
import { Module } from '../backend/interfaces';
import { GenerateRequest, useGenerate } from '../api/useGenerate';
import { SaveRequest, useSave } from '../api/useSave';

// Cache keys
const CACHE_VERSION_KEY = 'generator_cache_version';
const CACHE_DATA_KEY = 'generator_cache_data';
const CACHE_GENERATED_DATA_KEY = 'generator_generated_data'; // Key for generated data

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

// Interface for transformer with its parameters
interface TransformerWithParams {
  module: Module;
  params: Record<string, any>;
}

type GeneratorContextType = {
  // Module data (never null since config is loaded first)
  dataGenerator: Module;
  dataGeneratorParams: Record<string, any>;
  distribution: Module;
  distributionParams: Record<string, any>;
  partitioner: Module;
  partitionerParams: Record<string, any>;
  transformers: TransformerWithParams[];

  // Setters
  setDataGenerator: (module: Module) => void;
  setDataGeneratorParams: React.Dispatch<
    React.SetStateAction<Record<string, any>>
  >;
  setDistribution: (module: Module) => void;
  setDistributionParams: React.Dispatch<
    React.SetStateAction<Record<string, any>>
  >;
  setPartitioner: (module: Module) => void;
  setPartitionerParams: React.Dispatch<
    React.SetStateAction<Record<string, any>>
  >;

  // Transformer operations
  addTransformer: (module: Module) => void;
  removeTransformer: (index: number) => void;
  updateTransformerParams: (index: number, params: Record<string, any>) => void;
  moveTransformerUp: (index: number) => void;
  moveTransformerDown: (index: number) => void;

  // File save
  fileSave: string;
  setFileSave: React.Dispatch<React.SetStateAction<string>>;

  // Actions
  handleGenerate: () => void;
  handleSave: () => Promise<void> | void;

  // Generated data
  generated: any;

  // Loading states
  isGenerating: boolean;
  isSaving: boolean;
};

const GeneratorContext = createContext<GeneratorContextType | null>(null);

export const useGeneratorContext = (): GeneratorContextType => {
  const context = useContext(GeneratorContext);
  if (!context) {
    throw new Error(
      'useGeneratorContext must be used within a GeneratorProvider'
    );
  }
  return context;
};

function generateDefaultParameters(module: Module): Record<string, any> {
  const parameters: Record<string, any> = {};

  for (const [key, parameter] of Object.entries(module.parameters)) {
    if ('default' in parameter) {
      parameters[key] = parameter.default;
    } else {
      parameters[key] = null;
    }
  }

  return parameters;
}

interface GeneratorProviderProps {
  children: ReactNode;
}

// Helper to load cached data safely
function loadFromCache<T>(configHash: string, key: string, defaultValue: T): T {
  try {
    const cacheVersion = localStorage.getItem(CACHE_VERSION_KEY);
    if (cacheVersion === configHash) {
      const cachedData = JSON.parse(localStorage.getItem(CACHE_DATA_KEY) || '{}');
      if (cachedData[key] !== undefined) {
        return cachedData[key];
      }
    }
  } catch (e) {
    console.error(`Error loading ${key} from cache:`, e);
  }
  return defaultValue;
}

// Debounce function implementation
function debounce<Params extends any[]>(
  func: (...args: Params) => any,
  timeout: number
): {
  (...args: Params): void;
  flush: () => void;
} {
  let timer: NodeJS.Timeout | null = null;
  let lastArgs: Params | null = null;
  let lastThis: any; // Consider a more specific type if possible

  const debouncedFunction = function (this: any, ...args: Params): void {
    lastArgs = args;
    lastThis = this;

    if (timer) {
      clearTimeout(timer);
    }

    timer = setTimeout(() => {
      if (lastArgs) {
        func.apply(lastThis, lastArgs);
        lastArgs = null;
        timer = null; // Clear timer after execution
      }
    }, timeout);
  };

  debouncedFunction.flush = () => {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
    if (lastArgs) {
      func.apply(lastThis, lastArgs);
      lastArgs = null;
    }
  };

  return debouncedFunction;
}

export const GeneratorProvider: React.FC<GeneratorProviderProps> = ({
  children,
}) => {
  const config = useConfigContext();

  // Create a hash version of the config for cache validation
  const configHash = hashConfig(config);
  console.log('Config hash:', configHash);

  // Try to load cached state
  const [dataGenerator, setDataGeneratorInternal] = useState<Module>(() => {
    const cachedName = loadFromCache<string | null>(
      configHash,
      'dataGeneratorName',
      null
    );
    if (cachedName) {
      const found = config.data_generators.find((m) => m.name === cachedName);
      if (found) return found;
    }
    return config.data_generators[0];
  });

  const [dataGeneratorParams, setDataGeneratorParamsInternal] = useState<
    Record<string, any>
  >(() => {
    return loadFromCache(
      configHash,
      'dataGeneratorParams',
      generateDefaultParameters(dataGenerator)
    );
  });

  const [distribution, setDistributionInternal] = useState<Module>(() => {
    const cachedName = loadFromCache<string | null>(
      configHash,
      'distributionName',
      null
    );
    if (cachedName) {
      const found = config.distributions.find((m) => m.name === cachedName);
      if (found) return found;
    }
    return config.distributions[0];
  });

  const [distributionParams, setDistributionParamsInternal] = useState<
    Record<string, any>
  >(() => {
    return loadFromCache(
      configHash,
      'distributionParams',
      generateDefaultParameters(distribution)
    );
  });

  const [partitioner, setPartitionerInternal] = useState<Module>(() => {
    const cachedName = loadFromCache<string | null>(
      configHash,
      'partitionerName',
      null
    );
    if (cachedName) {
      const found = config.partitioners.find((m) => m.name === cachedName);
      if (found) return found;
    }
    return config.partitioners[0];
  });

  const [partitionerParams, setPartitionerParamsInternal] = useState<
    Record<string, any>
  >(() => {
    return loadFromCache(
      configHash,
      'partitionerParams',
      generateDefaultParameters(partitioner)
    );
  });

  // Initialize transformers state
  const [transformers, setTransformersInternal] = useState<TransformerWithParams[]>(() => {
    return loadFromCache<TransformerWithParams[]>(configHash, 'transformers', []);
  });

  const [fileSave, setFileSaveInternal] = useState<string>(() => {
    return loadFromCache(configHash, 'fileSave', 'data');
  });

  const [generated, setGeneratedInternal] = useState<any>(() => {
    try {
      const cacheVersion = localStorage.getItem(CACHE_VERSION_KEY);
      if (cacheVersion === configHash) {
        const cachedData = localStorage.getItem(CACHE_GENERATED_DATA_KEY);
        if (cachedData) {
          return JSON.parse(cachedData);
        }
      }
    } catch (e) {
      console.error('Error loading generated data from cache:', e);
    }
    return null; // Default value if not in cache or error
  });

  // Save cache function
  const saveToCache = useCallback(() => {
    try {
      const cacheData = {
        dataGeneratorName: dataGenerator.name,
        dataGeneratorParams,
        distributionName: distribution.name,
        distributionParams,
        partitionerName: partitioner.name,
        partitionerParams,
        transformers,
        fileSave,
      };

      localStorage.setItem(CACHE_VERSION_KEY, configHash);
      localStorage.setItem(CACHE_DATA_KEY, JSON.stringify(cacheData));
    } catch (e) {
      console.error('Error saving to cache:', e);
    }
  }, [
    configHash,
    dataGenerator,
    dataGeneratorParams,
    distribution,
    distributionParams,
    partitioner,
    partitionerParams,
    transformers,
    fileSave,
  ]);

  // Debounced save function to prevent excessive writes
  const debouncedSaveToCache = useRef(debounce(saveToCache, 300)).current;

  // Effect to save to cache when state changes
  useEffect(() => {
    debouncedSaveToCache();
    return () => {
      // Flush any pending saves on unmount
      debouncedSaveToCache.flush();
    };
  }, [
    debouncedSaveToCache,
    dataGenerator,
    dataGeneratorParams,
    distribution,
    distributionParams,
    partitioner,
    partitionerParams,
    transformers,
    fileSave,
  ]);

  // Effect to save generated data to cache
  useEffect(() => {
    try {
      if (generated) {
        localStorage.setItem(
          CACHE_GENERATED_DATA_KEY,
          JSON.stringify(generated)
        );
      } else {
        localStorage.removeItem(CACHE_GENERATED_DATA_KEY); // Remove if null
      }
    } catch (e) {
      console.error('Error saving generated data to cache:', e);
    }
  }, [generated, configHash]); // Save when generated data changes

  const generate = useGenerate((data) => {
    setGeneratedInternal(data);
  });

  const save = useSave(() => {
    alert('Saved');
  });

  // Wrapped setters that maintain cached state
  const setDataGenerator = (newValue: Module) => {
    setDataGeneratorInternal(newValue);
    setDataGeneratorParamsInternal(generateDefaultParameters(newValue));
  };

  const setDistribution = (newValue: Module) => {
    setDistributionInternal(newValue);
    setDistributionParamsInternal(generateDefaultParameters(newValue));
  };

  const setPartitioner = (newValue: Module) => {
    setPartitionerInternal(newValue);
    setPartitionerParamsInternal(generateDefaultParameters(newValue));
  };

  const setDataGeneratorParams = (
    newValue: React.SetStateAction<Record<string, any>>
  ) => {
    setDataGeneratorParamsInternal(newValue);
  };

  const setDistributionParams = (
    newValue: React.SetStateAction<Record<string, any>>
  ) => {
    setDistributionParamsInternal(newValue);
  };

  const setPartitionerParams = (
    newValue: React.SetStateAction<Record<string, any>>
  ) => {
    setPartitionerParamsInternal(newValue);
  };

  // Transformer operations
  const addTransformer = (module: Module) => {
    setTransformersInternal((prev) => [
      ...prev,
      { module, params: generateDefaultParameters(module) },
    ]);
  };

  const removeTransformer = (index: number) => {
    setTransformersInternal((prev) => prev.filter((_, i) => i !== index));
  };

  const updateTransformerParams = (index: number, params: Record<string, any>) => {
    setTransformersInternal((prev) => {
      const newTransformers = [...prev];
      newTransformers[index] = { ...newTransformers[index], params };
      return newTransformers;
    });
  };

  const moveTransformerUp = (index: number) => {
    if (index <= 0) return; // Can't move up if already at the top
    setTransformersInternal((prev) => {
      const newTransformers = [...prev];
      const temp = newTransformers[index];
      newTransformers[index] = newTransformers[index - 1];
      newTransformers[index - 1] = temp;
      return newTransformers;
    });
  };

  const moveTransformerDown = (index: number) => {
    setTransformersInternal((prev) => {
      if (index >= prev.length - 1) return prev; // Can't move down if already at the bottom
      const newTransformers = [...prev];
      const temp = newTransformers[index];
      newTransformers[index] = newTransformers[index + 1];
      newTransformers[index + 1] = temp;
      return newTransformers;
    });
  };

  const setFileSave = (newValue: React.SetStateAction<string>) => {
    setFileSaveInternal(newValue);
  };

  const handleGenerate = () => {
    const request: GenerateRequest = {
      data_generator: {
        name: dataGenerator.name,
        parameters: dataGeneratorParams,
      },
      distribution: {
        name: distribution.name,
        parameters: distributionParams,
      },
      partitioner: {
        name: partitioner.name,
        parameters: partitionerParams,
      },
      transformers: transformers.map(t => ({
        name: t.module.name,
        parameters: t.params
      })),
    };

    generate.mutate(request);
  };

  const handleSave = async () => {
    const request: SaveRequest = {
      data_generator: {
        name: dataGenerator.name,
        parameters: dataGeneratorParams,
      },
      distribution: {
        name: distribution.name,
        parameters: distributionParams,
      },
      partitioner: {
        name: partitioner.name,
        parameters: partitionerParams,
      },
      transformers: transformers.map(t => ({
        name: t.module.name,
        parameters: t.params
      })),
      file_name: fileSave,
    };

    await save.mutateAsync(request);
    return;
  };

  const value: GeneratorContextType = {
    // Module data
    dataGenerator,
    dataGeneratorParams,
    distribution,
    distributionParams,
    partitioner,
    partitionerParams,
    transformers,

    // Setters
    setDataGenerator,
    setDataGeneratorParams,
    setDistribution,
    setDistributionParams,
    setPartitioner,
    setPartitionerParams,

    // Transformer operations
    addTransformer,
    removeTransformer,
    updateTransformerParams,
    moveTransformerUp,
    moveTransformerDown,

    // File save
    fileSave,
    setFileSave,

    // Actions
    handleGenerate,
    handleSave,

    // Generated data
    generated,

    // Loading states
    isGenerating: generate.isPending,
    isSaving: save.isPending,
  };

  return (
    <GeneratorContext.Provider value={value}>{children}</GeneratorContext.Provider>
  );
};
