import React from 'react';
import { Autocomplete, TextField } from '@mui/material';
import { useConfigContext } from '../contexts/ConfigContext';
import { useGeneratorContext } from '../contexts/GeneratorContext';
import { Module } from '../backend/interfaces';
import { ModuleRenderer } from './ModuleRenderer';

interface ModuleSelectorProps {
  category: 'data_generator' | 'distribution' | 'partitioner';
}

const ModuleSelector: React.FC<ModuleSelectorProps> = ({ category }) => {
  const config = useConfigContext();
  const generatorContext = useGeneratorContext();
  
  // Map category to the corresponding state and setters
  const getCategoryDetails = () => {
    switch (category) {
      case 'data_generator':
        return {
          options: config.data_generators,
          label: 'Data Generators',
          value: generatorContext.dataGenerator,
          params: generatorContext.dataGeneratorParams,
          setValue: generatorContext.setDataGenerator,
          setParams: generatorContext.setDataGeneratorParams,
        };
      case 'distribution':
        return {
          options: config.distributions,
          label: 'Distributions',
          value: generatorContext.distribution,
          params: generatorContext.distributionParams,
          setValue: generatorContext.setDistribution,
          setParams: generatorContext.setDistributionParams,
        };
      case 'partitioner':
        return {
          options: config.partitioners,
          label: 'Partitioners',
          value: generatorContext.partitioner,
          params: generatorContext.partitionerParams,
          setValue: generatorContext.setPartitioner,
          setParams: generatorContext.setPartitionerParams,
        };
      default:
        throw new Error(`Unknown category: ${category}`);
    }
  };

  const { options, label, value, params, setValue, setParams } = getCategoryDetails();

  if (!value) return null;

  return (
    <>
      <Autocomplete
        size="small"
        disableClearable
        value={value}
        options={options}
        getOptionLabel={(option: Module) => option.name}
        renderInput={(params) => <TextField {...params} label={label} />}
        isOptionEqualToValue={(option: Module, value: Module) => 
          option.name === value.name
        }
        onChange={(_, newValue: Module | null) => {
          if (newValue) {
            setValue(newValue);
          }
        }}
      />
      
      <ModuleRenderer
        module={value}
        parameters={params}
        onChange={setParams}
      />
    </>
  );
};

export default ModuleSelector;